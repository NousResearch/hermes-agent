"""Outbound webhooks: ``hooks.outbound`` entries (url, events, secret_env|secret, matcher for
pre/post_tool_call, timeout clamped to [1, 60], name) -> notify-only callbacks on the plugin hook
manager, so every ``invoke_hook()`` site can POST lifecycle events (mirror of
``gateway/platforms/webhook.py``).  Fire-and-forget through a bounded queue + one daemon worker,
so a target can never block a tool call or influence agent flow.  HMAC-SHA256 signed
(``X-Hermes-Signature-256: sha256=<hex>`` over the raw body) when a secret is configured;
``HERMES_SAFE_MODE=1`` skips registration; registration is idempotent.
"""

from __future__ import annotations

import atexit
import hashlib
import hmac
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

from agent.shell_hooks import (
    _TOOL_EVENTS as _TOOL_SCOPED_EVENTS,
    _ToolMatcherMixin,
    _forget_home_registrations,
    _home_key,
    _payload_fields,
    _utc_now_iso,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10
MAX_TIMEOUT_SECONDS = 60
MAX_DELIVERY_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 1.0
QUEUE_MAX_SIZE = 256

# (home, event, url) triples already wired in this process. Home is part of the key so a
# multiplexed gateway's secondary profiles (own plugin managers) can register identical targets.
_registered: Set[Tuple[str, str, str]] = set()
_registered_lock = threading.Lock()

_delivery_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=QUEUE_MAX_SIZE)
_worker_lock = threading.Lock()
_worker: Optional[threading.Thread] = None


@dataclass
class WebhookTarget(_ToolMatcherMixin):
    """Parsed and validated representation of one ``hooks.outbound`` entry."""
    _MATCHER_KIND = "outbound webhook"
    url: str
    events: List[str]
    name: str = ""
    secret: Optional[str] = None
    matcher: Optional[str] = None
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    compiled_matcher: Optional[re.Pattern] = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return self.name or self.url


def register_from_config(cfg: Optional[Dict[str, Any]]) -> List[WebhookTarget]:
    """Register every configured outbound webhook on the plugin manager.  Malformed ``hooks.outbound``
    means zero targets — never raises.  Returns the targets that ended up wired (deduplicated)."""
    if not isinstance(cfg, dict):
        return []
    from utils import env_var_enabled
    if env_var_enabled("HERMES_SAFE_MODE"):
        logger.info("HERMES_SAFE_MODE=1 — outbound webhook registration skipped")
        return []
    targets = iter_configured_targets(cfg)
    if not targets:
        return []
    from hermes_cli.plugins import get_plugin_manager
    manager = get_plugin_manager()
    home_key = _home_key()
    registered: List[WebhookTarget] = []
    with _registered_lock:
        for target in targets:
            wired_any = False
            for event in target.events:
                key = (home_key, event, target.url)
                if key in _registered:
                    continue
                manager._hooks.setdefault(event, []).append(_make_callback(event, target))
                _registered.add(key)
                wired_any = True
                logger.info(
                    "outbound webhook registered: %s -> %s (matcher=%s, timeout=%ds)",
                    event, target.label, target.matcher, target.timeout,
                )
            if wired_any:
                registered.append(target)
    return registered


def iter_configured_targets(cfg: Optional[Dict[str, Any]]) -> List[WebhookTarget]:
    """Parse ``hooks.outbound`` without registering anything (``hermes hooks list``)."""
    if not isinstance(cfg, dict):
        return []
    hooks_cfg = cfg.get("hooks")
    raw = hooks_cfg.get("outbound") if isinstance(hooks_cfg, dict) else None
    if raw is None:
        return []
    if not isinstance(raw, list):
        logger.warning("hooks.outbound must be a list of webhook targets; got %s", type(raw).__name__)
        return []
    return [t for t in (_parse_single_target(i, entry) for i, entry in enumerate(raw)) if t is not None]


def flush(timeout: float = 5.0) -> bool:
    """Block until all queued deliveries are done (or *timeout* elapses); True if drained."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _delivery_queue.all_tasks_done:
            if _delivery_queue.unfinished_tasks == 0:
                return True
        time.sleep(0.02)
    with _delivery_queue.all_tasks_done:
        return _delivery_queue.unfinished_tasks == 0


def re_register_config_hooks() -> None:
    """Re-register outbound webhooks after a plugin force-reload cleared ``_hooks``.  Only the
    current home's idempotence keys are cleared so a force-reload in one profile cannot
    invalidate another profile's still-live registration.

    Mirrors ``agent.shell_hooks.re_register_config_hooks``: config-owned outbound-webhook callbacks live in
    the same ``_hooks`` dict that ``PluginManager.discover_and_load(force=True)`` clears via ``unload()``,
    so without this the force-reloaded profile's outbound webhooks go silently inert (#92682 review).
    """
    from hermes_cli.config import load_config
    _forget_home_registrations(_registered, _registered_lock)
    register_from_config(load_config())


def reset_for_tests() -> None:
    """Clear the idempotence set and drain the queue.  Test-only helper."""
    with _registered_lock:
        _registered.clear()
    try:
        while True:
            _delivery_queue.get_nowait()
            _delivery_queue.task_done()
    except queue.Empty:
        pass


def _parse_single_target(index: int, raw: Any) -> Optional[WebhookTarget]:
    from hermes_cli.plugins import VALID_HOOKS

    def warn(msg: str, *args: Any) -> None:
        logger.warning("hooks.outbound[%d]" + msg, index, *args)

    if not isinstance(raw, dict):
        warn(" must be a mapping with 'url' and 'events' keys; got %s", type(raw).__name__)
        return None
    url = raw.get("url")
    if not isinstance(url, str) or not url.strip():
        warn(" is missing a non-empty 'url'")
        return None
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        warn(".url must be http(s); got %r — skipped", url)
        return None
    if url.lower().startswith("http://"):
        warn(".url uses plain http:// — payloads (including tool inputs) travel unencrypted. Prefer https.")
    events_raw = raw.get("events")
    valid_list = ", ".join(sorted(VALID_HOOKS))
    if not isinstance(events_raw, list) or not events_raw:
        warn(" needs a non-empty 'events' list (valid: %s)", valid_list)
        return None
    events: List[str] = [ev for ev in events_raw if ev in VALID_HOOKS]
    for ev in events_raw:
        if ev not in VALID_HOOKS:
            warn(": unknown event %r ignored (valid: %s)", ev, valid_list)
    if not events:
        warn(" has no valid events — skipped")
        return None
    matcher = raw.get("matcher")
    if matcher is not None and not isinstance(matcher, str):
        warn(".matcher must be a string regex; ignoring")
        matcher = None
    if matcher is not None and not any(e in _TOOL_SCOPED_EVENTS for e in events):
        warn(".matcher=%r will be ignored — matcher is only honored for pre_tool_call / post_tool_call.", matcher)
        matcher = None
    timeout_raw = raw.get("timeout", DEFAULT_TIMEOUT_SECONDS)
    try:
        timeout = int(timeout_raw)
    except (TypeError, ValueError):
        warn(".timeout must be an int (got %r); using default %ds", timeout_raw, DEFAULT_TIMEOUT_SECONDS)
        timeout = DEFAULT_TIMEOUT_SECONDS
    name = raw.get("name")
    # ``secret_env`` (env var name, preferred) wins over inline ``secret``.
    secret_env = raw.get("secret_env")
    if isinstance(secret_env, str) and secret_env.strip():
        secret = os.environ.get(secret_env.strip(), "") or None
        if secret is None:
            warn(".secret_env=%r is not set in the environment — deliveries will be UNSIGNED", secret_env.strip())
    else:
        secret = raw.get("secret")
        secret = secret if isinstance(secret, str) and secret else None
    return WebhookTarget(
        url=url, events=events, name=name.strip() if isinstance(name, str) else "", secret=secret,
        matcher=matcher, timeout=max(1, min(timeout, MAX_TIMEOUT_SECONDS)),
    )


def _make_callback(event: str, target: WebhookTarget):
    """Build the notify-only closure ``invoke_hook()`` calls per firing."""

    def _callback(**kwargs: Any) -> None:
        if event in _TOOL_SCOPED_EVENTS and not target.matches_tool(kwargs.get("tool_name")):
            return
        delivery_id = uuid.uuid4().hex
        try:
            body = _serialize_payload(event, kwargs, delivery_id)
        except Exception:  # a bad payload must not hurt the loop
            logger.warning("outbound webhook payload serialization failed (event=%s target=%s)", event, target.label, exc_info=True)
            return
        _enqueue(_build_delivery(event, target, body, delivery_id))

    _callback.__name__ = f"outbound_webhook[{event}:{target.label}]"
    _callback.__qualname__ = _callback.__name__
    return _callback


def _serialize_payload(event: str, kwargs: Dict[str, Any], delivery_id: str) -> bytes:
    """Render the POST body: shell-hooks stdin shape plus delivery metadata.  ``delivery_id``
    (also the ``X-Hermes-Delivery`` header) and ``timestamp`` live inside the HMAC-signed
    body, so they double as replay protection."""
    # Profile resolved at fire time so a multiplexed gateway's receivers can tell which profile emitted.
    # See #92674.
    from hermes_cli.profiles import get_active_profile_name
    payload = {
        "hook_event_name": event, "profile": get_active_profile_name(), **_payload_fields(kwargs),
        "delivery_id": delivery_id, "timestamp": _utc_now_iso(),
    }
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


def _build_delivery(event: str, target: WebhookTarget, body: bytes, delivery_id: str) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json", "User-Agent": "Hermes-Agent-Outbound-Webhook",
        "X-Hermes-Event": event, "X-Hermes-Delivery": delivery_id,
    }
    if target.secret:
        digest = hmac.new(target.secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        headers["X-Hermes-Signature-256"] = f"sha256={digest}"
    return {"url": target.url, "label": target.label, "event": event, "body": body, "headers": headers, "timeout": target.timeout}


def _enqueue(delivery: Dict[str, Any]) -> None:
    global _worker
    if _worker is None or not _worker.is_alive():
        with _worker_lock:
            if _worker is None or not _worker.is_alive():
                _worker = threading.Thread(target=_worker_loop, name="outbound-webhooks", daemon=True)
                _worker.start()
                # Daemon worker: a short-lived process could exit right after enqueuing on_session_end.
                # Drain at interpreter shutdown, bounded so a dead endpoint can only delay exit, never hang it.
                atexit.register(flush, timeout=5.0)
    try:
        _delivery_queue.put_nowait(delivery)
    except queue.Full:
        logger.warning(
            "outbound webhook queue full (%d pending) — dropping %s event for %s",
            QUEUE_MAX_SIZE, delivery["event"], delivery["label"],
        )


def _worker_loop() -> None:
    while True:
        delivery = _delivery_queue.get()
        try:
            if delivery is not None:
                _deliver(delivery)
        except Exception:  # pragma: no cover — defensive
            label = delivery.get("label") if isinstance(delivery, dict) else "?"
            logger.warning("outbound webhook delivery crashed (target=%s)", label, exc_info=True)
        finally:
            _delivery_queue.task_done()


class _NoRedirectHandler(urlrequest.HTTPRedirectHandler):
    """Refuse redirects: urllib would turn a redirected POST into a body-less GET,
    silently dropping the signed payload. Any 3xx surfaces as HTTPError instead."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


_opener = urlrequest.build_opener(_NoRedirectHandler)


def _deliver(delivery: Dict[str, Any]) -> None:
    """POST with bounded retries: retry on connection errors and 5xx; 4xx and 3xx are final."""
    event, label = delivery["event"], delivery["label"]
    last_error = ""
    for attempt in range(1, MAX_DELIVERY_ATTEMPTS + 1):
        req = urlrequest.Request(delivery["url"], data=delivery["body"], headers=delivery["headers"], method="POST")
        try:
            with _opener.open(req, timeout=delivery["timeout"]) as resp:
                status = getattr(resp, "status", 200)
            if 200 <= status < 300:
                logger.debug("outbound webhook delivered: %s -> %s (HTTP %d)", event, label, status)
                return
            last_error = f"HTTP {status}"
        except urlerror.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if 300 <= exc.code < 400:
                logger.warning(
                    "outbound webhook target redirected (event=%s target=%s): %s -> %s — redirects are not "
                    "followed; update the configured url", event, label, last_error, exc.headers.get("Location", "?"),
                )
                return
            if 400 <= exc.code < 500:
                logger.warning("outbound webhook rejected (event=%s target=%s): %s — not retrying", event, label, last_error)
                return
        except Exception as exc:
            last_error = str(exc) or type(exc).__name__
        if attempt < MAX_DELIVERY_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    logger.warning(
        "outbound webhook delivery failed after %d attempt(s) (event=%s target=%s): %s",
        MAX_DELIVERY_ATTEMPTS, event, label, last_error,
    )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import Path  # noqa: F401,E402
from datetime import datetime  # noqa: F401,E402
from datetime import timezone  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
