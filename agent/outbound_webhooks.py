"""
Outbound webhook notifications.

Reads the ``hooks.outbound:`` list from ``config.yaml`` and registers
notify-only callbacks on the existing plugin hook manager, so every
``invoke_hook()`` site can push lifecycle events to external HTTP
endpoints — CI systems, dashboards, other agents — with zero changes to
call sites and zero polling on the receiving end.

This is the outbound mirror of the inbound webhook platform
(``gateway/platforms/webhook.py``): inbound wakes Hermes when the world
changes; outbound tells the world when Hermes does something.

Design notes
------------
* Delivery is fire-and-forget through a bounded in-process queue and a
  single daemon worker thread.  ``invoke_hook()`` runs inside the agent
  loop, so callbacks must never block on network I/O — they serialize,
  enqueue, and return ``None`` immediately.  Outbound targets can never
  block a tool call, inject context, or otherwise influence agent flow.
* Payloads are signed with HMAC-SHA256 (GitHub-style
  ``X-Hermes-Signature-256: sha256=<hexdigest>`` over the raw body) when
  a secret is configured.  Receivers verify exactly like they verify
  GitHub webhooks.
* No consent prompt: unlike shell hooks, an outbound target executes no
  code on this machine — it POSTs JSON to a URL the user themselves put
  in config.  ``HERMES_SAFE_MODE=1`` still skips registration, matching
  plugins / MCP / shell hooks.
* Registration is idempotent — safe to invoke from both the CLI entry
  point and the gateway entry point.

Config schema (``~/.hermes/config.yaml``)::

    hooks:
      outbound:
        - url: https://ci.example.com/hermes-events
          events: [on_session_end, subagent_stop]
          # secret literal (discouraged) or env var name (preferred):
          secret_env: HERMES_OUTBOUND_WEBHOOK_SECRET
          # optional regex, honored for pre/post_tool_call only:
          matcher: "terminal|delegate_task"
          timeout: 10       # per-attempt seconds, clamped to [1, 60]
          name: ci-notify   # optional label for logs / `hermes hooks list`

Wire format (POST body)::

    {
        "hook_event_name": "on_session_end",
        "tool_name":       null,
        "tool_input":      null,
        "session_id":      "sess_abc123",
        "cwd":             "/home/user/project",
        "extra":           {...},          # event-specific kwargs
        "delivery_id":     "3f2c...",      # uuid4, unique per POST
        "timestamp":       "2026-07-22T14:00:00Z"
    }

Headers::

    Content-Type:            application/json
    User-Agent:              Hermes-Agent-Outbound-Webhook
    X-Hermes-Event:          <hook event name>
    X-Hermes-Delivery:       <delivery_id>
    X-Hermes-Signature-256:  sha256=<hmac hexdigest>   # only when secret set
"""

from __future__ import annotations

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10
MAX_TIMEOUT_SECONDS = 60
MAX_DELIVERY_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 1.0
QUEUE_MAX_SIZE = 256

# Events whose ``matcher`` field is honored (mirrors shell hooks).
_TOOL_SCOPED_EVENTS = {"pre_tool_call", "post_tool_call"}

# kwargs promoted to top-level payload keys (mirrors shell hooks wire).
_TOP_LEVEL_PAYLOAD_KEYS = {"tool_name", "args", "session_id", "parent_session_id"}

# (event, url) pairs already wired to the plugin manager in this process.
_registered: Set[Tuple[str, str]] = set()
_registered_lock = threading.Lock()

_delivery_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(
    maxsize=QUEUE_MAX_SIZE
)
_worker_lock = threading.Lock()
_worker: Optional[threading.Thread] = None


@dataclass
class WebhookTarget:
    """Parsed and validated representation of one ``hooks.outbound`` entry."""

    url: str
    events: List[str]
    name: str = ""
    secret: Optional[str] = None
    matcher: Optional[str] = None
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    compiled_matcher: Optional[re.Pattern] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.matcher, str):
            stripped = self.matcher.strip()
            self.matcher = stripped if stripped else None
        if self.matcher:
            try:
                self.compiled_matcher = re.compile(self.matcher)
            except re.error as exc:
                logger.warning(
                    "outbound webhook matcher %r is invalid (%s) — treating "
                    "as literal equality", self.matcher, exc,
                )
                self.compiled_matcher = None

    @property
    def label(self) -> str:
        return self.name or self.url

    def matches_tool(self, tool_name: Optional[str]) -> bool:
        if not self.matcher:
            return True
        if tool_name is None:
            return False
        if self.compiled_matcher is not None:
            return self.compiled_matcher.fullmatch(tool_name) is not None
        return tool_name == self.matcher


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_from_config(cfg: Optional[Dict[str, Any]]) -> List[WebhookTarget]:
    """Register every configured outbound webhook on the plugin manager.

    ``cfg`` is the full parsed config dict.  Missing, empty, or malformed
    ``hooks.outbound`` is treated as zero targets — config parsing never
    raises, because a broken webhook entry must not crash the agent.

    Returns the targets that ended up wired (deduplicated across repeat
    calls, so the CLI and gateway can both invoke this safely).
    """
    if not isinstance(cfg, dict):
        return []

    from utils import env_var_enabled

    if env_var_enabled("HERMES_SAFE_MODE"):
        logger.info("HERMES_SAFE_MODE=1 — outbound webhook registration skipped")
        return []

    hooks_cfg = cfg.get("hooks")
    targets = _parse_outbound_block(
        hooks_cfg.get("outbound") if isinstance(hooks_cfg, dict) else None
    )
    if not targets:
        return []

    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()

    registered: List[WebhookTarget] = []
    with _registered_lock:
        for target in targets:
            wired_any = False
            for event in target.events:
                key = (event, target.url)
                if key in _registered:
                    continue
                manager._hooks.setdefault(event, []).append(
                    _make_callback(event, target)
                )
                _registered.add(key)
                wired_any = True
                logger.info(
                    "outbound webhook registered: %s -> %s (matcher=%s, "
                    "timeout=%ds)",
                    event, target.label, target.matcher, target.timeout,
                )
            if wired_any:
                registered.append(target)

    return registered


def iter_configured_targets(cfg: Optional[Dict[str, Any]]) -> List[WebhookTarget]:
    """Parse ``hooks.outbound`` without registering anything.
    Used by ``hermes hooks list``."""
    if not isinstance(cfg, dict):
        return []
    hooks_cfg = cfg.get("hooks")
    return _parse_outbound_block(
        hooks_cfg.get("outbound") if isinstance(hooks_cfg, dict) else None
    )


def flush(timeout: float = 5.0) -> bool:
    """Block until all queued deliveries are done (or *timeout* elapses).
    Returns ``True`` when the queue fully drained.  Test/shutdown helper."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _delivery_queue.all_tasks_done:
            if _delivery_queue.unfinished_tasks == 0:
                return True
        time.sleep(0.02)
    with _delivery_queue.all_tasks_done:
        return _delivery_queue.unfinished_tasks == 0


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


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _parse_outbound_block(raw: Any) -> List[WebhookTarget]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        logger.warning(
            "hooks.outbound must be a list of webhook targets; got %s",
            type(raw).__name__,
        )
        return []

    targets: List[WebhookTarget] = []
    for i, entry in enumerate(raw):
        target = _parse_single_target(i, entry)
        if target is not None:
            targets.append(target)
    return targets


def _parse_single_target(index: int, raw: Any) -> Optional[WebhookTarget]:
    from hermes_cli.plugins import VALID_HOOKS

    if not isinstance(raw, dict):
        logger.warning(
            "hooks.outbound[%d] must be a mapping with 'url' and 'events' "
            "keys; got %s", index, type(raw).__name__,
        )
        return None

    url = raw.get("url")
    if not isinstance(url, str) or not url.strip():
        logger.warning("hooks.outbound[%d] is missing a non-empty 'url'", index)
        return None
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        logger.warning(
            "hooks.outbound[%d].url must be http(s); got %r — skipped",
            index, url,
        )
        return None
    if url.lower().startswith("http://"):
        logger.warning(
            "hooks.outbound[%d].url uses plain http:// — payloads (including "
            "tool inputs) travel unencrypted. Prefer https.", index,
        )

    events_raw = raw.get("events")
    if not isinstance(events_raw, list) or not events_raw:
        logger.warning(
            "hooks.outbound[%d] needs a non-empty 'events' list (valid: %s)",
            index, ", ".join(sorted(VALID_HOOKS)),
        )
        return None
    events: List[str] = []
    for ev in events_raw:
        if ev in VALID_HOOKS:
            events.append(ev)
        else:
            logger.warning(
                "hooks.outbound[%d]: unknown event %r ignored (valid: %s)",
                index, ev, ", ".join(sorted(VALID_HOOKS)),
            )
    if not events:
        logger.warning(
            "hooks.outbound[%d] has no valid events — skipped", index,
        )
        return None

    matcher = raw.get("matcher")
    if matcher is not None and not isinstance(matcher, str):
        logger.warning(
            "hooks.outbound[%d].matcher must be a string regex; ignoring",
            index,
        )
        matcher = None
    if matcher is not None and not any(e in _TOOL_SCOPED_EVENTS for e in events):
        logger.warning(
            "hooks.outbound[%d].matcher=%r will be ignored — matcher is only "
            "honored for pre_tool_call / post_tool_call.", index, matcher,
        )
        matcher = None

    timeout_raw = raw.get("timeout", DEFAULT_TIMEOUT_SECONDS)
    try:
        timeout = int(timeout_raw)
    except (TypeError, ValueError):
        logger.warning(
            "hooks.outbound[%d].timeout must be an int (got %r); using "
            "default %ds", index, timeout_raw, DEFAULT_TIMEOUT_SECONDS,
        )
        timeout = DEFAULT_TIMEOUT_SECONDS
    timeout = max(1, min(timeout, MAX_TIMEOUT_SECONDS))

    secret = _resolve_secret(index, raw)

    name = raw.get("name")
    if not isinstance(name, str):
        name = ""

    return WebhookTarget(
        url=url,
        events=events,
        name=name.strip(),
        secret=secret,
        matcher=matcher,
        timeout=timeout,
    )


def _resolve_secret(index: int, raw: Dict[str, Any]) -> Optional[str]:
    """``secret_env`` (env var name, preferred) wins over inline ``secret``."""
    secret_env = raw.get("secret_env")
    if isinstance(secret_env, str) and secret_env.strip():
        value = os.environ.get(secret_env.strip(), "")
        if value:
            return value
        logger.warning(
            "hooks.outbound[%d].secret_env=%r is not set in the environment "
            "— deliveries will be UNSIGNED", index, secret_env.strip(),
        )
        return None
    secret = raw.get("secret")
    if isinstance(secret, str) and secret:
        return secret
    return None


# ---------------------------------------------------------------------------
# Callback + delivery
# ---------------------------------------------------------------------------

def _make_callback(event: str, target: WebhookTarget):
    """Build the notify-only closure ``invoke_hook()`` calls per firing."""

    def _callback(**kwargs: Any) -> None:
        if event in _TOOL_SCOPED_EVENTS:
            if not target.matches_tool(kwargs.get("tool_name")):
                return None
        try:
            body = _serialize_payload(event, kwargs)
        except Exception:  # defensive — a bad payload must not hurt the loop
            logger.warning(
                "outbound webhook payload serialization failed (event=%s "
                "target=%s)", event, target.label, exc_info=True,
            )
            return None
        _enqueue(_build_delivery(event, target, body))
        return None

    _callback.__name__ = f"outbound_webhook[{event}:{target.label}]"
    _callback.__qualname__ = _callback.__name__
    return _callback


def _serialize_payload(event: str, kwargs: Dict[str, Any]) -> bytes:
    """Render the POST body.  Same top-level shape as shell hooks' stdin
    (documented in :mod:`agent.shell_hooks`), plus delivery metadata."""
    extras = {k: v for k, v in kwargs.items() if k not in _TOP_LEVEL_PAYLOAD_KEYS}
    try:
        cwd = str(Path.cwd())
    except OSError:
        cwd = ""
    payload = {
        "hook_event_name": event,
        "tool_name": kwargs.get("tool_name"),
        "tool_input": kwargs.get("args") if isinstance(kwargs.get("args"), dict) else None,
        "session_id": kwargs.get("session_id") or kwargs.get("parent_session_id") or "",
        "cwd": cwd,
        "extra": extras,
        "delivery_id": uuid.uuid4().hex,
        "timestamp": datetime.now(tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


def _build_delivery(
    event: str, target: WebhookTarget, body: bytes,
) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Hermes-Agent-Outbound-Webhook",
        "X-Hermes-Event": event,
        "X-Hermes-Delivery": uuid.uuid4().hex,
    }
    if target.secret:
        digest = hmac.new(
            target.secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        headers["X-Hermes-Signature-256"] = f"sha256={digest}"
    return {
        "url": target.url,
        "label": target.label,
        "event": event,
        "body": body,
        "headers": headers,
        "timeout": target.timeout,
    }


def _enqueue(delivery: Dict[str, Any]) -> None:
    _ensure_worker()
    try:
        _delivery_queue.put_nowait(delivery)
    except queue.Full:
        logger.warning(
            "outbound webhook queue full (%d pending) — dropping %s event "
            "for %s", QUEUE_MAX_SIZE, delivery["event"], delivery["label"],
        )


def _ensure_worker() -> None:
    global _worker
    if _worker is not None and _worker.is_alive():
        return
    with _worker_lock:
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(
            target=_worker_loop, name="outbound-webhooks", daemon=True,
        )
        _worker.start()


def _worker_loop() -> None:
    while True:
        delivery = _delivery_queue.get()
        try:
            if delivery is not None:
                _deliver(delivery)
        except Exception:  # pragma: no cover — defensive
            logger.warning(
                "outbound webhook delivery crashed (target=%s)",
                delivery.get("label") if isinstance(delivery, dict) else "?",
                exc_info=True,
            )
        finally:
            _delivery_queue.task_done()


def _deliver(delivery: Dict[str, Any]) -> None:
    """POST with bounded retries.  Retries on connection errors and 5xx;
    4xx is the receiver telling us the request itself is wrong — no retry."""
    last_error = ""
    for attempt in range(1, MAX_DELIVERY_ATTEMPTS + 1):
        req = urlrequest.Request(
            delivery["url"],
            data=delivery["body"],
            headers=delivery["headers"],
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=delivery["timeout"]) as resp:
                status = getattr(resp, "status", 200)
            if 200 <= status < 300:
                logger.debug(
                    "outbound webhook delivered: %s -> %s (HTTP %d)",
                    delivery["event"], delivery["label"], status,
                )
                return
            last_error = f"HTTP {status}"
        except urlerror.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if 400 <= exc.code < 500:
                logger.warning(
                    "outbound webhook rejected (event=%s target=%s): %s — "
                    "not retrying", delivery["event"], delivery["label"],
                    last_error,
                )
                return
        except Exception as exc:
            last_error = str(exc) or type(exc).__name__

        if attempt < MAX_DELIVERY_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    logger.warning(
        "outbound webhook delivery failed after %d attempt(s) (event=%s "
        "target=%s): %s",
        MAX_DELIVERY_ATTEMPTS, delivery["event"], delivery["label"], last_error,
    )
