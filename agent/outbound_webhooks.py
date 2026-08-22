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
  single daemon worker thread. ``invoke_hook()`` runs inside the agent
  loop, so callbacks must never block on network I/O — they serialize,
  enqueue, and return ``None`` immediately. Outbound targets can never
  block a tool call, inject context, or otherwise influence agent flow.
* The version-1 body is signed with HMAC-SHA256 when a secret reference
  resolves. ``X-Hermes-Signature-256`` remains for compatibility;
  ``X-Hermes-Signature-V2`` binds the raw ``X-Hermes-Timestamp`` value,
  a literal ``.`` byte, and the exact raw body.
* A configured ``secret_ref`` or compatibility ``secret_env`` is
  fail-closed. An empty, malformed, unscoped, or unresolved reference
  disables that target; Hermes never silently downgrades it to unsigned
  delivery. Unsigned delivery is possible only when neither reference
  field is present.
* Inline plaintext ``secret`` values are no longer accepted. Move the
  value to the environment or active profile secret scope and configure
  its name through ``secret_ref`` (preferred) or ``secret_env``.
* No consent prompt: unlike shell hooks, an outbound target executes no
  code on this machine — it POSTs JSON to a URL the user themselves put
  in config. ``HERMES_SAFE_MODE=1`` still skips registration, matching
  plugins / MCP / shell hooks.
* Registration is idempotent — safe to invoke from both the CLI entry
  point and the gateway entry point.

Config schema (``~/.hermes/config.yaml``)::

    hooks:
      outbound:
        - url: https://ci.example.com/hermes-events
          events: [on_session_end, subagent_stop]
          # Name of a secret in the active profile scope or environment.
          secret_ref: HERMES_OUTBOUND_WEBHOOK_SECRET
          # ``secret_env`` is accepted as a compatibility alias.
          matcher: "terminal|delegate_task"  # pre/post_tool_call only
          timeout: 10       # per-attempt seconds, clamped to [1, 60]
          name: ci-notify   # optional label for logs / `hermes hooks list`

Wire format (POST body)::

    {
        "schema_version":  1,
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
    User-Agent:              Hermes-Agent-Outbound-Webhook/1
    X-Hermes-Event:          <hook event name>
    X-Hermes-Delivery:       <delivery_id>
    X-Hermes-Schema-Version: 1
    X-Hermes-Timestamp:      <unix seconds>
    X-Hermes-Signature-256:  sha256=<HMAC(raw body)>  # compatibility
    X-Hermes-Signature-V2:   sha256=<HMAC(timestamp + b"." + raw body)>

Receiver verification contract
------------------------------
``X-Hermes-Signature-V2`` makes replay checks possible; the receiver must
complete them before performing side effects. Parse ``X-Hermes-Timestamp``
as integer Unix seconds and reject requests outside a bounded freshness
window (300 seconds is the recommended default, with clock skew included).
Compute HMAC-SHA256 over the exact timestamp header bytes, ``b"."``, and
the untouched request body, then compare the full ``sha256=...`` value with
``hmac.compare_digest``. Finally, accept each ``X-Hermes-Delivery`` value at
most once for at least the freshness window. A valid HMAC without freshness
and delivery-ID deduplication is authenticated but still replayable.
"""

from __future__ import annotations

import atexit
import hashlib
import hmac
import json
import logging
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

from agent.secret_scope import UnscopedSecretError, get_secret

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

_KNOWN_FIELDS = frozenset({
    "url",
    "events",
    "name",
    "secret_ref",
    "secret_env",
    "matcher",
    "timeout",
})
_SECRET_REFERENCE_FIELDS = ("secret_ref", "secret_env")

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

    ``cfg`` is the full parsed config dict. Missing, empty, or malformed
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
    Returns ``True`` when the queue fully drained. Test/shutdown helper."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _delivery_queue.all_tasks_done:
            if _delivery_queue.unfinished_tasks == 0:
                return True
        time.sleep(0.02)
    with _delivery_queue.all_tasks_done:
        return _delivery_queue.unfinished_tasks == 0


def reset_for_tests() -> None:
    """Clear the idempotence set and drain the queue. Test-only helper."""
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

    unknown = sorted(set(raw) - _KNOWN_FIELDS)
    if unknown:
        if "secret" in unknown:
            logger.error(
                "hooks.outbound[%d].secret contains unsupported inline "
                "plaintext. Move the value to the environment or active "
                "profile secret scope and set secret_ref (preferred) or "
                "secret_env to that name — target disabled",
                index,
            )
        other_unknown = [field for field in unknown if field != "secret"]
        if other_unknown:
            logger.warning(
                "hooks.outbound[%d] has unknown field(s) %s. Known fields: %s "
                "— target disabled",
                index,
                ", ".join(other_unknown),
                ", ".join(sorted(_KNOWN_FIELDS)),
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

    secret, usable = _resolve_secret(index, raw)
    if not usable:
        return None

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


def _resolve_secret(
    index: int, raw: Dict[str, Any],
) -> Tuple[Optional[str], bool]:
    """Resolve one optional secret reference.

    Returns ``(secret, usable)``. ``(None, True)`` means no secret field was
    configured and unsigned delivery was intentionally requested.
    ``(None, False)`` means a reference was configured but could not be used,
    so the caller must disable the target rather than downgrade it.
    """
    configured = [field for field in _SECRET_REFERENCE_FIELDS if field in raw]
    if not configured:
        return None, True
    if len(configured) != 1:
        logger.error(
            "hooks.outbound[%d] configures both secret_ref and secret_env. "
            "Configure exactly one reference field — target disabled",
            index,
        )
        return None, False

    field_name = configured[0]
    reference = raw.get(field_name)
    if not isinstance(reference, str) or not reference.strip():
        logger.error(
            "hooks.outbound[%d].%s must be a non-empty secret name — target "
            "disabled; Hermes will not send this webhook unsigned",
            index,
            field_name,
        )
        return None, False
    reference = reference.strip()

    try:
        value = get_secret(reference, "")
    except UnscopedSecretError as exc:
        logger.error(
            "hooks.outbound[%d] secret reference %r cannot be resolved "
            "without an active profile secret scope (%s) — target disabled; "
            "no process-environment fallback was attempted",
            index,
            reference,
            exc,
        )
        return None, False

    if value:
        return str(value), True

    logger.error(
        "hooks.outbound[%d] secret reference %r did not resolve — target "
        "disabled; Hermes will not send this webhook unsigned",
        index,
        reference,
    )
    return None, False


# ---------------------------------------------------------------------------
# Callback + delivery
# ---------------------------------------------------------------------------

def _make_callback(event: str, target: WebhookTarget):
    """Build the notify-only closure ``invoke_hook()`` calls per firing."""

    def _callback(**kwargs: Any) -> None:
        if event in _TOOL_SCOPED_EVENTS:
            if not target.matches_tool(kwargs.get("tool_name")):
                return None
        delivery_id = uuid.uuid4().hex
        try:
            body = _serialize_payload(event, kwargs, delivery_id)
        except Exception:  # defensive — a bad payload must not hurt the loop
            logger.warning(
                "outbound webhook payload serialization failed (event=%s "
                "target=%s)", event, target.label, exc_info=True,
            )
            return None
        _enqueue(_build_delivery(event, target, body, delivery_id))
        return None

    _callback.__name__ = f"outbound_webhook[{event}:{target.label}]"
    _callback.__qualname__ = _callback.__name__
    return _callback


def _serialize_payload(
    event: str, kwargs: Dict[str, Any], delivery_id: str,
) -> bytes:
    """Render the version-1 POST body.

    The shape mirrors shell hooks' stdin (documented in
    :mod:`agent.shell_hooks`) plus delivery metadata. ``delivery_id`` is shared
    with ``X-Hermes-Delivery`` so receivers can deduplicate. Replay protection
    is complete only when the receiver also validates the V2 timestamp and
    rejects a delivery ID it has already accepted.
    """
    extras = {k: v for k, v in kwargs.items() if k not in _TOP_LEVEL_PAYLOAD_KEYS}
    try:
        cwd = str(Path.cwd())
    except OSError:
        cwd = ""
    payload = {
        "schema_version": 1,
        "hook_event_name": event,
        "tool_name": kwargs.get("tool_name"),
        "tool_input": kwargs.get("args") if isinstance(kwargs.get("args"), dict) else None,
        "session_id": kwargs.get("session_id") or kwargs.get("parent_session_id") or "",
        "cwd": cwd,
        "extra": extras,
        "delivery_id": delivery_id,
        "timestamp": datetime.now(tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


def _build_delivery(
    event: str, target: WebhookTarget, body: bytes, delivery_id: str,
) -> Dict[str, Any]:
    timestamp = str(int(time.time()))
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Hermes-Agent-Outbound-Webhook/1",
        "X-Hermes-Event": event,
        "X-Hermes-Delivery": delivery_id,
        "X-Hermes-Schema-Version": "1",
        "X-Hermes-Timestamp": timestamp,
    }
    if target.secret:
        # Compatibility header for existing receivers.
        legacy = hmac.new(
            target.secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        headers["X-Hermes-Signature-256"] = f"sha256={legacy}"

        # V2 binds time + body so receivers can enforce bounded freshness.
        signed = timestamp.encode("ascii") + b"." + body
        digest = hmac.new(
            target.secret.encode("utf-8"), signed, hashlib.sha256
        ).hexdigest()
        headers["X-Hermes-Signature-V2"] = f"sha256={digest}"
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
        # The worker is a daemon thread, so a short-lived process (a `-q`
        # CLI run, a cron session) can exit right after enqueuing the
        # final events — silently dropping on_session_end, the headline
        # use case. Drain the queue at interpreter shutdown, bounded so
        # a dead endpoint can only delay exit, never hang it.
        atexit.register(flush, timeout=5.0)


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


class _NoRedirectHandler(urlrequest.HTTPRedirectHandler):
    """Refuse to follow redirects.

    urllib's default handler converts a redirected POST into a body-less
    GET — the signed payload would be silently dropped and the headers
    re-sent to a location the user never configured. Treat any 3xx as a
    delivery failure instead (surfaced as HTTPError by returning None).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


_opener = urlrequest.build_opener(_NoRedirectHandler)


def _deliver(delivery: Dict[str, Any]) -> None:
    """POST with bounded retries. Retries on connection errors and 5xx;
    4xx is the receiver telling us the request itself is wrong — no retry.
    3xx redirects are never followed (misconfiguration — fix the URL)."""
    last_error = ""
    for attempt in range(1, MAX_DELIVERY_ATTEMPTS + 1):
        req = urlrequest.Request(
            delivery["url"],
            data=delivery["body"],
            headers=delivery["headers"],
            method="POST",
        )
        try:
            with _opener.open(req, timeout=delivery["timeout"]) as resp:
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
            if 300 <= exc.code < 400:
                logger.warning(
                    "outbound webhook target redirected (event=%s target=%s): "
                    "%s -> %s — redirects are not followed; update the "
                    "configured url", delivery["event"], delivery["label"],
                    last_error, exc.headers.get("Location", "?"),
                )
                return
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
