"""Collapsing for provider fallback and provider auth-error notices.

A single provider outage can walk the whole fallback chain in under a
second.  Every hop emits its own user-facing "Model fallback: …" line and
every failed attempt can render its own "Provider authentication failed …"
reply, so an operator watching a chat surface sees a burst of near-identical
messages for what is one event.

``display.fallback_notifications`` (``on`` | ``collapse`` | ``off``) and
``display.fallback_notice_interval_seconds`` (default 3600) control that:

  ``on``       — unchanged: one notice per hop, one reply per failure.
  ``collapse`` — at most ONE user-facing notice per session per interval.
                 Hops inside the window are counted and folded into the next
                 allowed notice.  Provider auth-error replies are deduped by
                 error class over the same window.
  ``off``      — no user-facing notice at all; the log lines are unchanged.
                 A reply to a message somebody sent still gets the terse line
                 below: ``off`` silences notices, never an answer.

The default is ``on``, so behaviour is byte-identical unless configured, with
two deliberate exceptions: ``gateway/run.py``'s TurnRunner auth branch gains
one unconditional ``logger.warning`` (closing a log gap the other two auth
render sites already covered), and a whitespace-only notice is dropped in
every mode rather than emitted as a blank line.

Suppression is never silence on a reply to a person's own message.  The
collapse helpers return the :data:`SUPPRESSED_NOTICE` sentinel rather than
``""`` so a caller can tell "already told this session" apart from "not a
provider error".  Unsolicited status chatter resolves the sentinel to nothing
(:func:`resolve_status_notice`); a direct reply resolves it to a single terse
line (:func:`resolve_direct_reply`) so a person who asks a question during an
outage always gets an answer.  No caller may fall through to the raw provider
error text.

State lifetime: the auth-error windows here are module-level and process-local,
and the model-fallback window lives on the agent object (so it is per session,
via the gateway's session-keyed agent cache).  Both are in memory only.  A
gateway restart, or eviction of an agent from that cache, reopens the window
and allows one more notice.  That is the fail-loud direction — at worst one
extra notice per session, never a lost one — and it is why this is a display
knob rather than a delivery guarantee.
"""

from __future__ import annotations

import logging
import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

FALLBACK_NOTIFICATION_MODES = ("on", "collapse", "off")
DEFAULT_FALLBACK_NOTIFICATION_MODE = "on"
DEFAULT_FALLBACK_NOTICE_INTERVAL_SECONDS = 3600.0

# Bound the per-session auth-error dedupe map so a long-lived gateway that
# sees many session keys cannot grow it without limit.
_MAX_TRACKED_AUTH_SESSIONS = 512

_warned_config_values: set = set()
_warn_lock = threading.Lock()


def _warn_once(token: str) -> bool:
    """True the first time ``token`` is seen this process."""
    with _warn_lock:
        if token in _warned_config_values:
            return False
        _warned_config_values.add(token)
        return True


def reset_config_warning_state() -> None:
    """Forget which config warnings were already emitted (tests)."""
    with _warn_lock:
        _warned_config_values.clear()


def resolve_fallback_notification_mode(cfg: Any = None) -> str:
    """Return the validated ``display.fallback_notifications`` mode.

    An unknown value warns once per process (the sanitizer re-resolves on
    every provider error event, so a typo during an outage would otherwise
    spam the log) and falls back to ``on``, the same fail-open shape as
    ``display.background_process_notifications``.
    """
    raw = _display_value(cfg, "fallback_notifications")
    if raw is None or raw == "":
        return DEFAULT_FALLBACK_NOTIFICATION_MODE
    if isinstance(raw, bool):
        # A bool is a plausible user typo for an on/off knob; honour it.
        return "on" if raw else "off"
    mode = str(raw).strip().lower()
    if mode not in FALLBACK_NOTIFICATION_MODES:
        if _warn_once(f"mode:{mode}"):
            logger.warning(
                "Unknown display.fallback_notifications %r, defaulting to %r "
                "(valid: %s)",
                raw, DEFAULT_FALLBACK_NOTIFICATION_MODE,
                ", ".join(FALLBACK_NOTIFICATION_MODES),
            )
        return DEFAULT_FALLBACK_NOTIFICATION_MODE
    return mode


def resolve_fallback_notice_interval(cfg: Any = None) -> float:
    """Return ``display.fallback_notice_interval_seconds`` as a float.

    Non-numeric or non-positive values warn and fall back to the default.
    """
    raw = _display_value(cfg, "fallback_notice_interval_seconds")
    if raw is None or raw == "":
        return DEFAULT_FALLBACK_NOTICE_INTERVAL_SECONDS
    try:
        interval = float(raw)
    except (TypeError, ValueError):
        interval = -1.0
    if interval <= 0:
        if _warn_once(f"interval:{raw!r}"):
            logger.warning(
                "Invalid display.fallback_notice_interval_seconds %r, "
                "defaulting to %s",
                raw, DEFAULT_FALLBACK_NOTICE_INTERVAL_SECONDS,
            )
        return DEFAULT_FALLBACK_NOTICE_INTERVAL_SECONDS
    return interval


def _display_value(cfg: Any, key: str) -> Any:
    """Read ``display.<key>`` from ``cfg``, loading user config when omitted."""
    if cfg is None:
        try:
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly() or {}
        except Exception:
            return None
    try:
        display = cfg.get("display") or {}
        return display.get(key)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Fallback notices
# --------------------------------------------------------------------------

@dataclass
class FallbackNoticeState:
    """Per-session collapse window for model-fallback notices."""

    last_emitted_at: Optional[float] = None
    suppressed: int = 0

    def reset(self) -> None:
        self.last_emitted_at = None
        self.suppressed = 0


def _hhmm_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M")


def collapse_fallback_notices(
    notices: list,
    *,
    mode: str,
    interval: float,
    state: FallbackNoticeState,
    now: float,
    current_route: Optional[str] = None,
) -> list:
    """Return the notice lines to emit for one flush of buffered hops.

    ``on`` returns every hop unchanged.  ``off`` returns nothing.  ``collapse``
    returns at most one line: the first hop's notice while the window is open,
    or a folded "N further fallbacks since HH:MM UTC; now on <model>" line once
    hops have been suppressed.  ``state`` is mutated to carry the window.
    """
    lines = [str(n) for n in (notices or []) if str(n).strip()]
    if mode == "off" or not lines:
        return []
    if mode != "collapse":
        return lines

    if state.last_emitted_at is not None and (now - state.last_emitted_at) < interval:
        # Window still open — count the hops and fold them into the next notice.
        state.suppressed += len(lines)
        return []

    carried = state.suppressed
    extra = len(lines) - 1
    route = current_route or "the fallback model"
    if carried:
        since = _hhmm_utc(state.last_emitted_at or now)
        line = (
            f"⚠️ Model fallback: {carried + extra} further fallbacks since "
            f"{since} UTC; now on {route}."
        )
    else:
        line = lines[0]
        if extra:
            line = (
                f"{line.rstrip('.')} ({extra} further fallbacks; now on {route}.)"
            )
    state.last_emitted_at = now
    state.suppressed = 0
    return [line]


# --------------------------------------------------------------------------
# Provider auth-error replies
# --------------------------------------------------------------------------

@dataclass
class _AuthErrorWindow:
    seen: dict = field(default_factory=dict)  # class digest -> last emitted ts
    last_touched: float = 0.0


_auth_error_windows: dict = {}
_auth_error_lock = threading.Lock()

# Returned in place of a user-facing reply that was deliberately withheld.
# Control characters keep it out of any real provider text, and every consumer
# resolves it through ``resolve_direct_reply`` / ``resolve_status_notice``
# before delivery, so it can never reach a chat surface.
SUPPRESSED_NOTICE = "\x00hermes:notice-suppressed\x00"

# What a person sees when they message during an outage the window already
# reported.  One line, no raw provider detail, never empty.
TERSE_SUPPRESSED_REPLY = (
    "⚠️ Still failing on the provider error reported earlier; details in the "
    "gateway logs."
)


def is_suppressed_notice(value: Any) -> bool:
    """True when a collapse helper deliberately withheld a user-facing notice."""
    return isinstance(value, str) and value == SUPPRESSED_NOTICE


def resolve_direct_reply(reply: Any) -> str:
    """Resolve a collapsed reply for a message a person actually sent.

    A direct reply must never become silence, and must never fall through to
    the raw provider error, so suppression collapses to one terse line.
    """
    if is_suppressed_notice(reply):
        return TERSE_SUPPRESSED_REPLY
    return "" if reply is None else str(reply)


def resolve_status_notice(reply: Any) -> Optional[str]:
    """Resolve a collapsed notice for the unsolicited status path.

    This is the only path allowed to collapse to nothing: nobody asked for it.
    """
    if is_suppressed_notice(reply):
        return None
    text = "" if reply is None else str(reply)
    return text or None


def reset_provider_auth_error_state() -> None:
    """Drop all auth-error dedupe state (tests, and gateway restarts)."""
    with _auth_error_lock:
        _auth_error_windows.clear()


def provider_auth_error_reply(
    exc: Any,
    *,
    session_key: Any = None,
    mode: Optional[str] = None,
    interval: Optional[float] = None,
    now: Optional[float] = None,
    cfg: Any = None,
) -> str:
    """Render the user-facing provider auth-failure reply, collapsed.

    The single renderer for ``⚠️ Provider authentication failed: …`` — every
    gateway site that surfaces a provider auth failure to a user calls this so
    one credential outage cannot produce one reply per failed attempt.
    """
    return collapse_provider_error_reply(
        f"⚠️ Provider authentication failed: {exc}",
        session_key=session_key,
        error_class="provider_auth",
        mode=mode,
        interval=interval,
        now=now,
        cfg=cfg,
    )


def collapse_provider_error_reply(
    reply: str,
    *,
    session_key: Any = None,
    error_class: str = "provider_error",
    mode: Optional[str] = None,
    interval: Optional[float] = None,
    now: Optional[float] = None,
    cfg: Any = None,
) -> str:
    """Collapse a user-facing provider-error reply per session per window.

    Returns ``reply``, or :data:`SUPPRESSED_NOTICE` when this session has
    already been told about this error class inside the current window
    (``collapse``) or when notices are ``off``.  Callers keep their own log
    line either way, and must pass the result through
    :func:`resolve_direct_reply` (a reply to a person's own message) or
    :func:`resolve_status_notice` (unsolicited chatter) before delivery.  The
    sentinel means "do not repeat this to the user", never "nothing happened"
    and never "not a provider error".
    """
    if not reply:
        return reply
    if mode is None:
        mode = resolve_fallback_notification_mode(cfg)
    if mode == "off":
        return SUPPRESSED_NOTICE
    if mode != "collapse":
        return reply
    if interval is None:
        interval = resolve_fallback_notice_interval(cfg)
    ts = _time.time() if now is None else now

    key = str(session_key or "")
    # Keyed on the error CLASS alone, never the reply text: the auth sites
    # embed the raw exception, and provider 401/429 bodies carry per-request
    # ids and timestamps, so any text in the key means nothing ever dedupes.
    digest = str(error_class)
    with _auth_error_lock:
        window = _auth_error_windows.get(key)
        if window is None:
            window = _AuthErrorWindow()
            _auth_error_windows[key] = window
        last = window.seen.get(digest)
        window.last_touched = ts
        if last is not None and (ts - last) < interval:
            return SUPPRESSED_NOTICE
        window.seen[digest] = ts
        if len(_auth_error_windows) > _MAX_TRACKED_AUTH_SESSIONS:
            _prune_auth_error_windows_locked()
    return reply


def _prune_auth_error_windows_locked() -> None:
    """Drop the least recently touched sessions. Caller holds the lock.

    Eviction silently reopens the window for those sessions, so a box past the
    cap can emit one extra notice for an evicted session.  Same fail-loud
    direction as a restart; the cap is what keeps the map bounded.
    """
    stale = sorted(_auth_error_windows.items(), key=lambda kv: kv[1].last_touched)
    for key, _ in stale[: max(1, len(stale) // 4)]:
        _auth_error_windows.pop(key, None)
