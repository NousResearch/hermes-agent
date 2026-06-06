"""Auto-resume freshness helpers for Hermes gateway restart recovery.

This module contains the timestamp parsing and freshness policy used to decide
whether an interrupted gateway turn should be auto-continued after restart or
on the user's next message.  The behavior is extracted from ``gateway.run`` as a
strangler seam; defaults, env-var names, malformed-value fallbacks, and legacy
"unknown timestamp is fresh" compatibility are intentionally unchanged.
"""

from __future__ import annotations

from datetime import datetime
import os
import time
from typing import Any, Optional

# Only auto-continue interrupted gateway turns while the interruption is fresh.
# Stale tool-tail/resume markers can otherwise revive an unrelated old task
# after a gateway restart when the user's next message starts new work.
#
# The freshness signal is the timestamp of the last transcript row, which
# ``hermes_state.get_messages`` carries on every persisted message. This handles
# the two auto-continue cases uniformly:
#   * resume_pending (gateway restart/shutdown watchdog marked the session)
#   * tool-tail     (last persisted message is a tool result the agent never got
#                    to reply to)
# In both cases "when did we last do anything on this transcript" is the correct
# freshness question, so one signal replaces two divergent ones.
#
# Default window: 1 hour. This comfortably covers ``agent.gateway_timeout``
# (30 min default) plus runtime slack — a legitimate long-running turn that gets
# interrupted near its timeout boundary and is resumed shortly after is still
# classified fresh. Override via
# ``config.yaml`` ``agent.gateway_auto_continue_freshness``.
AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT = 60 * 60
STARTUP_AUTO_RESUME_MAX_DEFAULT = 1


def coerce_gateway_timestamp(value: Any) -> Optional[float]:
    """Best-effort conversion of stored gateway timestamps to epoch seconds.

    Missing/unparseable timestamps return None so legacy transcripts keep the
    historical auto-continue behaviour instead of being silently dropped.
    Accepts: datetime, epoch seconds (int/float), epoch milliseconds (when
    the magnitude exceeds year-2286), ISO-8601 strings (with or without a
    trailing ``Z``), and numeric strings.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, bool):  # bool is a subclass of int — skip it
        return None
    if isinstance(value, (int, float)):
        # Some platform events use milliseconds; Hermes state rows use seconds.
        return float(value) / 1000.0 if float(value) > 10_000_000_000 else float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
            return numeric / 1000.0 if numeric > 10_000_000_000 else numeric
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def auto_continue_freshness_window() -> float:
    """Return the configured auto-continue freshness window in seconds.

    Reads ``HERMES_AUTO_CONTINUE_FRESHNESS`` (bridged from
    ``config.yaml`` ``agent.gateway_auto_continue_freshness`` at gateway
    startup, same pattern as ``HERMES_AGENT_TIMEOUT``). Falls back to the
    module default when unset or malformed. Non-positive values disable
    the freshness gate (restores the pre-fix "always fresh" behaviour for
    users who want to opt out).
    """
    raw = os.environ.get("HERMES_AUTO_CONTINUE_FRESHNESS")
    if raw is None or raw == "":
        return float(AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT)


def startup_auto_resume_max() -> int:
    """Return max synthetic resume turns scheduled at gateway startup.

    Real user messages still resume any remaining ``resume_pending`` sessions.
    The cap only prevents restart from immediately stampeding provider calls
    across every interrupted chat at once.
    """
    raw = os.environ.get("HERMES_STARTUP_AUTO_RESUME_MAX")
    if raw is None or raw == "":
        return STARTUP_AUTO_RESUME_MAX_DEFAULT
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return STARTUP_AUTO_RESUME_MAX_DEFAULT


def is_fresh_gateway_interruption(
    value: Any,
    *,
    now: Optional[float] = None,
    window_secs: Optional[float] = None,
) -> bool:
    """Return True when an interruption marker is fresh enough to auto-continue.

    Unknown timestamps are treated as fresh for backward compatibility with
    legacy transcripts (pre-dating timestamp persistence) and with in-memory
    test scaffolding that constructs history entries without timestamps.

    A non-positive ``window_secs`` disables the gate (always fresh), which
    restores the pre-fix behaviour for users who opt out via config.
    """
    window = (
        float(window_secs)
        if window_secs is not None
        else float(AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT)
    )
    if window <= 0:
        return True
    timestamp = coerce_gateway_timestamp(value)
    if timestamp is None:
        return True
    current = time.time() if now is None else now
    return current - timestamp <= window


def last_transcript_timestamp(history: Optional[list[dict[str, Any]]]) -> Any:
    """Return the ``timestamp`` of the last usable transcript row, if any.

    Skips metadata-only rows (``session_meta``, system injections) that are
    dropped before being handed to the agent. Returns ``None`` when no usable row
    carries a timestamp — callers should treat that as "fresh" for backward
    compatibility.
    """
    if not history:
        return None
    for msg in reversed(history):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if not role or role in {"session_meta", "system"}:
            continue
        ts = msg.get("timestamp")
        if ts is not None:
            return ts
        # First non-meta row without a timestamp — legacy transcript row.
        # Returning None lets the caller fall through to the legacy-fresh path.
        return None
    return None
