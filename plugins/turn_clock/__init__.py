"""turn-clock: give the model a sense of "now" — and of elapsed time.

Injects a compact wall-clock stamp (plus elapsed time since the previous
message in this session) into the API copy of the current turn's user
message via the ``pre_llm_call`` hook. The hook returns
``{"context": "[Current time ...; last message ... ago]"}`` which
``agent/turn_context.py`` merges into the ephemeral user-content block
(see ``plugin_user_context``).

Properties:
- Ephemeral only: the stored transcript stays clean and the cached system
  prompt is untouched, so the provider prompt-cache prefix stays byte-stable.
- Per-turn fresh: the stamp is recomputed on every API call, so the model no
  longer has to guess whether something happened "this morning" or "3 days
  ago" — it can read the elapsed time right in the turn it is answering.
- TZ-aware: prefers ``hermes_time`` (Hermes' configured timezone), falls
  back to the system-local timezone.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from hermes_time import now as _hermes_now
except Exception:  # pragma: no cover - hermes_time is always present in-process
    _hermes_now = None


# session_id -> tz-aware timestamp of the previous turn
_last_ts: Dict[str, datetime] = {}
_lock = threading.Lock()


def _format_elapsed(seconds: float) -> str:
    """Human-readable English elapsed-time string, >= 60s granularity."""
    if seconds < 60:
        return "just now"
    m = int(seconds // 60)
    if m < 60:
        return f"{m} minute{'s' if m != 1 else ''} ago"
    h, rem = divmod(m, 60)
    if h < 24:
        return f"{h} hour{'s' if h != 1 else ''} {rem} minute{'s' if rem != 1 else ''} ago"
    d, rem = divmod(h, 24)
    return f"{d} day{'s' if d != 1 else ''} {rem} hour{'s' if rem != 1 else ''} ago"


def _now() -> datetime:
    if _hermes_now is not None:
        try:
            return _hermes_now()
        except Exception:
            pass
    return datetime.now().astimezone()


def _prev_turn_ts(conversation_history: Optional[List[Dict[str, Any]]]) -> Optional[datetime]:
    """Timestamp of the previous message in this session.

    The prologue appends the current message to ``conversation_history``
    before hooks fire (turn_context appends first, then invokes hooks), so
    ``[-1]`` is the current message and ``[-2]`` is the previous one. Message
    timestamps are restored from the store (``hermes_state._rows_to_conversation``),
    so they stay accurate across processes/restarts.
    """
    try:
        hist = conversation_history or []
        if len(hist) >= 2:
            ts = hist[-2].get("timestamp")
            if isinstance(ts, (int, float)) and ts > 0:
                return datetime.fromtimestamp(ts).astimezone()
    except (IndexError, TypeError, ValueError):
        pass
    return None


def _pre_llm_call(session_id: str = "", conversation_history: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> Dict[str, str]:
    key = session_id or "default"
    now = _now()
    prev = _prev_turn_ts(conversation_history or [])
    if prev is None:
        prev = _last_ts.get(key)  # fallback: in-process record of the previous turn
    elapsed = ""
    if prev is not None:
        delta = now - prev
        if delta.total_seconds() >= 60:
            elapsed = f"; last message {_format_elapsed(delta.total_seconds())} ago"
    with _lock:
        _last_ts[key] = now
    stamp = now.strftime("%Y-%m-%d %H:%M %Z")
    return {"context": f"[Current time {stamp}{elapsed}]"}


def register(ctx: Any) -> None:
    ctx.register_hook("pre_llm_call", _pre_llm_call)
