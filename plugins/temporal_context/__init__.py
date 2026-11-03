"""Temporal-context plugin — physical time grounding for long-running sessions.

The context assembly pipeline models conversation history as an atemporal
sequence: when a user returns hours or days later with "continue the
troubleshooting" or "revert what we did just now", the model has no perception
of how much physical time elapsed, so it misreads relative anchors and trusts
stale runtime state (ports, PIDs, scratch files) that may be long gone.

This plugin injects a compact temporal header once per user turn:

    <!-- system:temporal_context (ephemeral) -->
    [Current time: 2026-09-01 10:50 Tuesday]
    [Turn interval: 3h12m since your previous turn]
    <!-- end_temporal_context -->

It is delivered through the ``pre_llm_call`` hook, so the text is appended to
the *current user message* at API-call time only — never persisted to the
session DB, never placed in the system prompt (the prompt-cache prefix stays
byte-stable across turns). This is a self-contained, opt-in first cut of the
idea in RFC #99942; the full RFC additionally proposes core session-store
timestamps and multi-day timeline chunking.

Note on the interval metric: it measures wall-clock time between the first LLM
call of consecutive *user* turns for a session. That includes the previous
turn's own agent-execution time, so for back-to-back turns it over-reads by a
few seconds; for the return-from-idle case this plugin exists to serve (gaps of
minutes to days) that overhead is negligible. A precise
``user_sent_at - agent_finished_at`` delta needs the core timestamps the RFC
proposes and is intentionally out of scope here.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Optional

# session_id -> unix timestamp of that session's previous user turn.
# Process-local and best-effort: a gateway restart resets it (the next turn is
# simply reported as the first in the session), which is the safe direction.
_last_user_turn_at: "dict[str, float]" = {}
_lock = threading.Lock()


def _humanize(seconds: float) -> str:
    """Render an elapsed duration compactly: 45s, 12m, 3h12m, 2d4h."""
    total = int(max(0, seconds))
    if total < 60:
        return f"{total}s"
    minutes, _ = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, rem_min = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{rem_min}m" if rem_min else f"{hours}h"
    days, rem_hours = divmod(hours, 24)
    return f"{days}d{rem_hours}h" if rem_hours else f"{days}d"


def _temporal_header(session_id: str, now: float) -> str:
    """Build the header for ``session_id`` at wall-clock ``now`` and advance its
    last-turn marker. Split out from :func:`on_pre_llm_call` so the delta logic
    is deterministically testable with injected timestamps."""
    stamp = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M %A")
    with _lock:
        previous = _last_user_turn_at.get(session_id)
        _last_user_turn_at[session_id] = now
    if previous is None or now < previous:
        interval = "first turn in this session"
    else:
        interval = f"{_humanize(now - previous)} since your previous turn"
    return (
        "<!-- system:temporal_context (ephemeral) -->\n"
        f"[Current time: {stamp}]\n"
        f"[Turn interval: {interval}]\n"
        "<!-- end_temporal_context -->"
    )


def on_pre_llm_call(
    *,
    session_id: str = "",
    turn_type: str = "user",
    api_call_count: int = 0,
    **kwargs: Any,
) -> Optional[dict]:
    """Inject the temporal header on the first LLM call of each user turn.

    Skips internal tool-loop iterations (``api_call_count > 0``) so the header
    is not repeated within a turn and the interval clock is measured per user
    turn, and skips non-user turns (cron / goal continuations) whose cadence is
    machine-driven rather than a human coming back from idle.
    """
    if turn_type != "user" or api_call_count:
        return None
    return {"context": _temporal_header(session_id or "", time.time())}


def register(ctx: Any) -> None:
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
