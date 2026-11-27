"""Background idle compaction: the gateway runs ``compression.idle_compact_after_seconds`` on every route.

The housekeeping tick's cache-edge scan (``gateway/run_watchers.py``) asks :func:`idle_due` about each
cached chat whose prompt-cache edge claimed nothing, and an agent-cache eviction asks it at release. A due
chat runs through that same job (one per session key, the shared semaphore, the manual-compression
pipeline with its lease, fence and claim gate) as :data:`IDLE_ACTION`; :func:`idle_admitted` re-checks the
turn-start idle gates on the stored transcript before anything is summarised. So the summary is ready
when the user comes back instead of running in front of their first message.
"""

from __future__ import annotations

import time
from typing import Any

IDLE_ACTION = "idle"


def idle_policy_seconds(agent: Any) -> int:
    """The chat's idle policy the gateway stamped on *agent* at its last turn (0 = off)."""
    seconds = getattr(agent, "compression_idle_compact_after_seconds", 0)
    if getattr(agent, "compression_enabled", False) is not True or not isinstance(seconds, int):
        return 0
    return max(0, seconds)


def has_buffered_input(runner: Any, key: str) -> bool:
    """An adapter holds input for *key* that no turn has admitted yet (a pending message, an unflushed
    text batch, an in-flight dispatch) or the runner queued it: summarising now would race that turn."""
    groups = (getattr(runner, "adapters", None) or {}, *(getattr(runner, "_profile_adapters", None) or {}).values())
    for group in groups:
        for adapter in list(group.values()):
            if any(key in (getattr(adapter, name, None) or {}) for name in ("_pending_messages", "_active_sessions")):
                return True
            # Text batches are keyed by each adapter's own batch key (a sender or topic suffix, or the
            # bare chat id), not the session key: match the buffered event's session instead.
            batches = list((getattr(adapter, "_pending_text_batches", None) or {}).values())
            if any(getattr(event, "source", None) is not None and runner._session_key_for_source(event.source) == key
                   for event in batches):
                return True
    state = (runner.__dict__.get("_sessions") or {}).get(key)
    return bool(state is not None and state.conversation.queued_events)


def idle_due(runner: Any, key: str, agent: Any, now: float) -> bool:
    """*agent* (cached for *key*, no turn running) has been idle for its chat's policy and this idle
    stretch was not handled yet. Claiming latches the stretch on the agent's last activity, so a chat
    the gates refused, or whose summary failed, is not tried again until it has new activity."""
    idle_after = idle_policy_seconds(agent)
    last = getattr(agent, "_last_activity_ts", None)
    if idle_after <= 0 or not isinstance(last, (int, float)) or now - last < idle_after:
        return False
    stretches = runner.__dict__.setdefault("_idle_compaction_stretches", {})
    if stretches.get(key) == last or has_buffered_input(runner, key):
        return False
    stretches[key] = last
    return True


def forget_stretches(runner: Any, live_keys: set) -> None:
    """Drop the latch of every chat no longer cached (its next agent starts a new stretch)."""
    stretches = runner.__dict__.get("_idle_compaction_stretches") or {}
    for key in [k for k in stretches if k not in live_keys]:
        stretches.pop(key, None)


def idle_admitted(agent: Any, history: list) -> bool:
    """The turn-start idle gates (``idle_compaction_admission``) on the stored transcript: above the
    post-compaction floor, grown since the last compaction, no failure cooldown. The gap is measured
    from the transcript, so a row another process wrote since the claim counts as activity."""
    from agent.turn_context_compaction import (
        _codex_native_auto_compaction, idle_compaction_admission, idle_reference_timestamp,
    )

    # A Codex app-server thread compacts itself unless ``codex_app_server_auto: hermes``; the gateway's
    # compaction path would force it, as ``/compress`` does.
    if _codex_native_auto_compaction(agent):
        return False
    reference = idle_reference_timestamp(history, None) or getattr(agent, "_last_activity_ts", None) or time.time()
    gap = time.time() - reference
    system_prompt = getattr(agent, "_cached_system_prompt", None) or ""
    return idle_compaction_admission(agent, history, system_prompt, gap) is not None
