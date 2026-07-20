"""Per-turn must-deliver notes (api_content sidecar), not system-prompt text.

Volatile facts that must reach the model for **this turn only** — group-reply
directives, auto-reset notes, first-contact intro, voice-channel change — are
staged here and consumed once when the local AIAgent is assembled.  They must
never be appended to the frozen session ``context_prompt`` (prompt-cache /
agent-cache stability).

Production stages notes in ``_handle_message_with_agent`` and consumes them in
the agent run path.  Unconsumed notes are discarded on handler exit / proxy so
they cannot leak into a later turn (account-ban risk from stale directives).
"""

from __future__ import annotations

from typing import Dict, List, MutableMapping, Optional


def set_pending_turn_sidecar_notes(
    store: MutableMapping[str, List[str]],
    session_key: str,
    notes: List[str],
) -> None:
    """Stage one-shot must-deliver notes for the next agent run."""
    if not session_key or not notes:
        return
    store[session_key] = list(notes)


def consume_pending_turn_sidecar_notes(
    store: Optional[MutableMapping[str, List[str]]],
    session_key: str,
) -> List[str]:
    """Pop and return staged notes for ``session_key`` (empty if none)."""
    if not session_key or not isinstance(store, dict):
        return []
    staged = store.pop(session_key, None)
    return list(staged) if isinstance(staged, list) else []


def join_turn_sidecar_notes(notes: List[str]) -> str:
    """Join staged notes for assignment to ``agent._gateway_turn_context_notes``."""
    return "\n\n".join(n for n in notes if n)
