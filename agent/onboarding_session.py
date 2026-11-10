"""The onboarding chat's per-session pieces, shared by every runtime that hosts it (the session
authority's local turns and the old sidecar): the ``onboarding`` toolset fold-in, the scripted opening
and the per-turn note. Pure over the agent and the run record (``agent.onboarding_record``)."""

from __future__ import annotations

from typing import Any, Iterable, Iterator


def builds_onboarding(agent) -> bool:
    return "setup_beat" in (getattr(agent, "valid_tool_names", None) or ())


def keeps_toolsets(enabled) -> bool:
    """An explicit empty list is a zero-tool chat; None is unrestricted and still gets the fold-in."""
    return enabled is not None and (not enabled or "onboarding" in enabled)


def with_onboarding(enabled) -> list[str]:
    """*enabled* plus onboarding. None (every toolset outside the session-gated ones, which onboarding
    is) is spelled out by name: the agent's other toolset gates read the names, as they read None."""
    if enabled is None:
        from toolsets import TOOLSET_SESSION_PLATFORMS, get_all_toolsets
        enabled = [name for name in get_all_toolsets() if name not in TOOLSET_SESSION_PLATFORMS]
    return [*enabled, "onboarding"]


def onboarding_toolsets(enabled, db, session_id: str):
    """*enabled* plus ``onboarding`` for a chat whose run is in progress. Decided once, at agent build:
    the tool array is part of the cached prefix, so a finished run keeps setup_beat (as a refusal)
    until the session is rebuilt."""
    from agent.onboarding_flow import is_active
    from agent.onboarding_record import load
    if keeps_toolsets(enabled):
        return enabled
    return with_onboarding(enabled) if is_active(load(db, session_id)) else enabled


def onboarding_chat_toolsets(enabled, db, session_id: str):
    """The session authority's per-turn fold-in: *enabled* plus ``onboarding`` for a chat that is or was
    an onboarding chat. Keyed on the record existing, never on the run being open, because the
    authority resolves toolsets every turn and the tool array is part of the cached prefix: a run that
    ends must not change it mid-conversation (setup_beat answers a finished run with a refusal)."""
    from agent.onboarding_record import load
    if keeps_toolsets(enabled) or db is None:
        return enabled
    tip = db.get_compression_tip(session_id) or session_id
    return with_onboarding(enabled) if load(db, tip) is not None else enabled


def carry_onboarding_toolset(enabled, agent):
    """A tool refresh keeps the onboarding toolset the agent was built with."""
    if keeps_toolsets(enabled) or "onboarding" not in (getattr(agent, "enabled_toolsets", None) or ()):
        return enabled
    return with_onboarding(enabled)


def onboarding_prelude(agent, history: Iterable[dict]) -> Iterator[tuple[str, str, dict]] | None:
    """The opening's scripted step, greeting and name card, while the name beat was never shown. A
    greeting already in the chat (a relaunch) is not said twice."""
    from agent.onboarding_flow import PENDING, is_active
    from agent.onboarding_record import load
    from agent.onboarding_text import greeting
    if not builds_onboarding(agent):
        return None
    record = load(agent._session_db, agent.session_id)
    if record is None or not is_active(record) or record["beats"]["name"] != PENDING:
        return None
    hello = greeting((record.get("facts") or {}).get("locale"))
    said = any(m.get("role") == "assistant" and m.get("content") == hello for m in history)

    def opening():
        yield ("" if said else hello), "setup_beat", {"beat": "name"}
    return opening()


def onboarding_turn_note(agent, *, history_empty: bool, card_waiting: bool) -> str:
    """Before a user turn in the onboarding chat, while no card waits: which card is closed and how to
    bring it back."""
    from agent.onboarding_flow import is_active
    from agent.onboarding_record import load
    from agent.onboarding_text import turn_note
    if history_empty or card_waiting or not builds_onboarding(agent):
        return ""
    record = load(agent._session_db, agent.session_id)
    return (turn_note(record) or "") if record is not None and is_active(record) else ""


def append_turn_note(agent: Any, note: str) -> None:
    if note:
        prior = getattr(agent, "_gateway_turn_context_notes", "") or ""
        agent._gateway_turn_context_notes = f"{prior}\n\n{note}" if prior else note
