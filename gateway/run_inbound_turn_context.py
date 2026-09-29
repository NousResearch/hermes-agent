"""Chat-state notes that platform adapters add to inbound turns.

While a turn is prepared, ``BasePlatformAdapter.prepare_turn_context`` compares the chat with the
state saved on the most recent user transcript row and returns a note plus a new snapshot. The
note is prepended to the user message. The snapshot is saved on this turn's user row under
``display_metadata["channel_state"]``, so a change counts as acknowledged only once the turn that
reported it is in the transcript. ``display_metadata`` never reaches the model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gateway.platforms.base import BasePlatformAdapter

CHANNEL_STATE_METADATA_KEY = "channel_state"


def acknowledged_channel_state(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The channel state saved with the most recent user row that has one."""
    for message in reversed(history):
        if message.get("role") != "user":
            continue
        metadata = message.get("display_metadata")
        state = metadata.get(CHANNEL_STATE_METADATA_KEY) if isinstance(metadata, dict) else None
        if isinstance(state, dict):
            return state
    return None


def channel_state_metadata(event: Any) -> Dict[str, Any]:
    """The user row's ``display_metadata`` entry for the snapshot that *event* reported, if any."""
    state = getattr(event, "channel_state", None)
    return {} if state is None else {CHANNEL_STATE_METADATA_KEY: state}


def reports_turn_context(adapter: Any) -> bool:
    """Whether *adapter* overrides ``BasePlatformAdapter.prepare_turn_context``."""
    hook = getattr(type(adapter), "prepare_turn_context", None)
    return hook not in (None, BasePlatformAdapter.prepare_turn_context)


async def prepend_turn_context_note(
    runner: Any, *, event: Any, source: Any, session_key: str, history: List[Dict[str, Any]],
    message_text: str,
) -> str:
    """Ask the adapter that received *event* what changed in the chat, record its snapshot on the
    event and prepend its note to *message_text*."""
    adapter = runner._intake_adapter_for(source)
    if not reports_turn_context(adapter):
        return message_text
    entry = await runner.async_session_store.lookup_by_session_key(session_key)
    update = await adapter.prepare_turn_context(
        event, origin=entry.origin if entry else None,
        acknowledged_state=acknowledged_channel_state(history),
    )
    if update is None:
        return message_text
    event.channel_state = update.channel_state
    if not update.note:
        return message_text
    return f"{update.note}\n\n[New message]\n{message_text}"
