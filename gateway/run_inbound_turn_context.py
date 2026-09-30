"""Turn context that platform adapters add to inbound turns.

While a turn is prepared, ``BasePlatformAdapter.prepare_turn_context`` compares the chat with the
newest state saved in the transcript and returns a note plus a new snapshot. On a session's first
turn the note can also contain earlier messages, such as a thread's history. The note is prepended
to the user message, before the ``[New message]`` marker. The snapshot is saved on this turn's user
row under ``display_metadata["channel_state"]``, so a change counts as acknowledged only once the
turn that reported it is in the transcript. Compaction records the newest snapshot on the row that
replaces the rows it removes, which can be a summary row of any role. ``display_metadata`` never
reaches the model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.message_metadata import CHANNEL_STATE_METADATA_KEY, newest_channel_state
from gateway.platforms.base import BasePlatformAdapter


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
        acknowledged_state=newest_channel_state(history), first_turn=not history,
    )
    if update is None:
        return message_text
    if update.channel_state is not None:
        event.channel_state = update.channel_state
    if not update.note:
        return message_text
    return f"{update.note}\n\n[New message]\n{message_text}"
