"""Contract cases for display reads of a compacted tool result: a compaction that carries a tool result forward as a shortened stub (same uid and timestamp) leaves every display read showing that result once, as its archived original, while the model's replay keeps the stub."""

from __future__ import annotations

import pytest

from persistence.conversation.messages import ConversationRequest, NewMessage, TranscriptRead, TranscriptView
from persistence.conversation.sessions import NewSession

SID = "stub-display-contract"
FULL = "FULL " * 1000
STUB = "[read_file] pruned stub"
METADATA = {"tool_result_metadata": {"lines": 1000}}
DISPLAY = TranscriptRead(view=TranscriptView.DISPLAY)


def _seed(store) -> None:
    store.sessions.create(NewSession(SID, "cli"))
    store.messages.append_batch(SID, [
        NewMessage("user", "read it", timestamp=100.0),
        NewMessage("assistant", "", timestamp=101.0,
                   tool_calls=({"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},)),
        NewMessage("tool", FULL, timestamp=102.0, tool_call_id="call-1", tool_name="read_file", display_metadata=METADATA),
        NewMessage("assistant", "done reading", timestamp=103.0),
    ])


def _compact_with_stub(store, generation: int) -> None:
    """Compact the live history again, the tool result carried forward as the stub."""
    history = [m.as_openai_message() for m in store.messages.read_conversation(SID, ConversationRequest(include_row_ids=True))]
    next(m for m in history if m["role"] == "tool")["content"] = STUB
    store.messages.compact(SID, [NewMessage("user", f"summary {generation}", timestamp=150.0 + generation),
                                 *(NewMessage.from_live(m) for m in history if not m.get("_compressed_summary"))])


@pytest.mark.parametrize("generations", [1, 2], ids=["once", "carried-again"])
def test_a_compacted_tool_result_is_displayed_once_as_its_archived_original(backend, scope_a, generations):
    store = backend.open(scope_a)
    _seed(store)
    for generation in range(1, generations + 1):
        _compact_with_stub(store, generation)

    paged = store.messages.read_transcript(SID, DISPLAY)
    reads = (
        store.messages.read_resume_conversations(SID).display,
        paged,
        store.messages.read_conversation(SID, ConversationRequest(view=TranscriptView.DISPLAY)),
        store.messages.read_prompt_page(SID, paged[0].id).messages,
    )
    for read in reads:
        assert [m.content for m in read if m.role == "tool"] == [FULL]
    assert store.messages.display_count(SID) == len(paged)
    model_tool = [m for m in store.messages.read_conversation(SID) if m.role == "tool"]
    assert [m.content for m in model_tool] == [STUB]
    assert model_tool[0].display_metadata == METADATA
