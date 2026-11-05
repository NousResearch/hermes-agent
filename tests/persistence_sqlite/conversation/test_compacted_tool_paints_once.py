"""A compaction that carries a tool result forward as a shortened stub (same uid and timestamp) leaves every SQLite
display read painting that tool result once, as the archived full original, while the model keeps the stub."""

import pytest

from persistence.conversation.messages import ConversationRequest, NewMessage, TranscriptRead, TranscriptView
from persistence.conversation.sessions import NewSession

SID = "stub-display"
FULL = "FULL " * 1000
STUB = "[read_file] pruned stub"
METADATA = {"tool_result_metadata": {"lines": 1000}}
DISPLAY = TranscriptRead(view=TranscriptView.DISPLAY)


def seed(store) -> None:
    store.sessions.create(NewSession(SID, "test"))
    store.messages.append_batch(SID, [
        NewMessage("user", "read it", timestamp=100.0),
        NewMessage("assistant", "", timestamp=101.0,
                   tool_calls=({"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},)),
        NewMessage("tool", FULL, timestamp=102.0, tool_call_id="call-1", tool_name="read_file", display_metadata=METADATA),
        NewMessage("assistant", "done reading", timestamp=103.0),
    ])


def compact_with_stub(store, generation: int) -> None:
    """Compact the live history again, the tool result carried forward as the stub."""
    history = [m.as_openai_message() for m in store.messages.read_conversation(SID, ConversationRequest(include_row_ids=True))]
    next(m for m in history if m["role"] == "tool")["content"] = STUB
    store.messages.compact(SID, [NewMessage("user", f"summary {generation}", timestamp=150.0 + generation),
                                 *(NewMessage.from_live(m) for m in history if not m.get("_compressed_summary"))])


def _painted_tools(store) -> list[list]:
    resumed = store.messages.read_resume_conversations(SID).display
    paged = store.messages.read_transcript(SID, DISPLAY)
    shown = store.messages.read_conversation(SID, ConversationRequest(view=TranscriptView.DISPLAY))
    around = store.messages.read_prompt_page(SID, paged[0].id).messages
    return [[m.content for m in read if m.role == "tool"] for read in (resumed, paged, shown, around)]


@pytest.mark.parametrize("generations", [1, 2], ids=["once", "carried-again"])
def test_compacted_tool_result_paints_once_as_its_archived_original(backend, scope_a, generations):
    store = backend.open(scope_a)
    seed(store)
    for generation in range(1, generations + 1):
        compact_with_stub(store, generation)

    for bodies in _painted_tools(store):
        assert bodies == [FULL]
    assert store.messages.display_count(SID) == len(store.messages.read_transcript(SID, DISPLAY))
    model_tool = [m for m in store.messages.read_conversation(SID) if m.role == "tool"]
    assert [m.content for m in model_tool] == [STUB]
    assert model_tool[0].display_metadata == METADATA
