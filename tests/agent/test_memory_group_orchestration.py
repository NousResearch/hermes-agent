"""Bot Mode's generated room turns must not become human speech in memory (#141333)."""

from copy import deepcopy

import pytest

from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, SUMMARY_PREFIX, _SUMMARY_END_MARKER
from agent.conversation_compression import _direct_messages_for_pre_compress_memory
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


_ROOM_PROMPT = (
    '[Group chat: "Launch"] You are @builder, one participant in a group chat with @reviewer and the user.\n\n'
    'New messages in the room since your last turn (oldest first):\n'
    '  User (user): Is the release ready?\n'
    '  reviewer: Tests are green.\n\n'
    'Rules for this room:\n'
    '- If you have nothing new to add, reply with exactly "(pass)".\n'
    '- Never reveal content from your private 1:1 chats.'
)
_HUMAN = "My release notes should include migration steps."


class _Recorder(MemoryProvider):
    name = "recording"

    def __init__(self):
        self.seen = []

    def initialize(self, session_id="", **kwargs): pass
    def is_available(self): return True
    def get_tool_schemas(self): return []
    def prefetch(self, query, *, session_id=""): self.seen.append(query); return ""
    def queue_prefetch(self, query, *, session_id=""): self.seen.append(query)
    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        self.seen.append((user_content, assistant_content, messages))
    def on_turn_start(self, turn_number, message, **kwargs): self.seen.append(message)
    def on_session_end(self, messages): self.seen.append(messages)
    def on_pre_compress(self, messages, **kwargs): self.seen.append(messages); return ""


@pytest.fixture
def manager():
    manager, provider = MemoryManager(), _Recorder()
    manager.add_provider(provider)
    yield manager, provider
    manager.shutdown_all()


_TEXT_FANOUTS = {
    "prefetch": lambda m, text: m.prefetch_all(text),
    "queue_prefetch": lambda m, text: m.queue_prefetch_all(text),
    "sync": lambda m, text: m.sync_all(text, "The release is ready."),
    "turn_start": lambda m, text: m.on_turn_start(1, text),
}


@pytest.mark.parametrize("fanout", sorted(_TEXT_FANOUTS))
def test_room_harness_is_not_user_input_but_direct_input_still_reaches_providers(manager, fanout):
    manager, provider = manager
    send = _TEXT_FANOUTS[fanout]
    send(manager, _ROOM_PROMPT)
    assert manager.flush_pending(timeout=5)
    assert provider.seen == []

    send(manager, _HUMAN)
    assert manager.flush_pending(timeout=5)
    assert len(provider.seen) == 1
    assert _HUMAN in repr(provider.seen)
    if fanout == "turn_start":
        send(manager, "")
        assert provider.seen[-1] == ""


_TRANSCRIPT_FANOUTS = {
    "sync": lambda m, rows: m.sync_all(_HUMAN, "I will include them.", messages=rows),
    "session_end": lambda m, rows: m.on_session_end(rows),
    "pre_compress_v1": lambda m, rows: m.on_pre_compress(rows),
    "pre_compress_v2": lambda m, rows: m.on_pre_compress(
        rows, evidence_messages=_direct_messages_for_pre_compress_memory(rows), require_checkpoint=True),
}


@pytest.mark.parametrize("room_shape", ["text", "multimodal", "compacted"])
@pytest.mark.parametrize("fanout", sorted(_TRANSCRIPT_FANOUTS))
def test_transcript_ingestion_omits_room_exchange_and_keeps_human_turns_without_mutating_history(manager, fanout, room_shape):
    manager, provider = manager
    provider.pre_compress_checkpoint_api_version = 2 if fanout == "pre_compress_v2" else 1
    before = [{"role": "user", "content": "Use the release branch."},
              {"role": "assistant", "content": "I have checked it out."}]
    after = [{"role": "user", "content": _HUMAN},
             {"role": "assistant", "content": "I will include them."}]
    room_turn = {"role": "user", "content": _ROOM_PROMPT}
    if room_shape == "multimodal":
        room_turn["content"] = [{"type": "text", "text": _ROOM_PROMPT}]
    elif room_shape == "compacted":
        room_turn = {"role": "user", "content": f"{SUMMARY_PREFIX}\nEarlier room activity\n{_SUMMARY_END_MARKER}\n\n{_ROOM_PROMPT}",
                     COMPRESSED_SUMMARY_METADATA_KEY: True, "display_kind": "hidden"}
    rows = [*before,
            room_turn,
            {"role": "assistant", "tool_calls": [{"id": "room-tool", "function": {"name": "terminal"}}]},
            {"role": "tool", "tool_call_id": "room-tool", "content": "room-only build output"},
            {"role": "user", "content": "[System: A legacy untyped continuation.]"},
            {"role": "user", "content": "[System: Continue working on the room turn.]", "display_kind": "hidden"},
            {"role": "assistant", "content": "room-only reply"},
            *after]
    original = deepcopy(rows)

    _TRANSCRIPT_FANOUTS[fanout](manager, rows)
    assert manager.flush_pending(timeout=5)
    observed = provider.seen[0][2] if fanout == "sync" else provider.seen[0]
    assert observed == before + after
    assert rows == original
