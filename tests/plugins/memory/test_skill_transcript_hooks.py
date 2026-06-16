"""Raw-transcript hooks must persist user instructions, never loaded skill bodies."""

from unittest.mock import Mock

import pytest

from agent.context_compressor import _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER
from plugins.memory.byterover import ByteRoverMemoryProvider
from plugins.memory.holographic import HolographicMemoryProvider
from plugins.memory.supermemory import SupermemoryMemoryProvider

INSTRUCTION = "I prefer tabs over spaces for indentation"
BODY = "The project uses an imaginary skill template database."
BARE = ('[IMPORTANT: The user has invoked the "work" skill, indicating they want '
        'you to follow its instructions. The full skill content is loaded below.]\n\n' + BODY)
SINGLE = BARE + '\nThe user has provided the following instruction alongside the skill invocation: ' + INSTRUCTION
BUNDLE = ('[IMPORTANT: The user has invoked the "work" skill bundle, with loaded skills.]'
          '\nUser instruction: ' + INSTRUCTION + '\n\n[Loaded as part of the work bundle]\n' + BODY)


@pytest.mark.parametrize("reply_count", [1, 11])
@pytest.mark.parametrize("kind", ["supermemory", "byterover"])
@pytest.mark.parametrize("content,instruction", [(SINGLE, INSTRUCTION), (BUNDLE, INSTRUCTION),
                                                 (BARE, None), (INSTRUCTION, INSTRUCTION)])
def test_transcript_hooks_capture_instruction_and_drop_bare_turn(kind, content, instruction, monkeypatch, reply_count):
    if kind == "supermemory":
        monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
        client = Mock()
        monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", lambda **kwargs: client)
        provider = SupermemoryMemoryProvider()
        provider.initialize("test-session")
        capture = client.ingest_conversation
        hook = provider.on_session_end
    else:
        provider = ByteRoverMemoryProvider({"auto_extract": True})
        capture = Mock()
        monkeypatch.setattr(provider, "_curate_in_background", capture)
        hook = provider.on_pre_compress
    messages = [{"role": "user", "content": content},
                *[{"role": "assistant", "content": "skill response"} for _ in range(reply_count)],
                {"role": "user", "content": "ordinary conversation continues"},
                {"role": "assistant", "content": "ordinary reply"}]
    hook(messages)
    payload = str(capture.call_args)
    assert "ordinary conversation continues" in payload
    assert "ordinary reply" in payload
    assert BODY not in payload
    assert ("skill response" in payload) == bool(instruction)
    if instruction and (kind == "supermemory" or reply_count == 1):
        assert instruction in payload
    # Bare invocations alone must not write anything.
    capture.reset_mock()
    hook([{"role": "user", "content": BARE}, {"role": "assistant", "content": "skill response"}])
    capture.assert_not_called()


@pytest.mark.parametrize("content,expected", [(SINGLE, [INSTRUCTION]), (BUNDLE, [INSTRUCTION]),
                                              (BARE, []), (INSTRUCTION, [INSTRUCTION])])
@pytest.mark.parametrize("merged", [False, True])
def test_holographic_extracts_only_user_facts(tmp_path, content, expected, merged):
    provider = HolographicMemoryProvider({"auto_extract": True, "hrr_dim": 64,
                                          "db_path": str(tmp_path / "memory.db")})
    provider.initialize("test-session")
    if merged:
        content = (_MERGED_PRIOR_CONTEXT_HEADER + content + _MERGED_SUMMARY_DELIMITER
                   + "The project uses a compaction summary database.")
    try:
        provider.on_session_end([{"role": "user", "content": content},
                                 {"role": "assistant", "content": BODY}])
        assert [fact["content"] for fact in provider._store.list_facts()] == expected
    finally:
        provider.shutdown()
