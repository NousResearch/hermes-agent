"""A consumed routing note must not become a new task after compaction (#124170)."""

from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor, _SUMMARY_END_MARKER
from tests.agent.test_cron_inflight_prompt_reappend import _make_compressor, _tool_pairs, _handoff_idx, _text


NOTE = (
    "[Note: model was just switched from local/model to MiniMax-M3 via MiniMax. "
    "Adjust your self-identification accordingly.]"
)
TASK = "Inspect the next file and report the result."


@pytest.mark.parametrize("multimodal", [False, True])
def test_repeated_compactions_keep_the_task_and_images_without_replaying_the_note(multimodal):
    content = NOTE + "\n\n" + TASK
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    if multimodal:
        content = [{"type": "text", "text": content}, image]
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": content, "api_content": NOTE + "\n\n" + TASK},
        *_tool_pairs(40),
    ]
    for message in messages[2:]:
        if message["role"] == "assistant":
            message["content"] = NOTE + "\n\n" + message["content"]
    original = deepcopy(messages)
    compressor = _make_compressor()
    response = MagicMock()
    response.choices[0].message.content = "## Summary\nInspected earlier files."
    with patch("agent.context_compressor.call_llm", return_value=response) as summary_call:
        for cycle in range(3):
            compressed = compressor.compress(messages, current_tokens=200_000, force=True)
            assert NOTE not in str(compressed)
            assert NOTE not in str(summary_call.call_args)
            idx = _handoff_idx(compressed)
            assert idx >= 0
            assert TASK in (_text(compressed[idx]).split(_SUMMARY_END_MARKER)[-1] + str(compressed[idx + 1:]))
            assert compressed[0]["content"].startswith(original[0]["content"])
            # The first replay keeps the latest attachment; subsequent cycles obey
            # the existing historical-media policy rather than restoring old blobs.
            if multimodal and cycle == 0:
                assert any(image in (m.get("content") or []) for m in compressed if isinstance(m.get("content"), list))
            if cycle == 0:
                assert messages == original, "compaction must work on copies of the cached transcript"
            messages = compressed + _tool_pairs(40, start=(cycle + 1) * 40)


def test_static_fallback_does_not_promote_notes_but_keeps_tool_output_verbatim():
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": NOTE + "\n\n" + TASK},
        *_tool_pairs(40),
    ]
    for message in messages[2:]:
        if message["role"] == "assistant":
            message["content"] = NOTE + "\n\n" + message["content"]
    messages[-1]["content"] = NOTE
    original = deepcopy(messages)
    compressor = _make_compressor()
    compressor._fallback_compression_streak = 2
    with patch("agent.context_compressor.call_llm") as summary_call:
        compressed = compressor.compress(messages, current_tokens=200_000)
    summary_call.assert_not_called()
    assert compressor._last_summary_fallback_used
    assert NOTE not in str([m for m in compressed if m.get("role") != "tool"])
    assert any(m.get("role") == "tool" and m.get("content") == NOTE for m in compressed)
    assert TASK in str(compressed)
    assert messages == original


def test_aborted_summary_preserves_the_cached_transcript_including_the_notice():
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": NOTE + "\n\n" + TASK, "api_content": NOTE + "\n\n" + TASK},
        *_tool_pairs(40),
    ]
    original = deepcopy(messages)
    compressor = _make_compressor()
    response = MagicMock()
    response.choices[0].message.content = ""
    with patch("agent.context_compressor.call_llm", return_value=response):
        compressed = compressor.compress(messages, current_tokens=200_000, force=True)
    assert compressor._last_compress_aborted
    assert compressed == original
    assert messages == original


def test_no_op_compaction_keeps_the_cached_prompt_and_user_examples_unchanged():
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Explain this example:\n```text\n" + NOTE + "\n```", "api_content": NOTE + "\n\nExplain it."},
    ]
    original = deepcopy(messages)
    compressor = _make_compressor()
    assert compressor.compress(messages, current_tokens=20, force=True) == original
    assert messages == original
    snapshot = ContextCompressor._latest_user_task_snapshot(messages)
    assert NOTE in snapshot
