"""The summarizer must see tool-result bodies Phase 1 stubbed out.

Phase 1 of ``compress()`` replaces bulky tool results with 1-line metadata
stubs, and the messages it stubs are the same ones the LLM pass is about to
summarize. Lean mode already reads around this via the pre-prune
``_pristine_tools`` snapshot; legacy mode did not, so ``_serialize_for_summary``
saw ``[read_file] read tasks.md from line 1 (12,362 chars)`` and could record
that a file was read but never what was in it.

Observed downstream: an agent whose file reads were stubbed mid-turn reported
"every read_file returns stubs, I cannot verify anything" and refused to answer.
"""

from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor


def _make_compressor(tail_mode="legacy"):
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.quiet_mode = True
    compressor.tail_mode = tail_mode
    return compressor


def _make_summarizing_compressor():
    """A compressor with just enough state to run ``_generate_summary``."""
    compressor = _make_compressor()
    compressor.protect_first_n = 2
    compressor.protect_last_n = 5
    compressor.tail_token_budget = 20000
    compressor.context_length = 200000
    compressor.threshold_percent = 0.80
    compressor.threshold_tokens = 160000
    compressor.summary_target_ratio = 0.20
    compressor.max_summary_tokens = 10000
    compressor.compression_count = 0
    compressor.last_prompt_tokens = 0
    compressor._previous_summary = None
    compressor._ineffective_compression_count = 0
    compressor._verify_compaction_cleared_threshold = False
    compressor._summary_failure_cooldown_until = 0.0
    compressor.summary_model = None
    compressor.model = "test-model"
    compressor.provider = "test"
    compressor.base_url = "http://localhost"
    compressor.api_key = "test-key"
    compressor.api_mode = "chat_completions"
    return compressor


def _tool_msg(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_serializer_prefers_the_pristine_body_over_a_stub():
    compressor = _make_compressor()
    body = "TASK BACKLOG\n" + ("georgian documents register line\n" * 50)
    stub = "[read_file] read tasks.md from line 1 (1,700 chars)"
    turns = [
        {"role": "user", "content": "what is on my plate?"},
        _tool_msg("call-1", stub),
    ]

    text = compressor._serialize_for_summary(turns, {"call-1": body})

    assert "georgian documents register line" in text
    assert stub not in text


def test_summary_prompt_carries_the_body_in_legacy_mode():
    """The end-to-end pin: what the LLM pass actually receives.

    Before this fix the legacy summarizer prompt contained the stub only, so
    the summary could say a file was read and never what it said.
    """
    compressor = _make_summarizing_compressor()
    compressor._pristine_tools = {
        "call-1": "TASK BACKLOG\n" + ("georgian documents register line\n" * 50),
    }
    turns = [
        {"role": "user", "content": "what is on my plate?"},
        _tool_msg("call-1", "[read_file] read tasks.md from line 1 (1,700 chars)"),
    ]

    captured = {}

    def mock_call_llm(**kwargs):
        captured["messages"] = kwargs["messages"]
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "## Goal\nReview the backlog."
        return resp

    with patch("agent.context_compressor.call_llm", mock_call_llm):
        assert compressor._generate_summary(turns) is not None

    prompt = "\n".join(m["content"] for m in captured["messages"])
    assert "georgian documents register line" in prompt


def test_serializer_without_a_snapshot_is_unchanged():
    """Every existing caller passes no snapshot and must keep old behaviour."""
    compressor = _make_compressor()
    turns = [_tool_msg("call-1", "[read_file] read tasks.md from line 1")]

    assert compressor._serialize_for_summary(turns) == (
        compressor._serialize_for_summary(turns, None)
    )
    assert "[read_file] read tasks.md" in compressor._serialize_for_summary(turns)


def test_snapshot_never_shortens_a_live_tool_result():
    """A result Phase 1 left alone must not be replaced by a stale snapshot."""
    compressor = _make_compressor()
    live = "full live output\n" + ("still here\n" * 50)
    turns = [_tool_msg("call-1", live)]

    text = compressor._serialize_for_summary(turns, {"call-1": "shorter old body"})

    assert "still here" in text
    assert "shorter old body" not in text


def test_restored_body_is_redacted_like_a_live_one():
    """Restoration happens before redaction, so secrets cannot ride it in."""
    compressor = _make_compressor()
    body = "here is the key sk-ant-api03-DEADBEEFDEADBEEFDEADBEEFDEADBEEF\n" * 5
    turns = [_tool_msg("call-1", "[terminal] ran `cat .env` -> exit 0")]

    text = compressor._serialize_for_summary(turns, {"call-1": body})

    assert "DEADBEEFDEADBEEFDEADBEEFDEADBEEF" not in text


def test_restored_body_is_truncated_per_message():
    """A recovered body cannot crowd the prompt: _CONTENT_MAX still applies."""
    compressor = _make_compressor()
    body = "x" * (ContextCompressor._CONTENT_MAX * 4)
    turns = [_tool_msg("call-1", "[read_file] read big.txt from line 1")]

    text = compressor._serialize_for_summary(turns, {"call-1": body})

    assert "[truncated]" in text
    assert len(text) < ContextCompressor._CONTENT_MAX * 2
