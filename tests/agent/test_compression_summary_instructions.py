"""compression.summary_instructions prompt contracts.

Unset keeps the default Be CONCRETE sentence. When set, that sentence is
replaced on batch compact and the same string is appended after the
NEVER/[REDACTED] block on micro-compact. The value is inserted verbatim.
"""

from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor

_DEFAULT_CONCRETE = (
    "Be CONCRETE — include file paths, command outputs, error messages, "
    "line numbers, and specific values. Avoid vague descriptions like "
    '"made some changes" — say exactly what changed.'
)
_CUSTOM = (
    "Preserve ticket IDs and exact error text. Omit {prices} and promos."
)


def _make_compressor():
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.protect_first_n = 2
    compressor.protect_last_n = 5
    compressor.context_length = 200_000
    compressor.threshold_percent = 0.80
    compressor.threshold_tokens = 160_000
    compressor.summary_target_ratio = 0.20
    compressor.tail_token_budget = 20_000
    compressor.max_summary_tokens = 10_000
    compressor.quiet_mode = True
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
    compressor.api_key = ""
    compressor.api_mode = "chat_completions"
    compressor.summary_instructions = ""
    return compressor


def _summary_response(content="## Goal\nCompaction complete."):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _turns():
    return [
        {"role": "user", "content": "Fix the auth bug"},
        {"role": "assistant", "content": "Fixed the JWT expiry check."},
    ]


def _capture_batch_prompt(compressor, **kwargs):
    prompts = []

    def mock_call_llm(**call_kwargs):
        prompts.append(call_kwargs["messages"][0]["content"])
        return _summary_response()

    with patch("agent.context_compressor.call_llm", mock_call_llm):
        result = compressor._generate_summary(_turns(), **kwargs)

    assert result is not None
    assert len(prompts) == 1
    return prompts[0]


def test_unset_keeps_default_concrete_sentence():
    compressor = _make_compressor()
    prompt = _capture_batch_prompt(compressor)

    assert _DEFAULT_CONCRETE in prompt
    assert _CUSTOM not in prompt


def test_set_replaces_concrete_and_keeps_surrounding_template():
    compressor = _make_compressor()
    compressor.summary_instructions = _CUSTOM
    prompt = _capture_batch_prompt(compressor)

    assert _CUSTOM in prompt
    assert _DEFAULT_CONCRETE not in prompt
    assert "Be CONCRETE" not in prompt
    assert "Target ~" in prompt
    assert "Write only the summary body" in prompt
    assert "## Goal" in prompt
    assert "[REDACTED]" in prompt
    # Verbatim: braces in the user string must not be interpolated.
    assert "{prices}" in prompt


def test_set_with_focus_topic_appends_focus_after_custom():
    compressor = _make_compressor()
    compressor.summary_instructions = _CUSTOM
    prompt = _capture_batch_prompt(compressor, focus_topic="authentication")

    assert _CUSTOM in prompt
    assert 'FOCUS TOPIC: "authentication"' in prompt
    assert prompt.index(_CUSTOM) < prompt.index('FOCUS TOPIC: "authentication"')


def test_micro_set_appends_after_redacted_keeps_merge_prose():
    compressor = _make_compressor()
    compressor.summary_instructions = _CUSTOM
    messages = compressor._build_micro_summary_prompt(
        "Prior rolling summary.",
        "user: open ticket T-42\nassistant: noted",
    )

    prompt = messages[1]["content"]
    assert _CUSTOM in prompt
    assert "Merge the exchange" in prompt
    assert "NEVER" in prompt
    assert "[REDACTED]" in prompt
    assert "Return ONLY the updated summary" in prompt
    assert prompt.index("[REDACTED]") < prompt.index(_CUSTOM)
    assert prompt.index(_CUSTOM) < prompt.index("## Current Running Summary")
    assert "{prices}" in prompt


def test_whitespace_only_is_unset():
    compressor = _make_compressor()
    compressor.summary_instructions = "   \n\t  "
    prompt = _capture_batch_prompt(compressor)

    assert _DEFAULT_CONCRETE in prompt
    assert _CUSTOM not in prompt


def test_none_and_non_string_are_unset():
    for value in (None, 12, ["not", "a", "string"]):
        compressor = _make_compressor()
        compressor.summary_instructions = value
        prompt = _capture_batch_prompt(compressor)
        assert _DEFAULT_CONCRETE in prompt
        assert _CUSTOM not in prompt
