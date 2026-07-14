"""Behavior contracts for memory-provider context in compression prompts."""

from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor


def _make_compressor():
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.protect_first_n = 2
    compressor.protect_last_n = 5
    compressor.tail_token_budget = 20_000
    compressor.context_length = 200_000
    compressor.threshold_percent = 0.80
    compressor.threshold_tokens = 160_000
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
    return compressor


def _summary_response(content="## Goal\nCompaction complete."):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def test_memory_context_injected_into_initial_summary_prompt_with_focus():
    compressor = _make_compressor()
    turns = [
        {"role": "user", "content": "Fix the auth bug"},
        {"role": "assistant", "content": "Fixed the JWT expiry check."},
    ]
    prompts = []

    def mock_call_llm(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return _summary_response()

    with patch("agent.context_compressor.call_llm", mock_call_llm):
        compressor._generate_summary(
            turns,
            focus_topic="authentication",
            memory_context="User uses JWT tokens with a one-hour expiry.",
        )

    assert len(prompts) == 1
    assert "MEMORY PROVIDER CONTEXT" in prompts[0]
    assert "User uses JWT tokens with a one-hour expiry." in prompts[0]
    assert 'FOCUS TOPIC: "authentication"' in prompts[0]


def test_memory_context_injected_into_iterative_summary_prompt():
    compressor = _make_compressor()
    compressor._previous_summary = "Previous checkpoint."
    turns = [
        {"role": "user", "content": "Continue the migration"},
        {"role": "assistant", "content": "Migration continued."},
    ]
    prompts = []

    def mock_call_llm(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return _summary_response("## Goal\nMigration updated.")

    with patch("agent.context_compressor.call_llm", mock_call_llm):
        compressor._generate_summary(
            turns,
            memory_context="Checkpoint id: ctx-123",
        )

    assert len(prompts) == 1
    assert "PREVIOUS SUMMARY:\nPrevious checkpoint." in prompts[0]
    assert "MEMORY PROVIDER CONTEXT" in prompts[0]
    assert "Checkpoint id: ctx-123" in prompts[0]


def test_whitespace_memory_context_is_not_injected():
    compressor = _make_compressor()
    turns = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    prompts = []

    def mock_call_llm(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return _summary_response()

    with patch("agent.context_compressor.call_llm", mock_call_llm):
        compressor._generate_summary(turns, memory_context="  \n\t ")

    assert len(prompts) == 1
    assert "MEMORY PROVIDER CONTEXT" not in prompts[0]


@pytest.mark.parametrize(
    "error_message",
    ["auxiliary provider failed", "model_not_found"],
)
def test_memory_context_survives_summary_model_retry(error_message):
    compressor = _make_compressor()
    compressor.summary_model = "aux/model"
    compressor._summary_model_fallen_back = False
    turns = [
        {"role": "user", "content": "Remember this"},
        {"role": "assistant", "content": "Noted."},
    ]
    prompts = []

    def mock_call_llm(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        if len(prompts) == 1:
            raise RuntimeError(error_message)
        return _summary_response()

    with patch("agent.context_compressor.call_llm", mock_call_llm):
        result = compressor._generate_summary(
            turns,
            memory_context="Checkpoint id: ctx-retry",
        )

    assert result is not None
    assert len(prompts) == 2
    assert all("Checkpoint id: ctx-retry" in prompt for prompt in prompts)


def test_compress_passes_memory_context_with_auto_focus():
    compressor = _make_compressor()
    received_kwargs = {}

    def tracking_generate(_turns, **kwargs):
        received_kwargs.update(kwargs)
        return "## Goal\nTest."

    compressor._generate_summary = tracking_generate
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply1"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply2"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "reply3"},
        {"role": "user", "content": "fourth"},
        {"role": "assistant", "content": "reply4"},
    ]

    compressor.compress(
        messages,
        current_tokens=100_000,
        memory_context="Checkpoint id: ctx-auto-focus",
    )

    assert received_kwargs["memory_context"] == "Checkpoint id: ctx-auto-focus"
    assert received_kwargs["focus_topic"].startswith("Recent user focus:")
