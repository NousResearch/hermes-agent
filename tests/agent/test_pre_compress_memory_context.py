"""Behavior contracts for memory-provider context in compression prompts."""

import json

from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor


def _make_compressor():
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.protect_first_n = 2
    compressor.protect_last_n = 5
    # Set context_length BEFORE the derived budgets: its setter resets the
    # lazily-cached threshold/tail/summary budgets (#32221 lazy init), so
    # assigning it later would clear the explicit values below.
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


def test_memory_context_is_strictly_redacted_before_summary_llm(monkeypatch):
    compressor = _make_compressor()
    prefix_secret = "sk-" + "b" * 30
    query_secret = "opaque-query-secret"
    userinfo_value = "opaque-userinfo-value"
    hyphen_client_secret = "HYPHEN_CLIENT_SECRET"
    hyphen_access_secret = "HYPHEN_ACCESS_SECRET"
    hyphen_api_secret = "HYPHEN_API_SECRET"
    encoded_hyphen_secret = "ENCODED_HYPHEN_SECRET"
    array_query_secret = "ARRAY_QUERY_SECRET"
    encoded_array_query_secret = "ENCODED_ARRAY_QUERY_SECRET"
    prompts = []

    def mock_call_llm(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return _summary_response()

    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    with patch("agent.context_compressor.call_llm", mock_call_llm):
        compressor._generate_summary(
            [{"role": "user", "content": "Continue"}],
            memory_context=(
                f"api key: {prefix_secret}\n"
                f"callback: https://example.test/cb?token={query_secret}\n"
                f"endpoint: https://user:{userinfo_value}@example.test/private\n"
                f"hyphen-client: /resume?client-secret={hyphen_client_secret}\n"
                f"hyphen-access: /resume?Access-Token={hyphen_access_secret}\n"
                f"hyphen-api: /resume?api-key={hyphen_api_secret}\n"
                f"encoded-hyphen: /resume?client%2Dsecret={encoded_hyphen_secret}\n"
                f"array: /resume?access_token[]={array_query_secret}\n"
                f"encoded-array: /resume?access_token%5B%5D={encoded_array_query_secret}"
            ),
        )

    assert len(prompts) == 1
    prompt = prompts[0]
    assert prefix_secret not in prompt
    assert query_secret not in prompt
    assert userinfo_value not in prompt
    assert hyphen_client_secret not in prompt
    assert hyphen_access_secret not in prompt
    assert hyphen_api_secret not in prompt
    assert encoded_hyphen_secret not in prompt
    assert array_query_secret not in prompt
    assert encoded_array_query_secret not in prompt
    assert "token=***" in prompt
    assert "https://user:***@example.test/private" in prompt
    assert "client-secret=***" in prompt
    assert "Access-Token=***" in prompt
    assert "api-key=***" in prompt
    assert "client%2Dsecret=***" in prompt
    assert "access_token[]=***" in prompt
    assert "access_token%5B%5D=***" in prompt






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




