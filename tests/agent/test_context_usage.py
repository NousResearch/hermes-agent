from types import SimpleNamespace
from unittest.mock import patch

from agent.context_usage import build_context_usage_payload, emit_context_usage


class _FakeCompressor:
    last_prompt_tokens = 111_400
    context_length = 200_000
    compression_count = 2


def test_build_context_usage_payload_includes_context_and_session_fields():
    agent = SimpleNamespace(
        model="openai/gpt-5.5-pro",
        session_input_tokens=50_000,
        session_output_tokens=20_000,
        session_cache_read_tokens=5_000,
        session_cache_write_tokens=1_000,
        session_reasoning_tokens=500,
        session_prompt_tokens=50_000,
        session_completion_tokens=20_000,
        session_total_tokens=70_000,
        context_compressor=_FakeCompressor(),
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
    )

    payload = build_context_usage_payload(agent)

    assert payload["model"] == "openai/gpt-5.5-pro"
    assert payload["context_used"] == 111_400
    assert payload["context_max"] == 200_000
    assert payload["context_percent"] == 56
    assert payload["compressions"] == 2
    assert payload["session"]["input_tokens"] == 50_000
    assert payload["session"]["output_tokens"] == 20_000
    assert payload["session"]["cache_read_tokens"] == 5_000
    assert payload["session"]["reasoning_tokens"] == 500


def test_build_context_usage_payload_emits_categories_from_prompt_parts_and_tools():
    agent = SimpleNamespace(
        model="openai/gpt-5.5-pro",
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        context_compressor=_FakeCompressor(),
        provider="openrouter",
        base_url="",
        tools=[{"type": "function", "function": {"name": "shell"}}],
        conversation_history=[
            {"role": "user", "content": "x" * 400},
            {"role": "assistant", "content": "y" * 200},
        ],
    )

    fake_parts = {
        "stable": "a" * 800,    # ~200 tokens
        "context": "b" * 1600,  # ~400 tokens
        "volatile": "c" * 200,  # ~50 tokens
    }

    with patch(
        "agent.system_prompt.build_system_prompt_parts",
        return_value=fake_parts,
    ):
        payload = build_context_usage_payload(agent)

    categories = payload.get("categories")
    assert categories, "expected a categories array in the payload"
    by_key = {entry["key"]: entry for entry in categories}
    assert by_key["system"]["label"] == "System prompt"
    assert by_key["system"]["tokens"] == 200
    assert by_key["rules"]["tokens"] == 400
    assert by_key["memory"]["tokens"] == 50
    assert by_key["tools"]["tokens"] > 0
    assert by_key["conversation"]["tokens"] > 0


def test_build_context_usage_payload_omits_categories_when_no_sources():
    agent = SimpleNamespace(
        model="test",
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        context_compressor=_FakeCompressor(),
        provider="openrouter",
        base_url="",
    )

    with patch(
        "agent.system_prompt.build_system_prompt_parts",
        side_effect=Exception("no system prompt builder for this agent shape"),
    ):
        payload = build_context_usage_payload(agent)

    assert "categories" not in payload


def test_emit_context_usage_invokes_callback():
    agent = SimpleNamespace(
        model="test",
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        context_compressor=_FakeCompressor(),
        provider="openrouter",
        base_url="",
    )
    seen = []

    agent.context_usage_callback = seen.append
    emit_context_usage(agent)

    assert len(seen) == 1
    assert seen[0]["context_percent"] == 56
