"""Real response accounting must retain zero-vs-unavailable cache telemetry."""
import logging
from types import SimpleNamespace

import pytest

from agent.turn_usage import record_response_usage, logger


@pytest.mark.parametrize("advisor", [False, True])
@pytest.mark.parametrize("details,state", [
    ({"cached_tokens": 40, "cache_write_tokens": 0}, "hit"),
    ({"cached_tokens": 0, "cache_write_tokens": 0}, "miss"),
    ({"cached_tokens": 0, "cache_write_tokens": 60}, "cold_write"),
    ({}, "no_field"),
    ({"cached_tokens": None}, "no_field"),
])
def test_real_accounting_cache_states(tmp_path, monkeypatch, caplog, details, state, advisor):
    from run_agent import AIAgent
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = AIAgent(api_key="fixture", provider="openai", api_mode="chat_completions",
                    base_url="https://api.openai.com/v1", model="gpt-4o", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, enabled_toolsets=[])
    usage = {"prompt_tokens": 100, "completion_tokens": 7,
             "prompt_tokens_details": details, "private_payload": "PRIVATE_USAGE"}
    response = SimpleNamespace(usage=usage, choices="PRIVATE_RESPONSE", id="fixture-id")
    if advisor:
        from agent.usage_pricing import normalize_usage
        extra = normalize_usage({"prompt_tokens": 50, "prompt_cache_hit_tokens": 50})
        monkeypatch.setattr(agent.client, "consume_reference_usage", lambda: (extra, None), raising=False)
    try:
        with caplog.at_level(logging.INFO, logger=logger.name):
            record_response_usage(agent, response,
                messages=[{"role": "user", "content": "PRIVATE_PROMPT"}],
                api_call_count=1, api_duration=0.2, compression_attempts=0,
                max_compression_attempts=3)
        line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("API call #"))
        assert f"cache_state={state}" in line
        assert "PRIVATE_" not in line
        assert " id=fixture-id" in line
        assert agent.session_prompt_tokens == (150 if advisor else 100)
        assert "cache_scope=response" in line
        if state == "hit" and not advisor:
            assert " cache=40/100 (40%)" in line
        if state == "no_field":
            assert "cache_read=" not in line
    finally:
        agent.close()


@pytest.mark.parametrize("mode,path", [
    ("anthropic_messages", ("cache_read_input_tokens",)),
    ("anthropic_messages", ("cache_creation_input_tokens",)),
    ("codex_responses", ("input_tokens_details", "cached_tokens")),
    ("codex_responses", ("input_tokens_details", "cache_write_tokens")),
    ("codex_responses", ("input_tokens_details", "cache_creation_tokens")),
    ("chat_completions", ("prompt_tokens_details", "cached_tokens")),
    ("chat_completions", ("prompt_tokens_details", "cache_write_tokens")),
    ("chat_completions", ("prompt_tokens_details", "cache_creation_input_tokens")),
    ("chat_completions", ("cache_read_input_tokens",)),
    ("chat_completions", ("prompt_cache_hit_tokens",)),
    ("chat_completions", ("cached_tokens",)),
    ("chat_completions", ("cache_creation_input_tokens",)),
    ("chat_completions", ("cache_write_tokens",)),
])
@pytest.mark.parametrize("value,present", [(0, True), (7, True), (None, False), ("bad", False)])
@pytest.mark.parametrize("as_object", [False, True])
def test_cache_availability_follows_normalization_paths(mode, path, value, present, as_object):
    from agent.usage_pricing import normalize_usage
    usage = {"prompt_tokens": 100, "input_tokens": 100}
    cursor = usage
    for key in path[:-1]:
        cursor[key] = {}
        cursor = cursor[key]
    cursor[path[-1]] = value
    def convert(obj):
        return SimpleNamespace(**{k: convert(v) if isinstance(v, dict) else v for k, v in obj.items()})
    result = normalize_usage(convert(usage) if as_object else usage, api_mode=mode)
    assert result.cache_telemetry_present is present


def test_combining_usage_does_not_invent_complete_cache_coverage():
    from agent.usage_pricing import normalize_usage
    known = normalize_usage({"prompt_tokens": 100, "cached_tokens": 0})
    unknown = normalize_usage({"prompt_tokens": 100})
    assert (known + known).cache_telemetry_present is True
    assert (known + unknown).cache_telemetry_present is False
    assert (unknown + known).cache_telemetry_present is False
    assert (known + known).raw_usage is None
