from types import SimpleNamespace

from gateway.platforms.api_server import (
    _agent_usage_dict,
    _chat_completion_usage_payload,
    _responses_usage_payload,
)


def test_agent_usage_dict_includes_cache_counters():
    agent = SimpleNamespace(
        session_prompt_tokens=100,
        session_completion_tokens=25,
        session_cache_read_tokens=40,
        session_cache_write_tokens=10,
        session_total_tokens=175,
    )

    usage = _agent_usage_dict(agent)

    assert usage == {
        "input_tokens": 100,
        "output_tokens": 25,
        "cache_read_tokens": 40,
        "cache_write_tokens": 10,
        "total_tokens": 175,
    }


def test_chat_completion_usage_payload_keeps_openai_fields_and_adds_cache_details():
    payload = _chat_completion_usage_payload(
        {
            "input_tokens": 100,
            "output_tokens": 25,
            "cache_read_tokens": 40,
            "cache_write_tokens": 10,
            "total_tokens": 175,
        }
    )

    assert payload["prompt_tokens"] == 100
    assert payload["completion_tokens"] == 25
    assert payload["total_tokens"] == 175
    assert payload["prompt_tokens_details"] == {
        "cached_tokens": 40,
        "cache_write_tokens": 10,
        "cache_creation_tokens": 10,
    }
    assert payload["cache_read_input_tokens"] == 40
    assert payload["cache_creation_input_tokens"] == 10


def test_responses_usage_payload_keeps_openai_fields_and_adds_cache_details():
    payload = _responses_usage_payload(
        {
            "input_tokens": 100,
            "output_tokens": 25,
            "cache_read_tokens": 40,
            "cache_write_tokens": 10,
            "total_tokens": 175,
        }
    )

    assert payload["input_tokens"] == 100
    assert payload["output_tokens"] == 25
    assert payload["total_tokens"] == 175
    assert payload["input_tokens_details"] == {
        "cached_tokens": 40,
        "cache_write_tokens": 10,
        "cache_creation_tokens": 10,
    }
    assert payload["cache_read_input_tokens"] == 40
    assert payload["cache_creation_input_tokens"] == 10
