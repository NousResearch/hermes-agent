from agent.usage_pricing import CanonicalUsage
from run_agent import _build_context_engine_usage_dict


def test_context_engine_usage_dict_exposes_canonical_cache_fields():
    usage = CanonicalUsage(
        input_tokens=600,
        output_tokens=120,
        cache_read_tokens=400,
        cache_write_tokens=50,
        reasoning_tokens=30,
    )

    payload = _build_context_engine_usage_dict(usage)

    assert payload == {
        "prompt_tokens": 1050,
        "completion_tokens": 120,
        "total_tokens": 1170,
        "input_tokens": 600,
        "output_tokens": 120,
        "cache_read_tokens": 400,
        "cache_write_tokens": 50,
        "reasoning_tokens": 30,
    }


def test_context_engine_usage_dict_keeps_zero_cache_fields_present():
    usage = CanonicalUsage(input_tokens=600, output_tokens=120)

    payload = _build_context_engine_usage_dict(usage)

    assert "cache_read_tokens" in payload
    assert "cache_write_tokens" in payload
    assert payload["cache_read_tokens"] == 0
    assert payload["cache_write_tokens"] == 0
