from agent.chat_completion_helpers import _validated_openrouter_provider_sort


def test_validated_openrouter_provider_sort_accepts_valid_values():
    assert _validated_openrouter_provider_sort("price") == "price"
    assert _validated_openrouter_provider_sort(" latency ") == "latency"
    assert _validated_openrouter_provider_sort("THROUGHPUT") == "throughput"


def test_validated_openrouter_provider_sort_rejects_invalid_values():
    assert _validated_openrouter_provider_sort("intelligence") is None
    assert _validated_openrouter_provider_sort("") is None
    assert _validated_openrouter_provider_sort(None) is None
