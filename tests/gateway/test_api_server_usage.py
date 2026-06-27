"""Tests for _build_openai_usage and _build_responses_usage helpers (#25400)."""

from gateway.platforms.api_server import _build_openai_usage, _build_responses_usage


class TestBuildOpenAIUsage:
    """Chat Completions format usage construction."""

    def test_basic_usage(self):
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        result = _build_openai_usage(usage)
        assert result == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    def test_defaults_to_zero(self):
        result = _build_openai_usage({})
        assert result == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def test_includes_cached_tokens(self):
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cache_read_tokens": 80,
        }
        result = _build_openai_usage(usage)
        assert result["prompt_tokens_details"] == {"cached_tokens": 80}

    def test_includes_cache_write_tokens(self):
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cache_write_tokens": 30,
        }
        result = _build_openai_usage(usage)
        assert result["prompt_tokens_details"] == {"cache_creation_input_tokens": 30}

    def test_includes_both_cache_fields(self):
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cache_read_tokens": 60,
            "cache_write_tokens": 20,
        }
        result = _build_openai_usage(usage)
        assert result["prompt_tokens_details"] == {
            "cached_tokens": 60,
            "cache_creation_input_tokens": 20,
        }

    def test_no_details_when_no_cache(self):
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        result = _build_openai_usage(usage)
        assert "prompt_tokens_details" not in result


class TestBuildResponsesUsage:
    """Responses API format usage construction."""

    def test_basic_usage(self):
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        result = _build_responses_usage(usage)
        assert result == {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

    def test_includes_cached_tokens(self):
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cache_read_tokens": 80,
        }
        result = _build_responses_usage(usage)
        assert result["input_tokens_details"] == {"cached_tokens": 80}

    def test_no_details_when_no_cache(self):
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        result = _build_responses_usage(usage)
        assert "input_tokens_details" not in result
