import pytest
from agent.model_metadata import parse_available_output_tokens_from_error


class TestParseOpenRouterOutputCap:
    """OpenRouter/Nous phrase the output-cap error as a context breakdown."""

    def test_openrouter_breakdown_format(self):
        msg = ("This endpoint's maximum context length is 200000 tokens. "
               "However, you requested about 195000 tokens "
               "(150000 of text input, 40000 of tool input, 5000 in the output).")
        # available output = 200000 - 150000 - 40000 = 10000
        assert parse_available_output_tokens_from_error(msg) == 10000

    def test_anthropic_format_still_works(self):
        msg = ("max_tokens: 32768 > context_window: 200000 - "
               "input_tokens: 190000 = available_tokens: 10000")
        assert parse_available_output_tokens_from_error(msg) == 10000

    def test_non_output_cap_error_returns_none(self):
        assert parse_available_output_tokens_from_error("some unrelated 400 error") is None

    def test_breakdown_with_no_room_returns_none(self):
        # ctx - text - tool <= 0 -> None (don't return a non-positive cap)
        msg = ("maximum context length is 1000 tokens "
               "(900 of text input, 200 of tool input, 0 in the output)")
        assert parse_available_output_tokens_from_error(msg) is None
