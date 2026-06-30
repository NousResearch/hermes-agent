import pytest
from agent.model_metadata import (
    is_output_cap_error,
    parse_available_output_tokens_from_error,
)


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


class TestParseCharBasedOutputCap:
    """LM Studio / llama.cpp report context in tokens but prompt in characters.

    These servers send a hard 400 even on a trivial prompt when the default
    output cap equals the context window (#42741): the request asks for the
    whole window as output, leaving zero room for input.
    """

    def test_char_based_output_cap_format(self):
        msg = ("This model's maximum context length is 65536 tokens. However, "
               "you requested 65536 output tokens and your prompt contains "
               "77409 characters (more than 0 characters, which is the upper "
               "bound for 0 input tokens). Please reduce the length of the "
               "input prompt or the number of requested output tokens.")
        # est input = ceil(77409 / 3) = 25803; available = 65536 - 25803 = 39733
        assert parse_available_output_tokens_from_error(msg) == 39733

    def test_char_based_leaves_room_for_input(self):
        # The whole point: the retried output cap + the estimated input must
        # fit inside the reported context window.
        ctx = 65536
        chars = 77409
        available = parse_available_output_tokens_from_error(
            f"maximum context length is {ctx} tokens. However, you requested "
            f"{ctx} output tokens and your prompt contains {chars} characters."
        )
        assert available is not None
        assert available + (chars + 2) // 3 <= ctx

    def test_char_based_no_room_returns_none(self):
        # Prompt larger than the window (in tokens) -> not an output-cap fix;
        # let the prompt-too-long / compression path handle it.
        msg = ("maximum context length is 1000 tokens. However, you requested "
               "1000 output tokens and your prompt contains 9000 characters.")
        assert parse_available_output_tokens_from_error(msg) is None


class TestParseDashScopeOutputCap:
    """DashScope / Alibaba Cloud (Qwen) reject an over-cap output request with
    a bounded range whose upper bound is the real max-output cap (#55546)."""

    def test_dashscope_range_format(self):
        msg = ("HTTP 400: InternalError.Algo.InvalidParameter: "
               "Range of max_tokens should be [1, 65536]")
        assert parse_available_output_tokens_from_error(msg) == 65536

    def test_dashscope_range_arbitrary_bound(self):
        msg = "Range of max_tokens should be [1, 8192]"
        assert parse_available_output_tokens_from_error(msg) == 8192

    def test_dashscope_range_with_spaces(self):
        msg = "range of max_tokens should be [ 1 , 32768 ]"
        assert parse_available_output_tokens_from_error(msg) == 32768


class TestIsOutputCapError:
    """`is_output_cap_error` is the broader yes/no gate that keeps an
    output-cap 400 out of the compression death-loop even when we can't parse
    a number from the provider's wording (#55546)."""

    def test_dashscope_is_output_cap(self):
        assert is_output_cap_error(
            "Range of max_tokens should be [1, 65536]"
        ) is True

    def test_unknown_numeric_max_tokens_cap_is_output_cap(self):
        # Provider we don't yet parse a number from, but clearly an output cap.
        assert is_output_cap_error("Invalid value: max_tokens should be <= 8192") is True

    def test_anthropic_available_tokens_is_output_cap(self):
        assert is_output_cap_error(
            "max_tokens: 32768 > context_window: 200000 - "
            "input_tokens: 190000 = available_tokens: 10000"
        ) is True

    def test_real_input_overflow_is_not_output_cap(self):
        # Mentions max_tokens but the INPUT is the problem -> compression path.
        assert is_output_cap_error(
            "prompt is too long: 250000 tokens > 200000 max_tokens window"
        ) is False

    def test_gpt5_unsupported_param_is_not_output_cap(self):
        # format_error caught earlier; must NOT be treated as an output cap.
        assert is_output_cap_error(
            "Unsupported parameter: 'max_tokens' is not supported with this "
            "model. Use 'max_completion_tokens' instead."
        ) is False

    def test_unrelated_error_is_not_output_cap(self):
        assert is_output_cap_error("some unrelated 400 error") is False
