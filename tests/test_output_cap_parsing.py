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
    """DashScope/Alibaba (Qwen) reports max_tokens range as [1, N]."""

    def test_dashscope_range_format(self):
        msg = "InternalError.Algo.InvalidParameter: Range of max_tokens should be [1, 65536]"
        assert parse_available_output_tokens_from_error(msg) == 65536

    def test_dashscope_different_cap(self):
        msg = "Range of max_tokens should be [1, 8192]"
        assert parse_available_output_tokens_from_error(msg) == 8192

    def test_dashscope_with_surrounding_text(self):
        msg = ("HTTP 400: InternalError.Algo.InvalidParameter: "
               "Range of max_tokens should be [1, 65536]. Please check and retry.")
        assert parse_available_output_tokens_from_error(msg) == 65536
