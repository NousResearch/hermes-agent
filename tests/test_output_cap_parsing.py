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


class TestParseVolcengineCeilingOutputCap:
    """Volcengine Ark plan/v3 rejects an over-cap max_tokens with a parameter-
    validation 400 whose ceiling IS the real max-output cap (#51773)."""

    def test_volcengine_above_maximum_value(self):
        msg = (
            "HTTP 400 InvalidParameter: the parameter `max_tokens` specified in "
            "the request are not valid: integer above maximum value, expected a "
            "value <= 32768, but got 65536 instead."
        )
        assert parse_available_output_tokens_from_error(msg) == 32768

    def test_volcengine_less_than_or_equal_to_maximum(self):
        msg = (
            "Invalid value for max_completion_tokens: must be less than or equal "
            "to the maximum value of 16384"
        )
        assert parse_available_output_tokens_from_error(msg) == 16384

    def test_volcengine_is_output_cap(self):
        assert is_output_cap_error(
            "the parameter `max_tokens` ... integer above maximum value, "
            "expected a value <= 32768, but got 65536 instead."
        ) is True

    def test_volcengine_input_overflow_still_compresses(self):
        # Even with a ceiling phrase, if the message clearly describes an
        # oversized INPUT it must stay on the compression path.
        assert is_output_cap_error(
            "prompt is too long: max_tokens above maximum value"
        ) is False


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


class TestParseVllmTokenBasedOutputCap:
    """vLLM reports both the window and the prompt in TOKENS.

    Until this format was parsed, the recovery path misclassified it as
    prompt-too-long and looped through compression (which frees little) while
    retrying with the same oversized max_tokens — terminating in "cannot
    compress further" even though simply lowering the output cap would have
    succeeded.
    """

    # Verbatim vLLM 0.22 / OpenAI-compatible server response (max_tokens set).
    _VLLM_MSG = (
        "This model's maximum context length is 131072 tokens. However, you "
        "requested 65536 output tokens and your prompt contains at least "
        "65537 input tokens, for a total of at least 131073 tokens. Please "
        "reduce the length of the input prompt or the number of requested "
        "output tokens."
    )

    def test_vllm_token_based_format(self):
        # available output = 131072 - 65537 = 65535
        assert parse_available_output_tokens_from_error(self._VLLM_MSG) == 65535

    def test_vllm_without_at_least_qualifier(self):
        # Some versions omit the "at least" hedge.
        msg = ("This model's maximum context length is 131072 tokens. However, "
               "you requested 4096 output tokens and your prompt contains "
               "100000 input tokens, for a total of 104096 tokens.")
        assert parse_available_output_tokens_from_error(msg) == 31072

    def test_vllm_retry_fits_inside_window(self):
        # The retried cap plus the reported input must fit in the window.
        available = parse_available_output_tokens_from_error(self._VLLM_MSG)
        assert available is not None
        assert available + 65537 <= 131072

    def test_vllm_input_alone_exceeds_window_returns_none(self):
        # Input >= window -> lowering the output cap cannot help; the caller
        # must fall through to the compression path.
        msg = ("This model's maximum context length is 131072 tokens. However, "
               "you requested 1024 output tokens and your prompt contains at "
               "least 140000 input tokens, for a total of at least 141024 "
               "tokens.")
        assert parse_available_output_tokens_from_error(msg) is None
