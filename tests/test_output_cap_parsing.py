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


class TestParseMaximumOutputTokensCap:
    """Some OpenAI-compatible relays report the model's separate output cap."""

    def test_parenthesized_max_output_cap(self):
        msg = (
            "API call failed after 3 retries: [400]: max_tokens (98304) "
            "exceeds model's maximum output tokens (65536)"
        )
        assert parse_available_output_tokens_from_error(msg) == 65536

    def test_parenthesized_max_output_cap_is_output_cap(self):
        assert is_output_cap_error(
            "max_tokens (98304) exceeds model's maximum output tokens (65536)"
        ) is True


class TestIsOutputCapError:
    """`is_output_cap_error` is the broader yes/no gate that keeps an
    output-cap 400 out of the compression death-loop even when we can't parse
    a number from the provider's wording (#55546)."""

    def test_dashscope_is_output_cap(self):
        assert is_output_cap_error(
            "Range of max_tokens should be [1, 65536]"
        ) is True


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

    # Verbatim vLLM response where the input is MEASURED, not back-computed:
    # window - input != requested - 1, so the reported figure is real.
    _VLLM_MSG_REAL_INPUT = (
        "This model's maximum context length is 131072 tokens. However, you "
        "requested 65536 output tokens and your prompt contains 100000 "
        "input tokens, for a total of 165536 tokens. Please reduce the length "
        "of the input prompt or the number of requested output tokens."
    )

    def test_vllm_token_based_format(self):
        # The reported input is a LOWER BOUND that vLLM back-computes from the
        # constraint (65537 == 131072 + 1 - 65536), so window - input is just
        # requested - 1 and carries no information about the real prompt.
        # Halve the requested cap instead so the retry actually converges.
        assert parse_available_output_tokens_from_error(self._VLLM_MSG) == 32768

    def test_vllm_measured_input_is_trusted(self):
        # When the input is measured rather than derived, use it as-is.
        # available output = 131072 - 100000 = 31072
        assert parse_available_output_tokens_from_error(
            self._VLLM_MSG_REAL_INPUT
        ) == 31072

    def test_vllm_retry_fits_inside_window(self):
        # The retried cap plus the reported input must fit in the window.
        available = parse_available_output_tokens_from_error(self._VLLM_MSG)
        assert available is not None
        assert available + 65537 <= 131072

    def test_vllm_retry_converges(self):
        """The retry sequence must reach a working cap in a few attempts.

        Regression test for the 65-tokens-per-retry crawl: with a 102400
        window and a real prompt of ~37000 tokens, retrying from a 65536 cap
        used to produce 65471 -> 65406 -> 65341 and exhaust the compression
        budget without ever fitting.
        """
        window, real_input, cap = 102400, 37000, 65536
        for _ in range(5):
            if real_input + cap <= window:
                break
            # vLLM's message when max_tokens is the binding constraint.
            msg = (
                f"This model's maximum context length is {window} tokens. "
                f"However, you requested {cap} output tokens and your prompt "
                f"contains at least {window + 1 - cap} input tokens, for a "
                f"total of at least {window + 1} tokens."
            )
            available = parse_available_output_tokens_from_error(msg)
            assert available is not None
            assert available < cap, "each retry must lower the cap"
            cap = available
        assert real_input + cap <= window, f"did not converge: cap={cap}"

