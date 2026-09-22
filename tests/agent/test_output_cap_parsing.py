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

    def test_anthropic_output_ceiling_format(self):
        msg = (
            "max_tokens: 100000 > 64000, which is the maximum allowed number "
            "of output tokens for claude-sonnet-4-5"
        )
        assert parse_available_output_tokens_from_error(msg) == 64000

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

    def test_anthropic_max_tokens_output_ceiling_is_output_cap(self):
        # Anthropic invalid_request_error uses the same error type for bad
        # schemas, images, and output-cap violations. This message is about the
        # requested response budget, not input/context overflow, so it must stay
        # out of the compression path. Port of cline/cline#12876.
        assert is_output_cap_error(
            "max_tokens: 100000 > 64000, which is the maximum allowed number "
            "of output tokens for claude-sonnet-4-5"
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



class TestParseAdvertisedCeilingWordings:
    """Azure and SGLang name the ceiling without any phrase the parser knew (#78405, #83521).
    Unrecognized, the 400 carried the bare ``max_tokens`` substring (or nothing at all) into
    the compression path and a fresh session died with "cannot be shrunk further"."""

    @pytest.mark.parametrize("msg, available", [
        # Azure OpenAI (verbatim from #78405): the advertised completion ceiling IS the budget.
        ("Error: max_tokens is too large: 65536. This model supports at most 32768 completion tokens.", 32768),
        # SGLang (verbatim from #83521): window - input; never mentions max_tokens.
        ("Requested token count exceeds the model's maximum context length of 131072 tokens. You requested "
         "a total of 132528 tokens: 66992 tokens from the input messages and 65536 tokens for the completion. "
         "Please reduce the number of tokens in the input messages or the completion to fit within the limit.",
         64080),
    ])
    def test_ceiling_is_parsed_and_classified_as_output_cap(self, msg, available):
        assert parse_available_output_tokens_from_error(msg) == available
        assert is_output_cap_error(msg) is True

    def test_sglang_input_alone_over_window_routes_to_compression(self):
        # Same wording, but the input by itself exceeds the window: shrinking the output cannot help.
        msg = ("Requested token count exceeds the model's maximum context length of 100000 tokens. You requested "
               "a total of 150000 tokens: 120000 tokens from the input messages and 30000 tokens for the completion.")
        assert parse_available_output_tokens_from_error(msg) is None
        assert is_output_cap_error(msg) is False
def test_limited_to_phrasing_is_an_output_cap():
    """#67453: Scaleway rejects an oversized budget with "max_completion_tokens is limited to N for
    <model>" — an output cap (step the budget down), not a context overflow (do not compress)."""
    assert is_output_cap_error("max_completion_tokens is limited to 16384 for glm-5.2")
    assert parse_available_output_tokens_from_error("max_completion_tokens is limited to 16384 for glm-5.2") == 16384
    assert not is_output_cap_error("prompt is too long: max_tokens limited to 100 given the input")


class TestParseOpenAiCompletionSplit:
    """OpenAI's original overflow wording, copied by vLLM / llama-cpp-python, splits the request
    as "(A in the messages, B in the completion)" and never names max_tokens (#90607)."""

    @pytest.mark.parametrize("msg, budget", [
        ("This model's maximum context length is 102400 tokens. However, you requested 102401 tokens "
         "(36865 in the messages, 65536 in the completion). Please reduce the length of the messages or completion.",
         102400 - 36865),
        ("This model's maximum context length is 4097 tokens, however you requested 4771 tokens "
         "(771 in your prompt; 4000 for the completion). Please reduce your prompt; or completion length.",
         4097 - 771),
    ])
    def test_split_is_output_cap_with_window_minus_measured_prompt(self, msg, budget):
        assert parse_available_output_tokens_from_error(msg) == budget
        assert is_output_cap_error(msg)

    def test_split_with_prompt_filling_window_stays_on_compression(self):
        msg = ("This model's maximum context length is 4097 tokens. However, you requested 6000 tokens "
               "(5000 in the messages, 1000 in the completion). Please reduce the length of the messages or completion.")
        assert parse_available_output_tokens_from_error(msg) is None
        assert not is_output_cap_error(msg)
