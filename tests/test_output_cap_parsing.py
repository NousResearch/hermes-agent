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

    def test_vllm_token_based_format(self):
        # available output = 131072 - 65537 = 65535
        assert parse_available_output_tokens_from_error(self._VLLM_MSG) == 65535


    def test_vllm_retry_fits_inside_window(self):
        # The retried cap plus the reported input must fit in the window.
        available = parse_available_output_tokens_from_error(self._VLLM_MSG)
        assert available is not None
        assert available + 65537 <= 131072


class TestParseAzureOutputCap:
    """Azure OpenAI rejects an over-cap output request by naming the OUTPUT
    parameter and the model's advertised completion-token ceiling:

        "max_tokens is too large: 65536. This model supports at most
         32768 completion tokens."

    The input itself fits — this is purely an output-cap error.  Until this
    phrasing was parsed, the recovery path classified it as context overflow
    (the bare ``max_tokens`` substring) and looped through compression on a
    tiny conversation, ending in "cannot compress further" (issue #78405).
    """

    # Verbatim Azure OpenAI 400 from issue #78405 (gpt-4.1-mini, 32k ceiling).
    _AZURE_MSG = (
        "Error: max_tokens is too large: 65536. This model supports at most "
        "32768 completion tokens."
    )

    def test_azure_supports_at_most_format(self):
        # The advertised completion-token ceiling is the available output cap.
        assert parse_available_output_tokens_from_error(self._AZURE_MSG) == 32768

    def test_azure_arbitrary_bound(self):
        msg = ("Error: max_tokens is too large: 65536. This model supports "
               "at most 16384 completion tokens.")
        assert parse_available_output_tokens_from_error(msg) == 16384

    def test_azure_without_completion_word(self):
        # Variant without the "completion" qualifier — the regex allows it.
        msg = ("Error: max_tokens is too large: 65536. This model supports "
               "at most 32768 tokens.")
        assert parse_available_output_tokens_from_error(msg) == 32768
        assert is_output_cap_error(msg) is True

    def test_azure_case_insensitive(self):
        msg = ("ERROR: Max_Tokens Is Too Large: 65536. THIS MODEL SUPPORTS "
               "AT MOST 32768 COMPLETION TOKENS.")
        assert parse_available_output_tokens_from_error(msg) == 32768
        assert is_output_cap_error(msg) is True

    def test_azure_is_output_cap(self):
        assert is_output_cap_error(self._AZURE_MSG) is True

    def test_azure_retry_fits_inside_window(self):
        # The retried cap (cap - 64 margin, applied by the caller) plus the
        # tiny input must fit inside the advertised ceiling.
        available = parse_available_output_tokens_from_error(self._AZURE_MSG)
        assert available is not None
        assert available - 64 >= 1

    def test_input_overflow_mentioning_ceiling_is_not_output_cap(self):
        # An input-overflow that happens to cite the completion-token ceiling
        # must stay on the compression path: no "max_tokens is too large".
        msg = ("Error: input tokens (80000) exceed the model's context "
               "window; the model supports at most 32768 completion tokens.")
        assert parse_available_output_tokens_from_error(msg) is None
        assert is_output_cap_error(msg) is False

    def test_input_overflow_mentioning_max_tokens_is_not_output_cap(self):
        # Genuine input overflow that also names max_tokens (no "is too
        # large" phrasing) must not be flagged as an output-cap error.
        msg = ("This model's maximum context length is 32768 tokens. However, "
               "your messages resulted in 35000 tokens. Please reduce the "
               "length of the messages or max_tokens.")
        assert parse_available_output_tokens_from_error(msg) is None
        assert is_output_cap_error(msg) is False

    def test_input_overflow_with_too_large_phrase_is_not_output_cap(self):
        # Even with the "max_tokens is too large" phrase, a message that
        # clearly describes oversized INPUT must stay on the compression
        # path (input-overflow signals win in both functions).
        msg = ("Error: max_tokens is too large for this request: input tokens "
               "(80000) exceed the model's context window; this model "
               "supports at most 32768 completion tokens.")
        assert parse_available_output_tokens_from_error(msg) is None
        assert is_output_cap_error(msg) is False

