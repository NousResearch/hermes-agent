"""Provider error-text parsing: context windows vs output caps (sibling of ``model_metadata``)."""

from __future__ import annotations

import re
from typing import Optional


def parse_context_limit_from_error(error_msg: str) -> Optional[int]:
    """Context limit quoted in a provider error ("maximum context length is 32768 tokens"), if any.

    A message about only an OUTPUT cap ("... model output limit of 16384", "max_tokens exceeds the
    limit of 4096") never says "context"; bail out so the generic "limit ... of N" pattern can't
    cache the output cap as the window. vLLM's max_model_len IS the window, so it still parses."""
    error_lower = error_msg.lower()
    output_subject = any(p in error_lower for p in (
        "output limit", "output token", "max_tokens", "max_completion_tokens", "max_output_tokens"))
    window_named = any(p in error_lower for p in ("context", "max_model_len", "maximum model length"))
    if output_subject and not window_named:
        return None
    patterns = (
        r'max_model_len\s*(?:is\s*)?[:=(]?\s*(\d{4,})',  # vLLM: "max_model_len 32768", "=32768", ": 32768", "(32768)", "is 32768"
        r'maximum model length\s*(?:is\s*)?[:=(]?\s*(\d{4,})',  # vLLM alt: "maximum model length 131072", "... is 131072"
        r'(?:max(?:imum)?|limit)\s*(?:context\s*)?(?:length|size|window)?\s*(?:is|of|:)?\s*(\d{4,})',
        r'context\s*(?:length|size|window)\s*(?:is|of|:)?\s*(\d{4,})',
        r'(\d{4,})\s*(?:token)?\s*(?:context|limit)',
        r'>\s*(\d{4,})\s*(?:max|limit|token)',  # "250000 tokens > 200000 maximum"
        r'(\d{4,})\s*(?:max(?:imum)?)\b',  # "200000 maximum"
        # Gemini: "input token count is 32825 but model only supports up to
        # 32768" — anchor on the phrase so the input count isn't captured.
        r'supports?\s+(?:only\s+)?up\s+to\s+(\d{4,})',
    )
    for match in filter(None, (re.search(pattern, error_lower) for pattern in patterns)):
        limit = int(match.group(1))
        if 1024 <= limit <= 10_000_000:  # sanity: must be a plausible window
            return limit
    return None


def get_context_length_from_provider_error(error_msg: str, current_context_length: int) -> Optional[int]:
    """Provider-reported limit LOWER than the current window, else None. Overflow recovery must
    not invent a window: when the provider only says the input is too long, callers keep the
    configured length and compress rather than stepping down guessed probe tiers."""
    parsed_limit = parse_context_limit_from_error(error_msg)
    return parsed_limit if parsed_limit is not None and parsed_limit < current_context_length else None


# OpenAI's original overflow wording, copied by vLLM / llama-cpp-python: "(36865 in the messages,
# 65536 in the completion)"; legacy completions: "(771 in your prompt; 4000 for the completion)".
# The first figure is the prompt the server MEASURED, the second the requested max_tokens.
_COMPLETION_SPLIT_RE = re.compile(
    r'\((\d+)\s+(?:tokens\s+)?in (?:the messages|your prompt|the prompt)\s*[;,]\s*'
    r'(\d+)\s+(?:tokens\s+)?(?:in|for) the completion\)'
)


def _completion_split_budget(error_lower: str) -> Optional[int]:
    """window - measured prompt from the OpenAI-style parenthetical split, or None when the wording
    is absent or the prompt alone fills the window (a genuine input overflow -> compress)."""
    split = _COMPLETION_SPLIT_RE.search(error_lower)
    ctx = re.search(r'maximum context length is (\d+)', error_lower)
    if not split or not ctx:
        return None
    available = int(ctx.group(1)) - int(split.group(1))
    return available if available >= 1 else None


def parse_available_output_tokens_from_error(error_msg: str) -> Optional[int]:
    """Available OUTPUT tokens from a "max_tokens too large" error, or None. Distinct from "prompt
    too long" (-> compress): here input + requested_output > window, so the fix is a smaller
    max_tokens for this call and context_length must NOT be touched."""
    error_lower = error_msg.lower()
    if not _any_phrase_group(error_lower, _PARSEABLE_OUTPUT_CAP_SIGNALS):
        return None
    # Direct cap figures, most specific first: "exceeds model's maximum output tokens (65536)", "Range of
    # max_tokens should be [1, 65536]" (upper bound is the cap), Anthropic "max_tokens: 100000 > 64000, which
    # is the maximum allowed number of output tokens" (the ceiling is the right-hand side), Anthropic
    # "= available_tokens: 10000", last "= N".
    for pattern in (
        r'exceeds model(?:\'s)? maximum output tokens\s*\(?\s*(\d+)\s*\)?',
        r'exceeds the limit of\s*(\d+)',
        r'is greater than the maximum of\s*(\d+)',
        r'cannot be greater than\s*(?:max_model_len\s*=\s*)?(\d+)',
        r'max_tokens\s*:\s*\d+\s*>\s*(\d+)\s*,?\s*which is the maximum allowed number of output tokens',
        r'range of max_tokens should be\s*\[\s*\d+\s*,\s*(\d+)\s*\]',
        r'available_tokens[:\s]+(\d+)',
        r'available\s+tokens[:\s]+(\d+)',
        # Switchyard: "max_tokens cannot exceed the configured model output limit of 16384".
        r'output limit (?:of|is)\s*(\d+)',
        # Azure OpenAI: "max_tokens is too large: 65536. This model supports at most 32768 completion tokens."
        r'supports at most\s+(\d+)\s*(?:completion\s+)?tokens',
        # Scaleway: "max_completion_tokens is limited to 16384 for glm-5.2".
        r'(?:max_tokens|max_completion_tokens) is limited to\s*(\d+)',
        r'=\s*(\d+)\s*$',
    ):
        match = re.search(pattern, error_lower)
        if match and int(match.group(1)) >= 1:
            return int(match.group(1))
    # OpenRouter/Nous: "maximum context length is N … (A of text input, B of tool input, C in the output)" -> ctx - A - B.
    _m_ctx = re.search(r'maximum context length is (\d+)', error_lower)
    _m_parts = re.search(r'\((\d+)\s+of text input,\s*(\d+)\s+of tool input,\s*(\d+)\s+in the output\)', error_lower)
    if _m_ctx and _m_parts:
        _available = int(_m_ctx.group(1)) - int(_m_parts.group(1)) - int(_m_parts.group(2))
        if _available >= 1:
            return _available
    _split_available = _completion_split_budget(error_lower)
    if _split_available is not None:
        return _split_available
    # LM Studio / llama.cpp: window in tokens, prompt in CHARACTERS; ~3 chars/token over-reserves the input.
    _m_ctx_tok = re.search(r'maximum context length is (\d+)\s*token', error_lower)
    _m_chars = re.search(r'prompt contains (\d+)\s*character', error_lower)
    if _m_ctx_tok and _m_chars:
        _available = int(_m_ctx_tok.group(1)) - (int(_m_chars.group(1)) + 2) // 3
        if _available >= 1:
            return _available
    # SGLang: "maximum context length of 131072 tokens. You requested a total of 132528 tokens: 66992 tokens
    # from the input messages and 65536 tokens for the completion" -> window - input (None when the input
    # alone overflows -> compress).
    _m_sglang = _sglang_window_and_input(error_lower)
    if _m_sglang and _m_sglang[0] - _m_sglang[1] >= 1:
        return _m_sglang[0] - _m_sglang[1]
    # vLLM: window and prompt both in TOKENS; available = window - input (None when the input alone
    # overflows -> compress). When max_tokens is the BINDING constraint vLLM reports "at least N input
    # tokens" with N == window + 1 - requested_output, so window - N == requested_output - 1 and each
    # retry walks the cap down by the safety margin without ever fitting: halve the cap instead.
    _m_vllm_input = re.search(r'prompt contains (?:at least )?(\d+)\s*input tokens', error_lower)
    if _m_ctx_tok and _m_vllm_input:
        _available = int(_m_ctx_tok.group(1)) - int(_m_vllm_input.group(1))
        _m_requested_out = re.search(r'requested (\d+)\s*output tokens', error_lower)
        if 'at least' in error_lower and _m_requested_out:
            _requested_out = int(_m_requested_out.group(1))
            if _available >= _requested_out - 1:
                # The budget is derived from the constraint, not measured.
                return max(1, _requested_out // 2)
        if _available >= 1:
            return _available
    return None


# Each entry is a phrase group; the group matches when ALL phrases are present.
# DashScope, Anthropic (available_tokens / "maximum allowed number of output tokens"), OpenRouter/Nous,
# LM Studio/llama.cpp, generic "should be <= N", OpenAI-compat relays.
_OUTPUT_CAP_SIGNALS = (
    ("range of max_tokens should be",), ("available_tokens",), ("available tokens",),
    ("in the output", "maximum context length"), ("requested", "output tokens"),
    ("should be",), ("less than or equal",), ("must be",), ("exceeds model", "maximum output tokens"),
    ("output limit",), ("maximum allowed number of output tokens",),
    ("max_tokens is too large", "supports at most"), ("tokens from the input messages", "tokens for the completion"),
    ("limited to",),  # Scaleway: "max_completion_tokens is limited to 16384 for <model>" (#67453)
    ("exceeds the limit of",), ("is greater than the maximum of",), ("cannot be greater than",),
)
_INPUT_OVERFLOW_SIGNALS = (
    "prompt is too long", "prompt too long", "input is too long", "input token",
    "prompt length", "prompt contains", "reduce the length",
)
# Narrower than _OUTPUT_CAP_SIGNALS: only phrasings we can extract a number from.
# "requested N output tokens" means the OUTPUT cap is the problem (the input fits) —
# reduce max_tokens, don't compress. DashScope's bounded range upper bound IS the
# real max-output cap ("Range of max_tokens should be [1, 65536]").
_PARSEABLE_OUTPUT_CAP_SIGNALS = (
    ("max_tokens", "available_tokens"), ("max_tokens", "available tokens"),
    ("in the output", "maximum context length"),
    ("maximum context length", "requested", "output tokens"),
    ("maximum context length", "in the completion"), ("maximum context length", "for the completion"),
    ("range of max_tokens should be",), ("exceeds model", "maximum output tokens"),
    ("output limit",), ("max_tokens", "maximum allowed number of output tokens"),
    ("max_tokens is too large", "supports at most"), ("tokens from the input messages", "tokens for the completion"),
    ("limited to",), ("exceeds the limit of",), ("is greater than the maximum of",),
    ("cannot be greater than",),
)


def _sglang_window_and_input(error_lower: str) -> Optional[tuple[int, int]]:
    """``(window, input_tokens)`` from SGLang's wording, else None; both figures are explicit there."""
    _m_ctx = re.search(r'maximum context length of (\d+)\s*token', error_lower)
    _m_in = re.search(r'(\d+)\s*tokens from the input messages', error_lower)
    return (int(_m_ctx.group(1)), int(_m_in.group(1))) if _m_ctx and _m_in else None


def _any_phrase_group(text: str, groups: tuple) -> bool:
    return any(all(p in text for p in group) for group in groups)


def is_output_cap_error(error_msg: str) -> bool:
    """Yes/no sibling of :func:`parse_available_output_tokens_from_error` for unparseable wordings. An
    output-cap 400 misclassified as context overflow death-loops the compressor (same max_tokens, same
    rejection). Signal: talks about max_tokens as a cap/range/limit and NOT about an oversized input."""
    error_lower = error_msg.lower()
    # The OpenAI-style split names neither max_tokens nor "output tokens" and ends with "reduce the
    # length", so it fails both gates below; the measured prompt decides instead (#90607).
    if _completion_split_budget(error_lower) is not None:
        return True
    # An error that ALSO describes an oversized INPUT is a genuine overflow — compression can fix it.
    # SGLang states both figures: input >= window is that same genuine overflow.
    _m_sglang = _sglang_window_and_input(error_lower)
    return (
        any(p in error_lower for p in ("max_tokens", "max_output_tokens", "max_completion_tokens", "tokens for the completion"))
        and _any_phrase_group(error_lower, _OUTPUT_CAP_SIGNALS)
        and not any(p in error_lower for p in _INPUT_OVERFLOW_SIGNALS)
        and not (_m_sglang and _m_sglang[1] >= _m_sglang[0])
    )
