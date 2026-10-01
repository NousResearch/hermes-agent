"""Token estimation and provider-error parsing utilities.

Context-window metadata is owned by :mod:`models.metadata.context`."""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.message_metadata import PERSISTENCE_ONLY_MESSAGE_FIELDS

logger = logging.getLogger(__name__)


def install_context_metadata_hooks() -> None:
    """Bridge runtime-owned route/config integrations into models.metadata."""
    from models.metadata.context import configure_context_metadata_hooks

    def tls_settings(base_url: str):
        from hermes_cli.config import get_custom_provider_tls_settings
        return get_custom_provider_tls_settings(base_url)

    def http_get(url: str, **kwargs):
        from agent import model_metadata_http
        return model_metadata_http.get(url, **kwargs)

    def http_stream(url: str, **kwargs):
        from agent import model_metadata_http
        return model_metadata_http.stream(url, **kwargs)

    def resolve_verify(base_url: str = ""):
        from agent import model_metadata_http
        return model_metadata_http.resolve_verify(base_url)

    def is_codex_oauth_token(access_token: str) -> bool:
        from hermes_cli.auth_constants import _decode_jwt_claims
        return bool(_decode_jwt_claims(access_token))

    def materialize_api_key(api_key: object) -> str:
        from agent.command_token_source import materialize_probe_api_key
        return materialize_probe_api_key(api_key)

    def fingerprint_api_key(api_key: object) -> str:
        from agent.credential_persistence import fingerprint_secret_value
        return fingerprint_secret_value(api_key) or ""

    def custom_provider_api_mode(base_url: str, custom_providers: list | None):
        from hermes_cli.config import get_custom_provider_api_mode
        return get_custom_provider_api_mode(base_url, custom_providers)

    def codex_account_headers(access_token: str):
        from agent.codex_headers import codex_account_headers as _headers
        return _headers(access_token)

    def bedrock_static(model: str, *, probe: bool = False):
        from agent.bedrock_adapter import get_bedrock_context_length
        return get_bedrock_context_length(model, probe=probe)

    def bedrock_probe(model: str, region: str):
        from agent.bedrock_adapter import probe_bedrock_context_length
        return probe_bedrock_context_length(model, region)

    def bedrock_region():
        from agent.bedrock_adapter import resolve_bedrock_region
        return resolve_bedrock_region()

    def moa_context_length(model: str, custom_providers: list | None):
        from hermes_cli.config import get_compatible_custom_providers, load_config
        from hermes_cli.moa_config import resolve_moa_preset
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from models.metadata.context import get_model_context_length
        config = load_config()
        routes = custom_providers if custom_providers is not None else get_compatible_custom_providers(config)
        agg = resolve_moa_preset(config.get("moa") or {}, model).get("aggregator") or {}
        agg_provider = str(agg.get("provider") or "").strip()
        agg_model = str(agg.get("model") or "").strip()
        if not agg_model or not agg_provider or agg_provider.lower() == "moa":
            return None
        if agg_provider.lower().startswith("custom:"):
            name = agg_provider.split(":", 1)[1]
            row = (config.get("providers") or {}).get(name) if isinstance(config.get("providers"), dict) else None
            model_row = row.get("models", {}).get(agg_model) if isinstance(row, dict) and isinstance(row.get("models"), dict) else None
            if isinstance(model_row, dict) and isinstance(model_row.get("context_length"), int):
                return model_row["context_length"]
            base_url = row.get("api", "") if isinstance(row, dict) else ""
            return get_model_context_length(agg_model, base_url=base_url, provider=agg_provider, custom_providers=routes)
        try:
            runtime = resolve_runtime_provider(requested=agg_provider, target_model=agg_model)
        except Exception:
            runtime = {"provider": agg_provider, "base_url": "https://openrouter.ai/api/v1", "api_key": ""}
        return get_model_context_length(
            agg_model, base_url=runtime.get("base_url", "") or "", api_key=runtime.get("api_key", "") or "",
            provider=runtime.get("provider") or agg_provider, custom_providers=routes,
        )

    def model_override_context_length(provider: str, model: str):
        from agent.models_dev import _override_context_window
        return _override_context_window(provider, model)

    def custom_provider_context_length(*, model: str, base_url: str, custom_providers: list | None):
        from hermes_cli.config import get_custom_provider_context_length
        return get_custom_provider_context_length(model=model, base_url=base_url, custom_providers=custom_providers)

    def copilot_context_length(model: str, *, api_key: str):
        from models.metadata.github import github_model_context_length
        return github_model_context_length(model, api_key=api_key)

    def models_dev_context_length(provider: str, model: str):
        from agent.models_dev import lookup_models_dev_context
        return lookup_models_dev_context(provider, model)

    configure_context_metadata_hooks(
        tls_settings=tls_settings, http_get=http_get, http_stream=http_stream,
        resolve_verify=resolve_verify, is_codex_oauth_token=is_codex_oauth_token,
        materialize_api_key=materialize_api_key,
        fingerprint_api_key=fingerprint_api_key, custom_provider_api_mode=custom_provider_api_mode,
        codex_account_headers=codex_account_headers, bedrock_static=bedrock_static,
        bedrock_probe=bedrock_probe, bedrock_region=bedrock_region,
        moa_context_length=moa_context_length, model_override_context_length=model_override_context_length,
        custom_provider_context_length=custom_provider_context_length, copilot_context_length=copilot_context_length,
        models_dev_context_length=models_dev_context_length,
    )


install_context_metadata_hooks()

def parse_context_limit_from_error(error_msg: str) -> Optional[int]:
    """Context limit quoted in a provider error ("maximum context length is 32768 tokens"), if any.

    A message about only an OUTPUT cap ("... model output limit of 16384") never says "context";
    bail out so the generic "limit ... of N" pattern can't cache the output cap as the window."""
    error_lower = error_msg.lower()
    if ("output limit" in error_lower or "output tokens" in error_lower or "output token" in error_lower) and "context" not in error_lower:
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
    ("limited to",),
)


def _sglang_window_and_input(error_lower: str) -> Optional[Tuple[int, int]]:
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
# CJK/Hangul/Kana codepoints (~1 token each), counted in one C-level regex pass: Hangul
# Jamo (+Ext-A), CJK radicals/ideographs (+compat), Hangul syllables, fullwidth/halfwidth.
# Rough chars-per-token ratio for ASCII text; the single source for every "N tokens ≈ N*4 chars"
# budget conversion (context files, tool-output budgets, whisper prompt cap, compressor metadata).
CHARS_PER_TOKEN = 4

_CJK_DENSE_RE = re.compile("[\u1100-\u11ff\u2e80-\u9fff\ua960-\ua97f\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")


def _is_cjk_token_dense_char(ch: str) -> bool:
    return _CJK_DENSE_RE.fullmatch(ch) is not None


def estimate_tokens_rough(text: str) -> int:
    """Rough token estimate: CJK/Hangul/Kana codepoints ~1 token each; everything else ceil(UTF-8 bytes/4).
    Ceiling keeps short texts from estimating 0. Runs on every preflight walk, so all-ASCII stays O(1).

    Byte-counting (not chars) is the corrective for non-CJK, non-ASCII text: Cyrillic/Greek/Arabic are 2
    bytes/char so count ~chars/2, matching real BPE cost (~2-3 chars/token) where chars/4 under-counted
    ~2x and let sessions ride the provider ceiling below the compaction threshold. Calibrated vs
    cl100k/o200k/Qwen2.5 (estimate/real): Russian 0.67->1.24, Arabic 0.53->0.96, Hindi 0.34->0.90,
    Greek 0.37->0.68; accented Latin barely moves (French 1.02->1.03). errors="replace": lone surrogates
    (routine in tool output; see message_sanitization) must not turn an estimate into a raise."""
    if not text:
        return 0
    text = str(text)
    if text.isascii():  # flag check on CPython; ASCII cannot contain token-dense CJK
        return (len(text) + 3) // CHARS_PER_TOKEN
    stripped = _CJK_DENSE_RE.sub("", text)
    dense = len(text) - len(stripped)
    return dense + ((len(stripped.encode("utf-8", "replace")) + 3) // CHARS_PER_TOKEN)


def estimate_messages_tokens_rough(messages: List[Dict[str, Any]], *, charge_stale_thinking: bool = True) -> int:
    """Rough token estimate for a message list (pre-flight only). Images cost the per-image price
    learned from provider usage (``agent.image_token_cost``; flat default before calibration)
    rather than their base64 length. ``charge_stale_thinking=False`` mirrors the tail-budget
    walk (``context_compressor._estimate_msg_budget_tokens``): on non-echo routes stale reasoning
    rides the wire only for the NEWEST assistant turn, so excluding it keeps the compaction TRIGGER
    in the same size class as the walk — otherwise reasoning-heavy sessions fire preflight forever."""
    from agent.image_token_cost import current_image_token_cost

    image_cost = current_image_token_cost()
    if not charge_stale_thinking:
        messages = _strip_stale_thinking_for_estimate(messages)
    return sum(_estimate_message_tokens_cached(msg, image_cost) for msg in messages)


# Thinking-text keys replayed for at most the newest assistant turn on non-echo routes — must stay
# in lockstep with ``context_compressor._NEWEST_TURN_ONLY_BUDGET_KEYS``.
_STALE_THINKING_ESTIMATE_KEYS = ("reasoning", "reasoning_content")


def _strip_stale_thinking_for_estimate(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy of ``messages`` with stale thinking keys removed (newest kept). Shallow stripped copies
    share the original value objects, so the per-message memo still hits for the stripped shape."""
    def _is_assistant(m: Any) -> bool:
        return isinstance(m, dict) and m.get("role") == "assistant"
    newest = next((i for i in range(len(messages) - 1, -1, -1) if _is_assistant(messages[i])), -1)
    return [
        {k: v for k, v in m.items() if k not in _STALE_THINKING_ESTIMATE_KEYS}
        if i != newest and _is_assistant(m) and any(m.get(k) for k in _STALE_THINKING_ESTIMATE_KEYS) else m
        for i, m in enumerate(messages)
    ]


# Per-message token-estimate memo keyed by an exact value fingerprint: strings by ``id()`` AND
# pinned (strong ref in the entry, so the id can't be reused and immutability makes id-equality
# value-equality); numbers/bools/None by value; dicts/lists structurally in key order (``str(shadow)``
# depends on it); any other type aborts the memo. api_messages shallow-copies dicts but shares the strings.
# ``estimate_messages_tokens_rough`` is called on the full history every loop iteration (conversation_loop
# preflight), repeatedly during compaction telemetry, and inside an O(n^2) shrink loop in moa_loop. The
# per-message helpers are pure functions of the message's value, so a memo keyed on a fingerprint that
# uniquely determines the value is exactly equivalent. Fingerprint design (soundness argument): While the
# entry lives, that id cannot be reused by another object, so id-equality implies object-equality — strings
# are immutable, so value-equality too (no #50372-style aliasing). Equal fingerprints therefore imply
# deep-equal messages built from identical immutable leaves ⇒ identical ``str(shadow)`` bytes ⇒ identical
# estimate. Because the api_messages build shallow-copies history dicts each iteration, the copies share the
# same content strings — so unchanged history messages hit the memo even though the outer dicts are fresh
# objects every turn.
_MSG_TOKENS_CACHE: Dict[Any, Tuple[list, int, int]] = {}  # pins, text tokens, image count
_MSG_TOKENS_CACHE_MAX = 4096


def _msg_fingerprint(value: Any, pins: list) -> Any:
    if value is None or value is True or value is False:
        return value
    t = type(value)
    if t is str:
        pins.append(value)
        return ("s", id(value))
    if t is int or t is float:
        return ("n", t.__name__, value)
    if t is dict:
        return ("d", tuple((_msg_fingerprint(k, pins), _msg_fingerprint(v, pins)) for k, v in value.items()))
    if t is list or t is tuple:
        return ("l" if t is list else "t", tuple(_msg_fingerprint(v, pins) for v in value))
    raise ValueError("unfingerprintable message value")


def _estimate_message_tokens_cached(msg: Any, image_cost: int) -> int:
    """Text tokens + images x ``image_cost``; the memo holds text and image COUNT so a recalibrated
    per-image price re-prices cached rows without invalidating them."""
    def _compute() -> Tuple[int, int]:
        return _estimate_message_tokens_without_images(msg), _count_image_tokens(msg, 1)
    try:
        pins: list = []
        key = _msg_fingerprint(msg, pins)
        hash(key)
    except Exception:
        text, images = _compute()
        return text + images * image_cost
    cached = _MSG_TOKENS_CACHE.get(key)
    if cached is not None:
        return cached[1] + cached[2] * image_cost
    text, images = _compute()
    tokens = text + images * image_cost
    _MSG_TOKENS_CACHE[key] = (pins, text, images)
    while len(_MSG_TOKENS_CACHE) > _MSG_TOKENS_CACHE_MAX:
        try:
            _MSG_TOKENS_CACHE.pop(next(iter(_MSG_TOKENS_CACHE)))
        except (StopIteration, KeyError, RuntimeError):
            break
    return tokens


def _count_parts(parts: Any, types: set) -> int:
    return sum(1 for part in parts if isinstance(part, dict) and part.get("type") in types) if isinstance(parts, list) else 0


_IMAGE_PART_TYPES = frozenset({"image", "image_url", "input_image"})


def _count_image_tokens(msg: Dict[str, Any], cost_per_image: int) -> int:
    """Count image-like content parts in a message; return their token cost."""
    if not isinstance(msg, dict):
        return 0
    content = msg.get("content")
    count = _count_parts(content, _IMAGE_PART_TYPES)
    count += _count_parts(msg.get("_anthropic_content_blocks"), {"image"})
    # Multimodal tool results that haven't been converted yet.
    if isinstance(content, dict) and content.get("_multimodal"):
        count += _count_parts(content.get("content"), {"image", "image_url"})
    # Responses ``function_call_output`` items carry converted tool-result
    # parts under ``output`` (the converter moves chat ``content`` there).
    count += _count_parts(msg.get("output"), _IMAGE_PART_TYPES)
    return count * cost_per_image


def strip_opaque_replay_items(items: Any) -> Any:
    """``codex_reasoning_items`` with ``encrypted_content`` blanked for local token estimation.
    The ciphertext is priced by the provider's own count, never by its bytes (a compaction
    checkpoint alone can be 5M chars, #100611); only real usage prices it."""
    if not isinstance(items, list):
        return items
    return [
        {k: ("" if k == "encrypted_content" else v) for k, v in item.items()} if isinstance(item, dict) else item
        for item in items
    ]


def _wire_message_shadow(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Shadow of a message holding only what the provider actually receives.
    * ``api_content`` SUBSTITUTES ``content`` (mirrors ``turn_context.substitute_api_content`` exactly):
      only a non-empty STRING sidecar on a user/assistant row displaces content; substituting any
      other shape would UNDERcount — the dangerous direction.
    * Base64 images become a placeholder; ``_count_image_tokens`` charges them flat.
    * ``reasoning`` never ships as-is (request builds pop it after optionally promoting it into
      ``reasoning_content``); counting both inflated estimates up to +53%.
    * Opaque provider blobs (``encrypted_content`` on codex reasoning / compaction items) are
      ciphertext the provider prices by its OWN token count, never by bytes; a native compaction
      checkpoint alone can be 5M chars (#100611). They contribute 0 here: only real usage ever
      prices them, and the usage anchor carries that price forward."""
    sidecar = msg.get("api_content")
    sidecar_wins = isinstance(sidecar, str) and bool(sidecar) and msg.get("role") in ("user", "assistant")
    _rc = msg.get("reasoning_content")
    drop_reasoning_dup = isinstance(_rc, str) and bool(_rc.strip())
    shadow: Dict[str, Any] = {}
    for k, v in msg.items():
        if k in ("_anthropic_content_blocks", "reasoning_details") or k in PERSISTENCE_ONLY_MESSAGE_FIELDS or (k == "reasoning" and drop_reasoning_dup):
            continue
        if k == "api_content":
            if sidecar_wins:
                shadow["content"] = v
        elif k == "content" and sidecar_wins:
            continue
        elif k == "content" and isinstance(v, list):
            shadow[k] = [
                {"type": part.get("type"), "image": "[stripped]"}
                if isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES
                else part
                for part in v
            ]
        elif k == "content" and isinstance(v, dict) and v.get("_multimodal"):
            shadow[k] = v.get("text_summary", "")
        elif k == "output" and isinstance(v, list):
            # Responses ``function_call_output`` output parts: strip the image
            # payload like the ``content`` branch above so encoded bytes are
            # priced by the flat per-image model, never as text.
            shadow[k] = [
                {"type": part.get("type"), "image": "[stripped]"}
                if isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES
                else part
                for part in v
            ]
        elif k == "codex_reasoning_items":
            shadow[k] = strip_opaque_replay_items(v)
        elif k == "encrypted_content":  # a Responses reasoning/compaction item passed as a row
            shadow[k] = ""
        else:
            shadow[k] = v
    return shadow


def _estimate_message_tokens_without_images(msg: Dict[str, Any]) -> int:
    """Token estimate for a message shadow with image payloads stripped."""
    return estimate_tokens_rough(str(_wire_message_shadow(msg) if isinstance(msg, dict) else msg))


def estimate_request_tokens_rough(
    messages: List[Dict[str, Any]], *, system_prompt: str = "", tools: Optional[List[Dict[str, Any]]] = None, charge_stale_thinking: bool = True,
) -> int:
    """Rough token estimate for a full request: system prompt + messages + tool schemas (50+ tools
    add 20-30K on their own). ``charge_stale_thinking`` is forwarded — pass False when the route
    provably strips stale thinking (``message_sanitization.stale_thinking_reaches_wire``)."""
    total = estimate_tokens_rough(system_prompt) if system_prompt else 0
    if messages:
        # Positional call: test seams and plugin engines monkeypatch estimate_messages_tokens_rough with (messages)-only signatures.
        total += estimate_messages_tokens_rough(messages) if charge_stale_thinking else estimate_messages_tokens_rough(messages, charge_stale_thinking=False)
    if tools:
        total += _estimate_tools_tokens_rough(tools)
    return total


# Keyed by ``id(tools)``; bounded, oldest-first eviction. Repeated ``str(tools)`` on
# large schemas stalls GUI event loops under GIL pressure.
_TOOLS_TOKENS_CACHE: dict[int, Tuple[int, str, str, int]] = {}
_TOOLS_TOKENS_CACHE_MAX = 256


def _tool_name_for_cache(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    name = fn.get("name") if isinstance(fn, dict) else None
    name = name if isinstance(name, str) else tool.get("name")
    return name if isinstance(name, str) else ""


def _estimate_tools_tokens_rough(tools: List[Dict[str, Any]]) -> int:
    if not tools:
        return 0
    key = id(tools)
    signature = (len(tools), _tool_name_for_cache(tools[0]), _tool_name_for_cache(tools[-1]))
    cached = _TOOLS_TOKENS_CACHE.get(key)
    if cached is not None and cached[:3] == signature:
        return cached[3]
    # Sum the major schema fields (descriptions + parameters dominate).
    total_chars = 0
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        src = fn if isinstance(fn, dict) else tool
        params = src.get("parameters") or {}
        total_chars += sum(len(v) for v in (src.get("name") or "", src.get("description") or "") if isinstance(v, str))
        try:  # JSON is closer to wire size than repr()
            total_chars += len(json.dumps(params, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            total_chars += len(str(params))
    tokens = (total_chars + 3) // 4
    if len(_TOOLS_TOKENS_CACHE) >= _TOOLS_TOKENS_CACHE_MAX:
        _TOOLS_TOKENS_CACHE.pop(next(iter(_TOOLS_TOKENS_CACHE)), None)
    _TOOLS_TOKENS_CACHE[key] = (*signature, tokens)
    return tokens
