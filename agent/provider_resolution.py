"""Provider-selection helpers for auxiliary LLM clients."""

from __future__ import annotations

import logging
from typing import Any, Callable, cast

logger = logging.getLogger(__name__)

_PROVIDER_ALIASES: dict[str, str] = {
    "google": "gemini",
    "google-gemini": "gemini",
    "google-ai-studio": "gemini",
    "x-ai": "xai",
    "x.ai": "xai",
    "grok": "xai",
    "glm": "zai",
    "z-ai": "zai",
    "z.ai": "zai",
    "zhipu": "zai",
    "kimi": "kimi-coding",
    "moonshot": "kimi-coding",
    "kimi-cn": "kimi-coding-cn",
    "moonshot-cn": "kimi-coding-cn",
    "gmi-cloud": "gmi",
    "gmicloud": "gmi",
    "minimax-china": "minimax-cn",
    "minimax_cn": "minimax-cn",
    "claude": "anthropic",
    "claude-code": "anthropic",
    "github": "copilot",
    "github-copilot": "copilot",
    "github-model": "copilot",
    "github-models": "copilot",
    "github-copilot-acp": "copilot-acp",
    "copilot-acp-agent": "copilot-acp",
    "tencent": "tencent-tokenhub",
    "tokenhub": "tencent-tokenhub",
    "tencent-cloud": "tencent-tokenhub",
    "tencentmaas": "tencent-tokenhub",
}


def _normalize_aux_provider(
    provider: str | None,
    *,
    read_main_provider: Callable[[], str] | None = None,
) -> str:
    normalized = (provider or "auto").strip().lower()
    if normalized.startswith("custom:"):
        suffix = normalized.split(":", 1)[1].strip()
        if not suffix:
            return "custom"
        normalized = suffix
    if normalized == "codex":
        return "openai-codex"
    if normalized == "main":
        if read_main_provider is None:
            from agent.credential_resolution import _read_main_provider

            read_main_provider = _read_main_provider
        main_prov = (read_main_provider() or "").strip().lower()
        if main_prov and main_prov not in {"auto", "main", ""}:
            normalized = main_prov
        else:
            return "custom"
    return _PROVIDER_ALIASES.get(normalized, normalized)


# Sentinel: when returned by _fixed_temperature_for_model(), callers must
# strip the ``temperature`` key from API kwargs entirely so the provider's
# server-side default applies.
OMIT_TEMPERATURE: object = object()


def _is_kimi_model(model: str | None) -> bool:
    """True for any Kimi / Moonshot model that manages temperature server-side."""
    bare = (model or "").strip().lower().rsplit("/", 1)[-1]
    return bare.startswith("kimi-") or bare == "kimi"


def _is_arcee_trinity_thinking(model: str | None) -> bool:
    """True for Arcee Trinity Large Thinking (direct or via OpenRouter)."""
    bare = (model or "").strip().lower().rsplit("/", 1)[-1]
    return bare == "trinity-large-thinking"


def _fixed_temperature_for_model(
    model: str | None,
    base_url: str | None = None,
) -> float | object | None:
    """Return a temperature directive for models with strict contracts."""
    if _is_kimi_model(model):
        logger.debug("Omitting temperature for Kimi model %r (server-managed)", model)
        return OMIT_TEMPERATURE
    if _is_arcee_trinity_thinking(model):
        return 0.5
    return None


def _compression_threshold_for_model(model: str | None) -> float | None:
    """Return a context-compression threshold override for specific models."""
    if _is_arcee_trinity_thinking(model):
        return 0.75
    return None


def _get_aux_model_for_provider(provider_id: str) -> str:
    """Return the cheap auxiliary model for a provider."""
    try:
        from providers import get_provider_profile

        profile = get_provider_profile(provider_id)
        if profile and profile.default_aux_model:
            return str(profile.default_aux_model)
    except Exception:
        pass
    return _API_KEY_PROVIDER_AUX_MODELS_FALLBACK.get(provider_id, "")


_API_KEY_PROVIDER_AUX_MODELS_FALLBACK: dict[str, str] = {
    "gemini": "gemini-3-flash-preview",
    "zai": "glm-4.5-flash",
    "kimi-coding": "kimi-k2-turbo-preview",
    "stepfun": "step-3.5-flash",
    "kimi-coding-cn": "kimi-k2-turbo-preview",
    "gmi": "google/gemini-3.1-flash-lite-preview",
    "minimax": "MiniMax-M2.7",
    "minimax-oauth": "MiniMax-M2.7-highspeed",
    "minimax-cn": "MiniMax-M2.7",
    "anthropic": "claude-haiku-4-5-20251001",
    "ai-gateway": "google/gemini-3-flash",
    "opencode-zen": "gemini-3-flash",
    "opencode-go": "glm-5",
    "kilocode": "google/gemini-3-flash-preview",
    "ollama-cloud": "nemotron-3-nano:30b",
    "tencent-tokenhub": "hy3-preview",
}

_API_KEY_PROVIDER_AUX_MODELS: dict[str, str] = _API_KEY_PROVIDER_AUX_MODELS_FALLBACK

_PROVIDER_VISION_MODELS: dict[str, str] = {
    "xiaomi": "mimo-v2.5",
    "zai": "glm-5v-turbo",
}

_PROVIDERS_WITHOUT_VISION: frozenset[str] = frozenset({
    "kimi-coding",
    "kimi-coding-cn",
})


def _to_openai_base_url(base_url: str | None) -> str:
    """Normalize an Anthropic-style base URL to OpenAI-compatible format."""
    url = str(base_url or "").strip().rstrip("/")
    if url.endswith("/anthropic"):
        if "open.bigmodel.cn" in url or "bigmodel" in url:
            rewritten = url[: -len("/anthropic")] + "/paas/v4"
            logger.debug("Auxiliary client: rewrote ZAI base URL %s -> %s", url, rewritten)
            return rewritten
        rewritten = url[: -len("/anthropic")] + "/v1"
        logger.debug("Auxiliary client: rewrote base URL %s -> %s", url, rewritten)
        return rewritten
    if "api.kimi.com" in url and url.endswith("/coding"):
        rewritten = url + "/v1"
        logger.debug("Auxiliary client: rewrote Kimi base URL %s -> %s", url, rewritten)
        return rewritten
    return url


_AUTO_PROVIDER_LABELS: dict[str, str] = {
    "_try_openrouter": "openrouter",
    "_try_nous": "nous",
    "_try_custom_endpoint": "local/custom",
    "_resolve_api_key_provider": "api-key",
}

_MAIN_RUNTIME_FIELDS: tuple[str, ...] = (
    "provider",
    "model",
    "base_url",
    "api_key",
    "api_mode",
    "auth_mode",
)


def _normalize_main_runtime(main_runtime: dict[str, Any] | None) -> dict[str, Any]:
    """Return a sanitized copy of a live main-runtime override."""
    if not isinstance(main_runtime, dict):
        return {}
    normalized: dict[str, Any] = {}
    for field in _MAIN_RUNTIME_FIELDS:
        value = main_runtime.get(field)
        if field == "api_key" and callable(value) and not isinstance(value, str):
            normalized[field] = value
            continue
        if isinstance(value, str) and value.strip():
            normalized[field] = value.strip()
    provider = normalized.get("provider")
    if isinstance(provider, str):
        normalized["provider"] = provider.lower()
    return normalized


def get_provider_chain(
    try_openrouter: Callable[..., tuple[Any | None, str | None]],
    try_nous: Callable[..., tuple[Any | None, str | None]],
    try_custom_endpoint: Callable[..., tuple[Any | None, str | None]],
    resolve_api_key_provider: Callable[..., tuple[Any | None, str | None]],
) -> list[tuple[str, Callable[..., tuple[Any | None, str | None]]]]:
    """Return the ordered provider detection chain.

    Built by receiving callables from ``auxiliary_client`` so monkeypatches
    against that module's private resolver names still affect tests.
    """
    return [
        ("openrouter", try_openrouter),
        ("nous", try_nous),
        ("local/custom", try_custom_endpoint),
        ("api-key", resolve_api_key_provider),
    ]


def _normalize_resolved_model(model_name: str | None, provider: str) -> str | None:
    """Normalize a resolved model for the provider that will receive it."""
    if not model_name:
        return model_name
    try:
        from hermes_cli.model_normalize import normalize_model_for_provider

        return cast(str | None, normalize_model_for_provider(model_name, provider))
    except Exception:
        return model_name


_VISION_AUTO_PROVIDER_ORDER: tuple[str, ...] = (
    "openrouter",
    "nous",
)

_ANTHROPIC_COMPAT_PROVIDERS: frozenset[str] = frozenset({
    "minimax",
    "minimax-oauth",
    "minimax-cn",
})


def _is_anthropic_compat_endpoint(provider: str, base_url: str) -> bool:
    """Detect if an endpoint expects Anthropic-format content blocks."""
    if provider in _ANTHROPIC_COMPAT_PROVIDERS:
        return True
    url_lower = (base_url or "").lower()
    return "/anthropic" in url_lower
