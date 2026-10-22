"""Model-aware Gemini routing on endpoints that carry several wire protocols."""

from urllib.parse import urlsplit


GEMINI_NATIVE_PROVIDER_NAMES = frozenset({"gemini", "google", "google-gemini", "google-ai-studio"})


def is_opencode_zen_base_url(base_url: str | None) -> bool:
    """An official Zen/Console root, not Go or an arbitrary compatible proxy."""
    url = urlsplit(str(base_url or "").strip())
    return url.hostname == "opencode.ai" and url.path.rstrip("/").lower() in {
        "/zen", "/zen/v1", "/inference", "/inference/v1",
    }


def is_opencode_zen_gemini_route(provider: str | None, model: str | None, base_url: str | None) -> bool:
    """Only Gemini on Zen changes wire; a custom entry's explicit transport still wins."""
    if not is_opencode_zen_base_url(base_url):
        return False
    from agent.opencode_affinity import opencode_transport
    from hermes_cli.models import normalize_opencode_model_id, opencode_provider_family

    if opencode_provider_family(provider) == "opencode-go":
        return False
    bare_model = normalize_opencode_model_id(provider, model).rsplit("/", 1)[-1].lower()
    return bare_model.startswith("gemini-") and opencode_transport(provider, model, base_url)[0] is not None


def is_native_gemini_route(provider: str | None, model: str | None, base_url: str | None) -> bool:
    """Select a client without labeling a mixed-model relay universally Gemini-native."""
    from agent.gemini_native_adapter import is_native_gemini_base_url

    return (
        str(provider or "").strip().lower() in GEMINI_NATIVE_PROVIDER_NAMES
        and is_native_gemini_base_url(base_url or "")
    ) or is_opencode_zen_gemini_route(provider, model, base_url)


def is_native_gemini_request(provider: str | None, model: str | None, base_url: str | None) -> bool:
    """Preserve native Google URL shaping while selecting mixed-relay requests per model."""
    from agent.gemini_native_adapter import is_native_gemini_base_url

    return is_native_gemini_base_url(base_url or "") or is_opencode_zen_gemini_route(provider, model, base_url)


def _is_gemini_native_route(provider_norm: str, effective_base: str, model: str = "") -> bool:
    """Gemini native by provider name, else (best-effort) by base URL shape."""
    if provider_norm in GEMINI_NATIVE_PROVIDER_NAMES:
        return True
    if not effective_base:
        return False
    try:
        from agent.gemini_native_adapter import is_native_gemini_base_url
        return is_native_gemini_base_url(effective_base) or is_opencode_zen_gemini_route(provider_norm, model, effective_base)
    except Exception:
        return False


def project_zen_gemini_reasoning(api_kwargs: dict, extra_body: dict, *, provider, model, base_url, reasoning_config):
    """Relay-native Gemini shares aux thinking controls, not Google's long-decode extension."""
    if not is_opencode_zen_gemini_route(provider, model, base_url):
        return
    from agent.transports.chat_completions import _build_gemini_thinking_config

    api_kwargs.pop("reasoning_effort", None)
    extra_body.pop("max_output_tokens", None)
    thinking = _build_gemini_thinking_config(model.rsplit("/", 1)[-1], reasoning_config)
    if thinking:
        extra_body["thinking_config"] = thinking


def effective_gemini_provider(agent) -> str:
    """Named identity only for the custom billing class; active fallback providers win."""
    provider = getattr(agent, "provider", "")
    requested = getattr(agent, "requested_provider", "")
    return requested if provider == "custom" and requested.startswith("custom:") else provider


def _gemini_native_client(agent, client_kwargs: dict, httpx_verify, *, reason: str, shared: bool):
    """Native Gemini client for the effective provider, model and endpoint, else None."""
    from agent.agent_runtime_helpers import _ra
    from agent.gemini_native_adapter import GeminiNativeClient
    base_url = str(client_kwargs.get("base_url", "") or "")
    if not is_native_gemini_route(effective_gemini_provider(agent), getattr(agent, "model", ""), base_url):
        return None
    safe_kwargs = {
        k: v for k, v in client_kwargs.items()
        if k in {"api_key", "base_url", "default_headers", "default_query", "timeout", "http_client"}
    }
    if "http_client" not in safe_kwargs:
        keepalive_http = agent._build_keepalive_http_client(base_url, verify=httpx_verify)
        if keepalive_http is not None:
            safe_kwargs["http_client"] = keepalive_http
    client = GeminiNativeClient(**safe_kwargs)
    _ra().logger.info(
        "Gemini native client created (%s, shared=%s) %s", reason, shared, agent._client_log_context()
    )
    return client


def _compressor_max_tokens(agent):
    """``agent.max_tokens``, or the native-Gemini adapter default when unset: generateContent
    still sends maxOutputTokens=65,535 and the threshold is pct×(window − max_tokens), so
    reserving 0 let the provider 400 before compaction fired."""
    if getattr(agent, "max_tokens", None) is not None:
        return agent.max_tokens
    from contextlib import suppress
    with suppress(Exception):
        from agent.gemini_native_adapter import (
            GEMINI_DEFAULT_MAX_OUTPUT_TOKENS, is_native_gemini_base_url
        )
        _gemini_provider = str(agent.provider or "").strip().lower() in GEMINI_NATIVE_PROVIDER_NAMES
        if _gemini_provider or is_native_gemini_base_url(agent.base_url) or is_opencode_zen_gemini_route(
            effective_gemini_provider(agent), agent.model, agent.base_url):
            return GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
    return None


def compressor_output_budget_kwargs(agent) -> dict:
    """Match the built-in compressor to the native wire without widening plugin ABI."""
    from agent.context_compressor import ContextCompressor

    return ({"max_tokens": _compressor_max_tokens(agent) or 0}
            if isinstance(agent.context_compressor, ContextCompressor) else {})
