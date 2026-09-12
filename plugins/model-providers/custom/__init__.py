"""Custom / Ollama (local) provider profile: any endpoint registered as
provider="custom" (Ollama, vLLM, llama.cpp, GLM-5.2 on ARK, …)."""

import time
from typing import Any
from urllib.parse import urlparse

from agent.reasoning_effort import OPENAI_COMPAT_WIRE_EFFORTS, clamp_effort
from providers import register_provider
from providers.base import ProviderProfile


def _looks_like_ollama_endpoint(base_url: str | None) -> bool:
    """True only for explicit Ollama signatures (port 11434 or an ``ollama`` host label).
    ``think`` is Ollama-native; strict hosts (Mistral, Groq) 422 on it, and
    arbitrary localhost may be llama.cpp / vLLM / LM Studio."""
    raw = (base_url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    try:  # urlparse raises ValueError on malformed ports ("host:99999"); treat as not-Ollama.
        if parsed.port == 11434:
            return True
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    return bool(host) and (host == "ollama.com" or host.endswith(".ollama.com") or "ollama" in host.split("."))


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false and num_ctx support."""

    # (model, base_url) -> (supports_thinking | None when probe failed, monotonic probe time).
    # A definitive True/False is cached for the process lifetime (a model's capability is
    # static; the model name is part of the key); a failed probe is retried after the TTL
    # so a temporarily-down ollama doesn't pin a stale answer. Mirrors ReasoningParamsMixin.
    # Instance attribute: the registry holds this profile as a process-wide singleton.
    _THINKING_PROBE_CACHE: dict[tuple[str, str], tuple[bool | None, float]]
    _THINKING_PROBE_TTL_S = 60.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._THINKING_PROBE_CACHE = {}

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, ollama_num_ctx: int | None = None, **ctx: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        if ollama_num_ctx:
            extra_body["options"] = {"num_ctx": ollama_num_ctx}
        # disabled -> top-level reasoning_effort="none" (Ollama's /v1 ignores
        # extra_body.think) plus think=False only on Ollama URLs; enabled+effort ->
        # top-level reasoning_effort clamped to the OpenAI-compat wire (GLM/ARK,
        # vLLM and SGLang all top out at "max"; "ultra" verbatim 400s); enabled
        # without effort -> omit so the server default applies. Never emit
        # think=True (Ollama-only flag).
        if reasoning_config and isinstance(reasoning_config, dict):
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort == "none" or reasoning_config.get("enabled", True) is False:
                # See #14820.
                top_level["reasoning_effort"] = "none"
                if _looks_like_ollama_endpoint(ctx.get("base_url")):
                    extra_body["think"] = False
            elif effort and not self._endpoint_rejects_thinking(ctx):
                top_level["reasoning_effort"] = clamp_effort(effort, OPENAI_COMPAT_WIRE_EFFORTS)
        return extra_body, top_level

    def _endpoint_rejects_thinking(self, ctx: dict[str, Any]) -> bool:
        """True when the target model cannot think, so ``reasoning_effort`` must be omitted.

        Ollama's /v1/chat/completions 400s on ``reasoning_effort`` when the model
        lacks the ``thinking`` capability (e.g. granite4, ministral-3): "does not
        support thinking". Probe result cached per (model, base_url) — see
        ``_THINKING_PROBE_CACHE``.
        """
        base_url = ctx.get("base_url") or ""
        model = ctx.get("model") or ""
        if not _looks_like_ollama_endpoint(base_url) or not model:
            return False
        key = (model, str(base_url).strip().rstrip("/").lower())
        cached = self._THINKING_PROBE_CACHE.get(key)
        if cached is not None and (
            cached[0] is not None or time.monotonic() - cached[1] < self._THINKING_PROBE_TTL_S
        ):
            return cached[0] is False
        try:
            from hermes_cli.models_local import ollama_model_supports_thinking

            supports = ollama_model_supports_thinking(model, base_url, api_key=ctx.get("api_key"))
        except Exception:
            supports = None
        self._THINKING_PROBE_CACHE[key] = (supports, time.monotonic())
        if supports is True:
            return False  # model thinks — effort is safe to send
        if supports is False:
            return True  # probed OK, no thinking capability — omit the field
        return False  # probe failed (server down?): fail open, send effort

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """base_url is user-configured; fetch only if set."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


custom = CustomProfile(
    name="custom", aliases=("ollama", "local", "vllm", "llamacpp", "llama.cpp", "llama-cpp"),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
    # An arbitrary client ceiling can exceed a local server's actual output limit.
    # The endpoint owns its generation default.
)

register_provider(custom)
