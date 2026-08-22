"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances and OpenAI-compatible reasoning endpoints (GLM-5.2 on
Volcengine ARK, vLLM, llama.cpp). Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled → top-level reasoning_effort="none"
    (Ollama /v1/chat/completions ignores think=False — ollama#14820)
    + extra_body.think = False for /api/chat and proxies
  - reasoning_config enabled + effort → top-level reasoning_effort
    (the native OpenAI-compatible format GLM/ARK expect; unset omits it
    so the endpoint's server default applies)
  - providers.<name>.reasoning_format optionally overrides the wire shape
    per endpoint (#72649): "reasoning_object" nests the config as
    extra_body.reasoning = {"enabled": ..., "effort": ...} for
    OpenRouter-style gateways, "none" suppresses reasoning fields entirely
    for proxies that 400 on unknown params, and "top_level" (the default)
    keeps the behavior above.
"""

import logging
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


logger = logging.getLogger(__name__)


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false and num_ctx support."""

    @staticmethod
    def _resolve_reasoning_format(base_url: Any) -> str | None:
        """Look up ``providers.<name>.reasoning_format`` for this endpoint.

        Returns one of ``"top_level"`` / ``"reasoning_object"`` / ``"none"``,
        or ``None`` when unset/unavailable — ``None`` and ``"top_level"``
        both mean the historical default wire shape.
        """
        if not base_url or not isinstance(base_url, str):
            return None
        try:
            from hermes_cli.config import get_custom_provider_reasoning_format

            return get_custom_provider_reasoning_format(base_url)
        except Exception:
            return None

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # Ollama context window
        if ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_num_ctx
            extra_body["options"] = options

        # Reasoning / thinking control for custom OpenAI-compatible endpoints
        # (GLM-5.2 on Volcengine ARK, vLLM, Ollama, llama.cpp, …).
        #
        #   - disabled  → extra_body.think = False (Ollama's thinking-off flag)
        #   - enabled + effort set → TOP-LEVEL reasoning_effort string, the
        #     format GLM-5.2/ARK and other OpenAI-compatible reasoning APIs
        #     expect (GLM documents "high" and "max"; "max" is its default).
        #   - enabled + no effort  → omit both, so the endpoint applies its own
        #     server-side default (do NOT force a level the user didn't pick).
        #
        # We deliberately do NOT emit ``think=True`` on enable: it is an
        # Ollama-only flag and thinking is already server-default-on for these
        # backends, so forcing it risks a 400 on GLM/vLLM endpoints that don't
        # recognize it. Mirrors the DeepSeek/Zai profile precedent.
        #
        # ``providers.<name>.reasoning_format`` overrides the wire shape per
        # endpoint (#72649): "reasoning_object" nests the same intent as
        # extra_body.reasoning = {"enabled": ..., "effort": ...}
        # (OpenRouter-style gateways), "none" emits no reasoning fields at
        # all (proxies that 400 on unknown params). Unset/"top_level" keeps
        # the historical behavior above byte-for-byte.
        if reasoning_config and isinstance(reasoning_config, dict):
            _format = self._resolve_reasoning_format(ctx.get("base_url"))
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)
            if _effort == "none" or _enabled is False:
                if _format == "none":
                    logger.warning(
                        "reasoning_format='none' suppresses the explicit reasoning "
                        "disable request; the endpoint's server-side default applies"
                    )
                elif _format == "reasoning_object":
                    extra_body["reasoning"] = {"enabled": False}
                else:
                    # Ollama's /v1/chat/completions silently ignores
                    # extra_body.think (only /api/chat honours it — ollama#14820)
                    # but respects the top-level reasoning_effort field, so both
                    # are needed to actually stop a thinking-capable model from
                    # reasoning (#25758). Endpoints that recognize neither simply
                    # ignore them.
                    top_level["reasoning_effort"] = "none"
                    extra_body["think"] = False
            elif _effort:
                if _format == "none":
                    pass
                elif _format == "reasoning_object":
                    extra_body["reasoning"] = {"enabled": True, "effort": _effort}
                else:
                    # Clamp the internal ladder onto the widest OpenAI-compatible
                    # wire vocabulary (shared policy in agent.reasoning_effort) —
                    # GLM/ARK, vLLM and SGLang all top out at "max"; forwarding
                    # "ultra" verbatim is a guaranteed 400 (#89503).
                    from agent.reasoning_effort import (
                        OPENAI_COMPAT_WIRE_EFFORTS,
                        clamp_effort,
                    )

                    top_level["reasoning_effort"] = clamp_effort(
                        _effort, OPENAI_COMPAT_WIRE_EFFORTS
                    )

        return extra_body, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Custom/Ollama: base_url is user-configured; fetch if set."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


custom = CustomProfile(
    name="custom",
    aliases=(
        "ollama",
        "local",
        "vllm",
        "llamacpp",
        "llama.cpp",
        "llama-cpp",
    ),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
    # Without this, no max_tokens is sent and Ollama falls back to its internal
    # num_predict=128, truncating responses after a few tokens (#39281). This is
    # only a floor used when the user hasn't set model.max_tokens — they can
    # override per-model — so we set it generously rather than lowballing it.
    default_max_tokens=65536,
)

register_provider(custom)
