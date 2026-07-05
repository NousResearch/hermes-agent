"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances and OpenAI-compatible reasoning endpoints (GLM-5.2 on
Volcengine ARK, vLLM, llama.cpp). Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled → extra_body.think = False
  - reasoning_config enabled + effort → top-level reasoning_effort
    (the native OpenAI-compatible format GLM/ARK expect; unset omits it
    so the endpoint's server default applies)
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


# vLLM / llama.cpp validate ``reasoning_effort`` against the OpenAI set
# {none, low, medium, high} and return a non-retryable HTTP 400 on Hermes-only
# levels (minimal, xhigh, max). Clamp those to the nearest accepted value for
# *detected* vLLM / llama.cpp only. GLM-5.2 / ARK (also provider=custom)
# legitimately accept "high"/"max", so their effort passes through untouched.
_VLLM_EFFORT_CLAMP = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false and num_ctx support."""

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
        #   - disabled  → extra_body.think = False (Ollama's thinking-off flag).
        #     For detected vLLM / llama.cpp also emit
        #     chat_template_kwargs.enable_thinking = False — the ONLY key those
        #     backends honor to turn Qwen3-style reasoning off (they ignore
        #     ``think``; see vLLM reasoning-outputs docs).
        #   - enabled + effort set → TOP-LEVEL reasoning_effort string, the
        #     format GLM-5.2/ARK and other OpenAI-compatible reasoning APIs
        #     expect (GLM documents "high" and "max"; "max" is its default).
        #     For detected vLLM / llama.cpp also emit enable_thinking = True so
        #     the chat template actually produces reasoning content.
        #   - enabled + no effort  → omit both, so the endpoint applies its own
        #     server-side default (do NOT force a level the user didn't pick).
        #
        # chat_template_kwargs is scoped to *detected* vLLM / llama.cpp only: it
        # is a chat-template concept those servers understand, and sending it to
        # GLM/ARK (also provider=custom) risks a 400. We keep the deliberate
        # choice to NOT emit ``think=True`` on enable (Ollama-only flag).
        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)

            # Detect vLLM / llama.cpp so chat-template flags only go to backends
            # that understand them. base_url is passed by the chat_completions
            # transport (see agent/transports/chat_completions.py). The probe is
            # a lightweight local /version + /models GET (~8ms warm on a LAN box)
            # and is best-effort: any failure leaves _server_type None so we fall
            # back to the safe GLM/ARK path (no chat_template_kwargs emitted).
            _base_url = ctx.get("base_url") or self.base_url or ""
            _server_type = None
            if _base_url:
                try:
                    from agent.model_metadata import detect_local_server_type
                    _server_type = detect_local_server_type(_base_url)
                except Exception:
                    _server_type = None
            _templated = _server_type in ("vllm", "llamacpp")

            if _effort == "none" or _enabled is False:
                extra_body["think"] = False
                if _templated:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            elif _effort:
                # Templated backends (vLLM / llama.cpp) reject Hermes-only
                # levels (minimal/xhigh/max) with a non-retryable 400 — clamp
                # to the nearest OpenAI-set value. GLM/ARK keep their raw level.
                if _templated:
                    top_level["reasoning_effort"] = _VLLM_EFFORT_CLAMP.get(
                        _effort, "high"
                    )
                    extra_body["chat_template_kwargs"] = {"enable_thinking": True}
                else:
                    top_level["reasoning_effort"] = _effort

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
