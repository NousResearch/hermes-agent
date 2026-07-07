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


# Substrings that mark a model as supporting ``reasoning_effort`` on a
# custom OpenAI-compatible endpoint (#59660). Substring (not exact)
# match so vendor prefixes (``z-ai/glm-5.2``), alias spellings
# (``glm-5-2``, ``glm-5p2``), and the canonical name all hit. Tokens are
# deliberately conservative — every name on this list is a known
# reasoning-capable model family. A miss is benign (omits the field
# rather than 400ing the request); a false positive would 400.
_REASONING_MODEL_TOKENS: tuple = (
    "glm-4.5", "glm-4.6", "glm-5",     # GLM 4.5+ and 5.x (the Z.AI family)
    "o1", "o3", "o4",                   # OpenAI reasoning
    "qwq",                              # Qwen QwQ
    "qwen3",                            # Qwen3 (Qwen's hybrid reasoning model)
    "deepseek-r",                       # DeepSeek-R
    "claude-sonnet-4.5", "claude-opus-4.5",  # Claude 4.5+
    "magistral",                        # Mistral reasoning
    "gemma-3", "gemma-4",               # Gemma 3/4 — reasoning-capable per user
                                       # testing on ollama/unsloth/gemma-4-26B
                                       # (added per PR #59678 review comment)
)


def _model_supports_reasoning_effort(model: str | None) -> bool:
    """Return True when the model is known to accept ``reasoning_effort``.

    Used by :meth:`CustomProfile.build_api_kwargs_extras` to gate
    forwarding of the user's reasoning-effort preference. Models not on
    this list produce an empty ``top_level`` dict so the endpoint's server
    default applies — a graceful degradation rather than a 400 on an
    unknown field.

    The list is intentionally conservative. We only mark a model as
    "supports reasoning_effort" when we've verified it. Plain models
    (llama, mistral, qwen2.5, gemma, phi-3, etc.) are excluded by design
    — the field is not just ignored on these models, it actively 400s
    some local Ollama builds.
    """
    if not model:
        # No model name → no way to know. Default to safe: omit the
        # field. The user's primary cloud model will still get the
        # field via a different provider profile (OpenRouter/Nous).
        return False
    m = model.strip().lower()
    return any(token in m for token in _REASONING_MODEL_TOKENS)


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false and num_ctx support."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        model: str | None = None,
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
        #   - enabled + effort set + model is reasoning-capable → TOP-LEVEL
        #     reasoning_effort string, the format GLM-5.2/ARK and other
        #     OpenAI-compatible reasoning APIs expect (GLM documents "high"
        #     and "max"; "max" is its default).
        #   - enabled + no effort  → omit both, so the endpoint applies its own
        #     server-side default (do NOT force a level the user didn't pick).
        #   - enabled + effort set + model is NOT reasoning-capable → omit too.
        #     Forwarding ``reasoning_effort`` to a non-reasoning model 400s
        #     the request (the local Ollama plain models reject the field
        #     outright). The user's primary/interactive model is usually a
        #     cloud reasoning-capable model via OpenRouter/Nous; the bug
        #     surfaces only on cross-provider fallback to a plain model.
        #     Safer to omit than to 400. (#59660)
        #
        # We deliberately do NOT emit ``think=True`` on enable: it is an
        # Ollama-only flag and thinking is already server-default-on for these
        # backends, so forcing it risks a 400 on GLM/vLLM endpoints that don't
        # recognize it. Mirrors the DeepSeek/Zai profile precedent.
        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)
            if _effort == "none" or _enabled is False:
                extra_body["think"] = False
            elif _effort and _model_supports_reasoning_effort(model):
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
