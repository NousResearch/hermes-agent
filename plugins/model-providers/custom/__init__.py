"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances and OpenAI-compatible reasoning endpoints (GLM-5.2 on
Volcengine ARK, vLLM, llama.cpp). Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled on a verified Ollama endpoint → top-level
    reasoning_effort="none" + extra_body.think = False
  - reasoning_config effort → top-level reasoning_effort on a verified
    Ollama endpoint only (unset omits it so the server default applies)
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _is_verified_ollama_endpoint(base_url: str | None, api_key: str = "") -> bool:
    """Return true only when the endpoint passes the shared Ollama probe."""
    if not base_url:
        return False
    try:
        from agent.model_metadata import detect_local_server_type

        return detect_local_server_type(base_url, api_key=api_key) == "ollama"
    except Exception:
        return False


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

        # Reasoning / thinking controls are Ollama-specific.  ``custom`` is
        # also used for arbitrary OpenAI-compatible relays (including Groq),
        # which can reject either control.  Do not infer Ollama from a URL
        # substring or its default port: emit these fields only after the
        # shared endpoint probe identifies Ollama.
        #
        #   - disabled → reasoning_effort="none" and extra_body.think=False
        #   - enabled + effort → top-level reasoning_effort
        #   - enabled + no effort → omit both, preserving the server default
        #
        # We deliberately do NOT emit ``think=True`` on enable: it is an
        # Ollama-only flag and thinking is already server-default-on, so
        # forcing it is unnecessary.
        if (
            reasoning_config
            and isinstance(reasoning_config, dict)
            and _is_verified_ollama_endpoint(
                ctx.get("base_url"), str(ctx.get("api_key") or "")
            )
        ):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)
            if _effort == "none" or _enabled is False:
                # Ollama's /v1/chat/completions silently ignores
                # extra_body.think (only /api/chat honours it — ollama#14820)
                # but respects the top-level reasoning_effort field, so both
                # are needed to actually stop a thinking-capable model from
                # reasoning (#25758).
                top_level["reasoning_effort"] = "none"
                extra_body["think"] = False
            elif _effort:
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
