"""Cortecs provider profile.

Cortecs is an EU-hosted multi-provider gateway that fronts multiple LLM backends
(Alibaba Cloud, Deepseek, Google, Meta, MiniMax, Mistral, Moonshot, NousResearch,
Nvidia, OpenAI, Tencent, Xiaomi, Z.ai). The same API key works for all of them.

Endpoint:  https://api.cortecs.ai/v1
Protocol:  OpenAI chat completions (fully OpenAI-compatible)

Key facts verified July 2026 (live wire-probe):
  - All models are available through /v1/chat/completions
  - All models are available through /v1/models for catalog discovery
  - The model ID is just the bare ID from the /v1/models catalog
  - Bearer token auth (Authorization: Bearer <CORTECS_API_KEY>)
  - No special User-Agent required; default OpenAI-compat UA works
  - No temperature quirks; server accepts 0-1 range
  - Vision: only on vision-capable models (e.g. qwen3-vl-235b-a22b)
  - Reasoning effort: not exposed per-model in the API surface yet

The /v1/models endpoint is the canonical source for the model catalog. The
static list below is the same catalog frozen at July 2026 — used only if the
live /v1/models call fails (fetch_models falls through gracefully in that case).
"""

from typing import Any
from urllib.parse import urlparse

from providers import register_provider
from providers.base import ProviderProfile


def _is_confirmed_cortecs_url(base_url: str) -> bool:
    """Return True only for Cortecs's canonical HTTPS API surface."""
    try:
        parsed = urlparse(base_url)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.lower() == "https"
        and (parsed.hostname or "").lower() == "api.cortecs.ai"
        and port in (None, 443)
        and parsed.username is None
        and parsed.password is None
        and parsed.path.rstrip("/") in {"/v1", "/"}
        and not parsed.query
        and not parsed.fragment
    )


class CortecsProfile(ProviderProfile):
    """Cortecs — temperature omitted, plain OpenAI-compat transport."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch the live model list from Cortecs's /v1/models endpoint."""
        effective_base = (base_url or self.base_url or "").rstrip("/")
        confirmed = _is_confirmed_cortecs_url(effective_base)
        if not confirmed:
            # Not a Cortecs endpoint — let the base implementation try anyway
            return super().fetch_models(
                api_key=api_key, base_url=effective_base or None, timeout=timeout
            )
        # Append /models to /v1 → /v1/models
        url = effective_base + "/models"
        import json
        import urllib.request
        from hermes_cli.urllib_security import open_credentialed_url
        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "hermes-agent/1.0")
        for k, v in self.default_headers.items():
            req.add_header(k, v)
        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            items = data if isinstance(data, list) else data.get("data", [])
            return [m["id"] for m in items if isinstance(m, dict) and "id" in m]
        except Exception:
            return None


# A snapshot of the July 2026 catalog, frozen for offline/fallback use.
# The live fetch_models() above is the real-time source of truth.
_FALLBACK_MODELS = (
    # Alibaba Cloud (Qwen)
    "qwen3-coder-30b-a3b-instruct",
    "qwen3-coder-next",
    "qwen3-vl-235b-a22b",
    "qwen3.6-27b",
    # Deepseek
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-v3.2",
    # Google (Gemma)
    "gemma-4-31b-it",
    "gemma-3-27b-it",
    # Meta (Llama)
    "llama-3.3-70b-instruct",
    # MiniMax
    "minimax-m3",
    "minimax-m2.7",
    # Mistral AI
    "mistral-large-2512",
    "mistral-medium-3.5",
    "mistral-nemo-instruct-2407",
    "codestral-2508",
    "devstral-2512",
    "mistral-small-2603",
    # Moonshot AI (Kimi)
    "kimi-k2.7-code",
    "kimi-k2.6",
    # NousResearch
    "hermes-4-70b",
    "hermes-4-405b",
    # Nvidia
    "cosmos3-super-reasoner",
    "nvidia-nemotron-3-nano-omni",
    # OpenAI (open weights)
    "gpt-oss-120b",
    "gpt-oss-20b",
    # Tencent
    "hy3",
    # Xiaomi
    "mimo-v2.5",
    # Z.ai (GLM)
    "glm-5.2",
    "glm-5.1",
    "glm-5",
)


cortecs = CortecsProfile(
    name="cortecs",
    aliases=("cortecs-ai", "cortecs-eu", "cortecs-openrouter"),
    display_name="Cortecs",
    description="Cortecs (EU-hosted multi-provider gateway; one key covers many models)",
    signup_url="https://cortecs.ai",
    env_vars=("CORTECS_API_KEY",),
    base_url="https://api.cortecs.ai/v1",
    models_url="",           # falls back to {base_url}/models
    fixed_temperature=None,   # no special quirk; server accepts 0-1
    default_max_tokens=32000,
    default_headers={},
    default_aux_model="gpt-oss-20b",   # cheap model for aux tasks (0.027 €/1M in)
    supports_vision=True,               # OpenAI-compat; vision on vision-capable models
    fallback_models=_FALLBACK_MODELS,
    hostname="api.cortecs.ai",
)


register_provider(cortecs)
