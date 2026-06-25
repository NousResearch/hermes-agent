"""Static reference of known provider rate / token limits.

Powers the analytics dashboard's "how close am I to the limit" views. These
are PUBLISHED, REPRESENTATIVE defaults — real limits are account/tier
specific and change often, so every entry carries a source + as-of date and
the UI should present them as reference, not ground truth.

Units:
  * rpm  — requests per minute
  * rpd  — requests per day (None = not published / not enforced)
  * tpm_input / tpm_output — tokens per minute (input / output)
  * tpd  — tokens per day (None = not published)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_AS_OF = "2026-06"

# provider -> reference quota. Values are the common entry tier; higher tiers
# raise these substantially. Keep conservative so warnings fire early.
PROVIDER_QUOTAS: Dict[str, Dict[str, Any]] = {
    "anthropic": {
        "display": "Anthropic",
        "rpm": 50,
        "rpd": None,
        "tpm_input": 40_000,
        "tpm_output": 8_000,
        "tpd": None,
        "tier": "Tier 1 (default)",
        "notes": "Per-model; tiers 2-4 raise RPM/TPM ~2-20x. Usage tier auto-upgrades with spend.",
        "source_url": "https://docs.anthropic.com/en/api/rate-limits",
        "as_of": _AS_OF,
    },
    "openai": {
        "display": "OpenAI",
        "rpm": 500,
        "rpd": 10_000,
        "tpm_input": 200_000,
        "tpm_output": 200_000,
        "tpd": None,
        "tier": "Tier 1 (default)",
        "notes": "Tier-based (1-5) by spend; tpm is combined input+output for most models.",
        "source_url": "https://platform.openai.com/docs/guides/rate-limits",
        "as_of": _AS_OF,
    },
    "openrouter": {
        "display": "OpenRouter",
        "rpm": 200,
        "rpd": None,
        "tpm_input": None,
        "tpm_output": None,
        "tpd": None,
        "tier": "credit-based",
        "notes": "RPM scales with credit balance (~1 req/s per credit); upstream model limits also apply.",
        "source_url": "https://openrouter.ai/docs/limits",
        "as_of": _AS_OF,
    },
    "google": {
        "display": "Google (Gemini)",
        "rpm": 60,
        "rpd": 1_500,
        "tpm_input": 1_000_000,
        "tpm_output": None,
        "tpd": None,
        "tier": "Free / pay-as-you-go default",
        "notes": "Per-model; large TPM but small RPD on lower tiers.",
        "source_url": "https://ai.google.dev/gemini-api/docs/rate-limits",
        "as_of": _AS_OF,
    },
    "groq": {
        "display": "Groq",
        "rpm": 30,
        "rpd": 14_400,
        "tpm_input": 6_000,
        "tpm_output": None,
        "tpd": 500_000,
        "tier": "Free",
        "notes": "Free tier is tight on TPM; on-demand tier raises limits significantly.",
        "source_url": "https://console.groq.com/docs/rate-limits",
        "as_of": _AS_OF,
    },
    "mistral": {
        "display": "Mistral",
        "rpm": 60,
        "rpd": None,
        "tpm_input": 500_000,
        "tpm_output": None,
        "tpd": 1_000_000_000,
        "tier": "default",
        "notes": "Workspace-level token/month and per-second request limits also apply.",
        "source_url": "https://docs.mistral.ai/deployment/laplateforme/rate-limit/",
        "as_of": _AS_OF,
    },
}

# Provider-name aliases seen in billing_provider / base_url routing.
_ALIASES = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "azure": "openai",
    "azure_openai": "openai",
    "openrouter": "openrouter",
    "google": "google",
    "gemini": "google",
    "vertex": "google",
    "groq": "groq",
    "mistral": "mistral",
}


def normalize_provider(provider: Optional[str]) -> Optional[str]:
    if not provider:
        return None
    return _ALIASES.get(str(provider).strip().lower())


def get_provider_quota(provider: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the reference quota for a provider (alias-aware), or None."""
    key = normalize_provider(provider)
    if key is None:
        return None
    out = {"provider": key, **PROVIDER_QUOTAS[key]}
    return out


def list_provider_quotas() -> List[Dict[str, Any]]:
    """Return all known provider quotas as a list."""
    return [{"provider": k, **v} for k, v in PROVIDER_QUOTAS.items()]
