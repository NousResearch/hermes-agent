"""
Canonical model catalogs and lightweight validation helpers.

Add, remove, or reorder entries here — both `hermes setup` and
`hermes` provider-selection will pick up the change automatically.
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import Any, Optional

# (model_id, display description shown in menus)
OPENROUTER_MODELS: list[tuple[str, str]] = [
    ("anthropic/claude-opus-4.6",       "recommended"),
    ("anthropic/claude-sonnet-4.5",     ""),
    ("openai/gpt-5.4-pro",              ""),
    ("openai/gpt-5.4",                  ""),
    ("openai/gpt-5.3-codex",            ""),
    ("google/gemini-3-pro-preview",     ""),
    ("google/gemini-3-flash-preview",   ""),
    ("qwen/qwen3.5-plus-02-15",         ""),
    ("qwen/qwen3.5-35b-a3b",            ""),
    ("stepfun/step-3.5-flash",          ""),
    ("z-ai/glm-5",                      ""),
    ("moonshotai/kimi-k2.5",            ""),
    ("minimax/minimax-m2.5",            ""),
]

_PROVIDER_MODELS: dict[str, list[str]] = {
    "zai": [
        "glm-5",
        "glm-4.7",
        "glm-4.5",
        "glm-4.5-flash",
    ],
    "kimi-coding": [
        "kimi-k2.5",
        "kimi-k2-thinking",
        "kimi-k2-turbo-preview",
        "kimi-k2-0905-preview",
    ],
    "minimax": [
        "MiniMax-M2.5",
        "MiniMax-M2.5-highspeed",
        "MiniMax-M2.1",
    ],
    "minimax-cn": [
        "MiniMax-M2.5",
        "MiniMax-M2.5-highspeed",
        "MiniMax-M2.1",
    ],
}

_PROVIDER_LABELS = {
    "openrouter": "OpenRouter",
    "openai-codex": "OpenAI Codex",
    "nous": "Nous Portal",
    "zai": "Z.AI / GLM",
    "kimi-coding": "Kimi / Moonshot",
    "minimax": "MiniMax",
    "minimax-cn": "MiniMax (China)",
    "custom": "custom endpoint",
}

_PROVIDER_ALIASES = {
    "glm": "zai",
    "z-ai": "zai",
    "z.ai": "zai",
    "zhipu": "zai",
    "kimi": "kimi-coding",
    "moonshot": "kimi-coding",
    "minimax-china": "minimax-cn",
    "minimax_cn": "minimax-cn",
}


def model_ids() -> list[str]:
    """Return just the OpenRouter model-id strings."""
    return [mid for mid, _ in OPENROUTER_MODELS]


def menu_labels() -> list[str]:
    """Return display labels like 'anthropic/claude-opus-4.6 (recommended)'."""
    labels = []
    for mid, desc in OPENROUTER_MODELS:
        labels.append(f"{mid} ({desc})" if desc else mid)
    return labels


def normalize_provider(provider: Optional[str]) -> str:
    """Normalize provider aliases to Hermes' canonical provider ids."""
    normalized = (provider or "openrouter").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def provider_model_ids(provider: Optional[str]) -> list[str]:
    """Return the best known model catalog for a provider."""
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return model_ids()
    if normalized == "openai-codex":
        from hermes_cli.codex_models import get_codex_model_ids

        return get_codex_model_ids()
    return list(_PROVIDER_MODELS.get(normalized, []))


def validate_requested_model(
    model_name: str,
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Validate a `/model` value for the active provider.

    Returns a dict with:
      - accepted: whether the CLI should switch to the requested model now
      - persist: whether it is safe to save to config
      - recognized: whether it matched a known provider catalog
      - message: optional warning / guidance for the user
    """
    requested = (model_name or "").strip()
    normalized = normalize_provider(provider)
    if normalized == "openrouter" and base_url and "openrouter.ai" not in base_url:
        normalized = "custom"

    if not requested:
        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": "Model name cannot be empty.",
        }

    if any(ch.isspace() for ch in requested):
        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": "Model names cannot contain spaces.",
        }

    known_models = provider_model_ids(normalized)
    if requested in known_models:
        return {
            "accepted": True,
            "persist": True,
            "recognized": True,
            "message": None,
        }

    suggestion = get_close_matches(requested, known_models, n=1, cutoff=0.6)
    suggestion_text = f" Did you mean `{suggestion[0]}`?" if suggestion else ""
    provider_label = _PROVIDER_LABELS.get(normalized, normalized)

    if normalized == "custom":
        return {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": None,
        }

    if normalized == "openrouter":
        if "/" not in requested or requested.startswith("/") or requested.endswith("/"):
            return {
                "accepted": False,
                "persist": False,
                "recognized": False,
                "message": (
                    "OpenRouter model IDs should use the `provider/model` format "
                    "(for example `anthropic/claude-opus-4.6`)."
                    f"{suggestion_text}"
                ),
            }
        return {
            "accepted": True,
            "persist": False,
            "recognized": False,
            "message": (
                f"`{requested}` is not in Hermes' curated {provider_label} model list. "
                "Using it for this session only; config unchanged."
                f"{suggestion_text}"
            ),
        }

    if normalized == "nous":
        return {
            "accepted": True,
            "persist": False,
            "recognized": False,
            "message": (
                f"Could not validate `{requested}` against the live {provider_label} catalog here. "
                "Using it for this session only; config unchanged."
                f"{suggestion_text}"
            ),
        }

    if known_models:
        return {
            "accepted": True,
            "persist": False,
            "recognized": False,
            "message": (
                f"`{requested}` is not in the known {provider_label} model list. "
                "Using it for this session only; config unchanged."
                f"{suggestion_text}"
            ),
        }

    return {
        "accepted": True,
        "persist": False,
        "recognized": False,
        "message": (
            f"Could not validate `{requested}` for provider {provider_label}. "
            "Using it for this session only; config unchanged."
            f"{suggestion_text}"
        ),
    }
