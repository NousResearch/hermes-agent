"""
Canonical model catalogs and provider-aware validation helpers.

Add, remove, or reorder entries here — `hermes setup`, `hermes model`, and
runtime transport resolution all use the same provider/model definitions.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Optional

from hermes_cli.transport_profiles import (
    ANTHROPIC_MESSAGES,
    GOOGLE_GENERATE_CONTENT,
    OPENAI_CHAT_COMPLETIONS,
    OPENAI_RESPONSES,
)


@dataclass(frozen=True)
class CuratedModel:
    id: str
    description: str = ""
    transport: Optional[str] = None


# (model_id, display description shown in menus)
OPENROUTER_MODELS: list[tuple[str, str]] = [
    ("anthropic/claude-opus-4.6", "recommended"),
    ("anthropic/claude-sonnet-4.5", ""),
    ("openai/gpt-5.4-pro", ""),
    ("openai/gpt-5.4", ""),
    ("openai/gpt-5.3-codex", ""),
    ("google/gemini-3-pro-preview", ""),
    ("google/gemini-3-flash-preview", ""),
    ("qwen/qwen3.5-plus-02-15", ""),
    ("qwen/qwen3.5-35b-a3b", ""),
    ("stepfun/step-3.5-flash", ""),
    ("z-ai/glm-5", ""),
    ("moonshotai/kimi-k2.5", ""),
    ("minimax/minimax-m2.5", ""),
]

_PROVIDER_CATALOGS: dict[str, tuple[CuratedModel, ...]] = {
    "zai": (
        CuratedModel("glm-5"),
        CuratedModel("glm-4.7"),
        CuratedModel("glm-4.5"),
        CuratedModel("glm-4.5-flash"),
    ),
    "kimi-coding": (
        CuratedModel("kimi-k2.5"),
        CuratedModel("kimi-k2-thinking"),
        CuratedModel("kimi-k2-turbo-preview"),
        CuratedModel("kimi-k2-0905-preview"),
    ),
    "minimax": (
        CuratedModel("MiniMax-M2.5"),
        CuratedModel("MiniMax-M2.5-highspeed"),
        CuratedModel("MiniMax-M2.1"),
    ),
    "minimax-cn": (
        CuratedModel("MiniMax-M2.5"),
        CuratedModel("MiniMax-M2.5-highspeed"),
        CuratedModel("MiniMax-M2.1"),
    ),
    "opencode-go": (
        CuratedModel("glm-5", transport=OPENAI_CHAT_COMPLETIONS),
        CuratedModel("kimi-k2.5", transport=OPENAI_CHAT_COMPLETIONS),
        CuratedModel("minimax-m2.5", transport=ANTHROPIC_MESSAGES),
    ),
    "opencode-zen": (
        CuratedModel("gpt-5.4-pro", transport=OPENAI_RESPONSES),
        CuratedModel("gpt-5.4", transport=OPENAI_RESPONSES),
        CuratedModel("gpt-5.3-codex", transport=OPENAI_RESPONSES),
        CuratedModel("gpt-5.3-codex-spark", transport=OPENAI_RESPONSES),
        CuratedModel("gpt-5.2", transport=OPENAI_RESPONSES),
        CuratedModel("gpt-5.2-codex", transport=OPENAI_RESPONSES),
        CuratedModel("gpt-5.1", transport=OPENAI_RESPONSES),
        CuratedModel("gpt-5.1-codex", transport=OPENAI_RESPONSES),
        CuratedModel("gpt-5.1-codex-max", transport=OPENAI_RESPONSES),
        CuratedModel("gpt-5.1-codex-mini", transport=OPENAI_RESPONSES),
        CuratedModel("gpt-5", transport=OPENAI_RESPONSES),
        CuratedModel("gpt-5-codex", transport=OPENAI_RESPONSES),
        CuratedModel("gpt-5-nano", transport=OPENAI_RESPONSES),
        CuratedModel("claude-opus-4-6", transport=ANTHROPIC_MESSAGES),
        CuratedModel("claude-opus-4-5", transport=ANTHROPIC_MESSAGES),
        CuratedModel("claude-opus-4-1", transport=ANTHROPIC_MESSAGES),
        CuratedModel("claude-sonnet-4-6", transport=ANTHROPIC_MESSAGES),
        CuratedModel("claude-sonnet-4-5", transport=ANTHROPIC_MESSAGES),
        CuratedModel("claude-sonnet-4", transport=ANTHROPIC_MESSAGES),
        CuratedModel("claude-haiku-4-5", transport=ANTHROPIC_MESSAGES),
        CuratedModel("claude-3-5-haiku", transport=ANTHROPIC_MESSAGES),
        CuratedModel("gemini-3.1-pro", transport=GOOGLE_GENERATE_CONTENT),
        CuratedModel("gemini-3-pro", transport=GOOGLE_GENERATE_CONTENT),
        CuratedModel("gemini-3-flash", transport=GOOGLE_GENERATE_CONTENT),
        CuratedModel("minimax-m2.5", transport=OPENAI_CHAT_COMPLETIONS),
        CuratedModel("minimax-m2.5-free", transport=OPENAI_CHAT_COMPLETIONS),
        CuratedModel("minimax-m2.1", transport=OPENAI_CHAT_COMPLETIONS),
        CuratedModel("glm-5", transport=OPENAI_CHAT_COMPLETIONS),
        CuratedModel("glm-4.7", transport=OPENAI_CHAT_COMPLETIONS),
        CuratedModel("glm-4.6", transport=OPENAI_CHAT_COMPLETIONS),
        CuratedModel("kimi-k2.5", transport=OPENAI_CHAT_COMPLETIONS),
        CuratedModel("kimi-k2-thinking", transport=OPENAI_CHAT_COMPLETIONS),
        CuratedModel("kimi-k2", transport=OPENAI_CHAT_COMPLETIONS),
        CuratedModel("qwen3-coder", transport=OPENAI_CHAT_COMPLETIONS),
        CuratedModel("big-pickle", transport=OPENAI_CHAT_COMPLETIONS),
    ),
}

_PROVIDER_MODELS: dict[str, list[str]] = {
    provider: [entry.id for entry in entries]
    for provider, entries in _PROVIDER_CATALOGS.items()
}

_STRICT_CURATED_PROVIDERS = {"opencode-go", "opencode-zen"}
_LIVE_DISCOVERY_PROVIDERS = {"openrouter", "custom", "opencode-zen"}

_PROVIDER_LABELS = {
    "openrouter": "OpenRouter",
    "openai-codex": "OpenAI Codex",
    "nous": "Nous Portal",
    "nous-api": "Nous Portal API Key",
    "zai": "Z.AI / GLM",
    "kimi-coding": "Kimi / Moonshot",
    "minimax": "MiniMax",
    "minimax-cn": "MiniMax (China)",
    "opencode-go": "OpenCode Go",
    "opencode-zen": "OpenCode Zen",
    "custom": "Custom endpoint",
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
    "opencode_go": "opencode-go",
    "opencode-go": "opencode-go",
    "opencode_zen": "opencode-zen",
    "opencode-zen": "opencode-zen",
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


def provider_label(provider: Optional[str]) -> str:
    normalized = normalize_provider(provider)
    return _PROVIDER_LABELS.get(normalized, normalized)


def curated_model_specs(provider: Optional[str]) -> list[CuratedModel]:
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return [CuratedModel(mid, desc, OPENAI_CHAT_COMPLETIONS) for mid, desc in OPENROUTER_MODELS]
    return list(_PROVIDER_CATALOGS.get(normalized, ()))


def curated_models_for_provider(provider: Optional[str]) -> list[tuple[str, str]]:
    """Return ``(model_id, description)`` tuples for a provider's curated list."""
    return [(entry.id, entry.description) for entry in curated_model_specs(provider)]


def normalize_provider(provider: Optional[str]) -> str:
    """Normalize provider aliases to Hermes' canonical provider ids.

    Note: ``"auto"`` passes through unchanged — use
    ``hermes_cli.auth.resolve_provider()`` to resolve it to a concrete
    provider based on credentials and environment.
    """
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
    return [entry.id for entry in curated_model_specs(normalized)]


def provider_transport_for_model(provider: Optional[str], model: Optional[str]) -> Optional[str]:
    normalized = normalize_provider(provider)
    requested = (model or "").strip()
    if not requested:
        return None
    for entry in curated_model_specs(normalized):
        if entry.id == requested:
            return entry.transport
    return None


def provider_requires_explicit_model_mapping(provider: Optional[str]) -> bool:
    return normalize_provider(provider) in _STRICT_CURATED_PROVIDERS


def provider_uses_live_model_discovery(provider: Optional[str]) -> bool:
    return normalize_provider(provider) in _LIVE_DISCOVERY_PROVIDERS


# All provider IDs and aliases that are valid for the provider:model syntax.
_KNOWN_PROVIDER_NAMES: set[str] = (
    set(_PROVIDER_LABELS.keys())
    | set(_PROVIDER_ALIASES.keys())
    | {"openrouter", "custom"}
)


def list_available_providers() -> list[dict[str, str]]:
    """Return info about all providers the user could use with ``provider:model``.

    Each dict has ``id``, ``label``, and ``aliases``.
    Checks which providers have valid credentials configured.
    """
    provider_order = [
        "openrouter",
        "opencode-go",
        "opencode-zen",
        "nous",
        "nous-api",
        "openai-codex",
        "zai",
        "kimi-coding",
        "minimax",
        "minimax-cn",
    ]
    aliases_for: dict[str, list[str]] = {}
    for alias, canonical in _PROVIDER_ALIASES.items():
        aliases_for.setdefault(canonical, []).append(alias)

    result = []
    for pid in provider_order:
        label = _PROVIDER_LABELS.get(pid, pid)
        alias_list = aliases_for.get(pid, [])
        has_creds = False
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider

            runtime = resolve_runtime_provider(requested=pid)
            has_creds = bool(runtime.get("api_key"))
        except Exception:
            pass
        result.append(
            {
                "id": pid,
                "label": label,
                "aliases": alias_list,
                "authenticated": has_creds,
            }
        )
    return result


def parse_model_input(raw: str, current_provider: str) -> tuple[str, str]:
    """Parse ``/model`` input into ``(provider, model)``."""
    stripped = raw.strip()
    colon = stripped.find(":")
    if colon > 0:
        provider_part = stripped[:colon].strip().lower()
        model_part = stripped[colon + 1:].strip()
        if provider_part and model_part and provider_part in _KNOWN_PROVIDER_NAMES:
            return (normalize_provider(provider_part), model_part)
    return (current_provider, stripped)


def fetch_api_models(
    api_key: Optional[str],
    base_url: Optional[str],
    timeout: float = 5.0,
) -> Optional[list[str]]:
    """Fetch model IDs from the provider's ``/models`` endpoint when supported."""
    if not base_url:
        return None

    url = base_url.rstrip("/") + "/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return [m.get("id", "") for m in data.get("data", [])]
    except Exception:
        return None


def _invalid_model_response(
    *,
    requested: str,
    provider: str,
    known_models: list[str],
    message_prefix: str,
) -> dict[str, Any]:
    suggestions = get_close_matches(requested, known_models, n=3, cutoff=0.5)
    suggestion_text = ""
    if suggestions:
        suggestion_text = "\n  Did you mean: " + ", ".join(f"`{s}`" for s in suggestions)
    return {
        "accepted": False,
        "persist": False,
        "recognized": False,
        "message": f"{message_prefix}{suggestion_text}",
    }


def validate_requested_model(
    model_name: str,
    provider: Optional[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Validate a ``/model`` value for the active provider.

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

    provider_label_text = _PROVIDER_LABELS.get(normalized, normalized)
    known_models = provider_model_ids(normalized)

    if provider_requires_explicit_model_mapping(normalized):
        if requested not in known_models:
            return _invalid_model_response(
                requested=requested,
                provider=normalized,
                known_models=known_models,
                message_prefix=(
                    f"Error: `{requested}` is not a supported model for {provider_label_text}. "
                    "Choose one of the curated OpenCode models."
                ),
            )

        if provider_uses_live_model_discovery(normalized):
            api_models = fetch_api_models(api_key, base_url)
            if api_models is not None and requested not in set(api_models):
                return _invalid_model_response(
                    requested=requested,
                    provider=normalized,
                    known_models=[mid for mid in known_models if mid in set(api_models)] or known_models,
                    message_prefix=f"Error: `{requested}` is not currently available from {provider_label_text}.",
                )

        return {
            "accepted": True,
            "persist": True,
            "recognized": True,
            "message": None,
        }

    api_models = fetch_api_models(api_key, base_url)

    if api_models is not None:
        if requested in set(api_models):
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": None,
            }
        return _invalid_model_response(
            requested=requested,
            provider=normalized,
            known_models=api_models,
            message_prefix=f"Error: `{requested}` is not a valid model for this provider.",
        )

    if requested in known_models:
        return {
            "accepted": True,
            "persist": True,
            "recognized": True,
            "message": None,
        }

    suggestion = get_close_matches(requested, known_models, n=1, cutoff=0.6)
    suggestion_text = f" Did you mean `{suggestion[0]}`?" if suggestion else ""
    return {
        "accepted": True,
        "persist": False,
        "recognized": False,
        "message": (
            f"Could not validate `{requested}` against the live {provider_label_text} API. "
            "Using it for this session only; config unchanged."
            f"{suggestion_text}"
        ),
    }
