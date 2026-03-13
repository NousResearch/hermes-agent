"""
Canonical model catalogs and lightweight validation helpers.

Add, remove, or reorder entries here — both `hermes setup` and
`hermes` provider-selection will pick up the change automatically.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
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
    "nous": [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "gpt-5.4",
        "gemini-3-flash",
        "gemini-3.0-pro-preview",
        "deepseek-v3.2",
    ],
    "openai-codex": [
        "gpt-5.2-codex",
        "gpt-5.1-codex-mini",
        "gpt-5.1-codex-max",
    ],
    "zai": [
        "glm-5",
        "glm-4.7",
        "glm-4.5",
        "glm-4.5-flash",
    ],
    "kimi-coding": [
        "kimi-for-coding",
        "kimi-k2.5",
        "kimi-k2-thinking",
        "kimi-k2-thinking-turbo",
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
    "anthropic": [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
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
    "anthropic": "Anthropic",
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
    "claude": "anthropic",
    "claude-code": "anthropic",
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
    # Canonical providers in display order
    _PROVIDER_ORDER = [
        "openrouter", "nous", "openai-codex",
        "zai", "kimi-coding", "minimax", "minimax-cn", "anthropic",
    ]
    # Build reverse alias map
    aliases_for: dict[str, list[str]] = {}
    for alias, canonical in _PROVIDER_ALIASES.items():
        aliases_for.setdefault(canonical, []).append(alias)

    result = []
    for pid in _PROVIDER_ORDER:
        label = _PROVIDER_LABELS.get(pid, pid)
        alias_list = aliases_for.get(pid, [])
        # Check if this provider has credentials available
        has_creds = False
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider
            runtime = resolve_runtime_provider(requested=pid)
            has_creds = bool(runtime.get("api_key"))
        except Exception:
            pass
        result.append({
            "id": pid,
            "label": label,
            "aliases": alias_list,
            "authenticated": has_creds,
        })
    return result


def parse_model_input(raw: str, current_provider: str) -> tuple[str, str]:
    """Parse ``/model`` input into ``(provider, model)``.

    Supports ``provider:model`` syntax to switch providers at runtime::

        openrouter:anthropic/claude-sonnet-4.5  →  ("openrouter", "anthropic/claude-sonnet-4.5")
        nous:hermes-3                           →  ("nous", "hermes-3")
        anthropic/claude-sonnet-4.5             →  (current_provider, "anthropic/claude-sonnet-4.5")
        gpt-5.4                                 →  (current_provider, "gpt-5.4")

    The colon is only treated as a provider delimiter if the left side is a
    recognized provider name or alias.  This avoids misinterpreting model names
    that happen to contain colons (e.g. ``anthropic/claude-3.5-sonnet:beta``).

    Returns ``(provider, model)`` where *provider* is either the explicit
    provider from the input or *current_provider* if none was specified.
    """
    stripped = raw.strip()
    colon = stripped.find(":")
    if colon > 0:
        provider_part = stripped[:colon].strip().lower()
        model_part = stripped[colon + 1:].strip()
        if provider_part and model_part and provider_part in _KNOWN_PROVIDER_NAMES:
            return (normalize_provider(provider_part), model_part)
    return (current_provider, stripped)


def curated_models_for_provider(provider: Optional[str]) -> list[tuple[str, str]]:
    """Return ``(model_id, description)`` tuples for a provider's model list.

    Tries to fetch the live model list from the provider's API first,
    falling back to the static ``_PROVIDER_MODELS`` catalog if the API
    is unreachable.
    """
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return list(OPENROUTER_MODELS)

    # Try live API first (Codex, Nous, etc. all support /models)
    live = provider_model_ids(normalized)
    if live:
        return [(m, "") for m in live]

    # Fallback to static catalog
    models = _PROVIDER_MODELS.get(normalized, [])
    return [(m, "") for m in models]


def normalize_provider(provider: Optional[str]) -> str:
    """Normalize provider aliases to Hermes' canonical provider ids.

    Note: ``"auto"`` passes through unchanged — use
    ``hermes_cli.auth.resolve_provider()`` to resolve it to a concrete
    provider based on credentials and environment.
    """
    normalized = (provider or "openrouter").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def provider_model_ids(provider: Optional[str]) -> list[str]:
    """Return the best known model catalog for a provider.

    Tries live API endpoints for providers that support them (Codex, Nous),
    falling back to static lists.
    """
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return model_ids()
    if normalized == "openai-codex":
        from hermes_cli.codex_models import get_codex_model_ids

        return get_codex_model_ids()
    if normalized == "nous":
        # Try live Nous Portal /models endpoint
        try:
            from hermes_cli.auth import fetch_nous_models, resolve_nous_runtime_credentials
            creds = resolve_nous_runtime_credentials()
            if creds:
                live = fetch_nous_models(creds.get("api_key", ""), creds.get("base_url", ""))
                if live:
                    return live
        except Exception:
            pass
    if normalized == "anthropic":
        live = _fetch_anthropic_models()
        if live:
            return live
    return list(_PROVIDER_MODELS.get(normalized, []))


def _fetch_anthropic_models(timeout: float = 5.0) -> Optional[list[str]]:
    """Fetch available models from the Anthropic /v1/models endpoint.

    Uses resolve_anthropic_token() to find credentials (env vars or
    Claude Code auto-discovery).  Returns sorted model IDs or None.
    """
    try:
        from agent.anthropic_adapter import resolve_anthropic_token, _is_oauth_token
    except ImportError:
        return None

    token = resolve_anthropic_token()
    if not token:
        return None

    headers: dict[str, str] = {"anthropic-version": "2023-06-01"}
    if _is_oauth_token(token):
        headers["Authorization"] = f"Bearer {token}"
        headers["anthropic-beta"] = "oauth-2025-04-20"
    else:
        headers["x-api-key"] = token

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/models",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            models = [m["id"] for m in data.get("data", []) if m.get("id")]
            # Sort: latest/largest first (opus > sonnet > haiku, higher version first)
            return sorted(models, key=lambda m: (
                "opus" not in m,      # opus first
                "sonnet" not in m,    # then sonnet
                "haiku" not in m,     # then haiku
                m,                    # alphabetical within tier
            ))
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("Failed to fetch Anthropic models: %s", e)
        return None


def fetch_api_models(
    api_key: Optional[str],
    base_url: Optional[str],
    timeout: float = 5.0,
) -> Optional[list[str]]:
    """Fetch the list of available model IDs from the provider's ``/models`` endpoint.

    Returns a list of model ID strings, or ``None`` if the endpoint could not
    be reached (network error, timeout, auth failure, etc.).
    """
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
            # Standard OpenAI format: {"data": [{"id": "model-name", ...}, ...]}
            return [m.get("id", "") for m in data.get("data", [])]
    except Exception:
        return None


def validate_requested_model(
    model_name: str,
    provider: Optional[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Validate a ``/model`` value for the active provider.

    Performs format checks first, then probes the live API to confirm
    the model actually exists.

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

    # Custom endpoints can serve any model — skip validation
    if normalized == "custom":
        return {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": None,
        }

    # Probe the live API to check if the model actually exists
    api_models = fetch_api_models(api_key, base_url)

    if api_models is not None:
        if requested in set(api_models):
            # API confirmed the model exists
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": None,
            }
        else:
            # API responded but model is not listed
            suggestions = get_close_matches(requested, api_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Did you mean: " + ", ".join(f"`{s}`" for s in suggestions)

            return {
                "accepted": False,
                "persist": False,
                "recognized": False,
                "message": (
                    f"Error: `{requested}` is not a valid model for this provider."
                    f"{suggestion_text}"
                ),
            }

    # api_models is None — couldn't reach API, fall back to catalog check
    provider_label = _PROVIDER_LABELS.get(normalized, normalized)
    known_models = provider_model_ids(normalized)

    if requested in known_models:
        return {
            "accepted": True,
            "persist": True,
            "recognized": True,
            "message": None,
        }

    # Can't validate — accept for session only
    suggestion = get_close_matches(requested, known_models, n=1, cutoff=0.6)
    suggestion_text = f" Did you mean `{suggestion[0]}`?" if suggestion else ""
    return {
        "accepted": True,
        "persist": False,
        "recognized": False,
        "message": (
            f"Could not validate `{requested}` against the live {provider_label} API. "
            "Using it for this session only; config unchanged."
            f"{suggestion_text}"
        ),
    }
