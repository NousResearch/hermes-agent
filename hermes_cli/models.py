"""
Canonical model catalogs and lightweight validation helpers.

Add, remove, or reorder entries here — both `hermes setup` and
`hermes` provider-selection will pick up the change automatically.
"""

from __future__ import annotations

import json
import os
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
        "gpt-5.3-codex",
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
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "opencode-zen": [
        "gpt-5.4-pro",
        "gpt-5.4",
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
        "gpt-5",
        "gpt-5-codex",
        "gpt-5-nano",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-opus-4-1",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4",
        "claude-haiku-4-5",
        "claude-3-5-haiku",
        "gemini-3.1-pro",
        "gemini-3-pro",
        "gemini-3-flash",
        "minimax-m2.5",
        "minimax-m2.5-free",
        "minimax-m2.1",
        "glm-5",
        "glm-4.7",
        "glm-4.6",
        "kimi-k2.5",
        "kimi-k2-thinking",
        "kimi-k2",
        "qwen3-coder",
        "big-pickle",
    ],
    "opencode-go": [
        "glm-5",
        "kimi-k2.5",
        "minimax-m2.5",
    ],
    "ai-gateway": [
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-haiku-4.5",
        "openai/gpt-5",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "google/gemini-3-pro-preview",
        "google/gemini-3-flash",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "deepseek/deepseek-v3.2",
    ],
    "kilocode": [
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
        "google/gemini-3-pro-preview",
        "google/gemini-3-flash-preview",
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
    "deepseek": "DeepSeek",
    "opencode-zen": "OpenCode Zen",
    "opencode-go": "OpenCode Go",
    "ai-gateway": "AI Gateway",
    "kilocode": "Kilo Code",
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
    "deep-seek": "deepseek",
    "opencode": "opencode-zen",
    "zen": "opencode-zen",
    "go": "opencode-go",
    "opencode-go-sub": "opencode-go",
    "aigateway": "ai-gateway",
    "vercel": "ai-gateway",
    "vercel-ai-gateway": "ai-gateway",
    "kilo": "kilocode",
    "kilo-code": "kilocode",
    "kilo-gateway": "kilocode",
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
        "zai", "kimi-coding", "minimax", "minimax-cn", "kilocode", "anthropic",
        "opencode-zen", "opencode-go",
        "ai-gateway", "deepseek", "custom",
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
            if pid == "custom":
                has_creds = bool(_get_custom_base_url())
            else:
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


def _get_custom_base_url() -> str:
    """Get the custom endpoint base_url from config.yaml."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        model_cfg = config.get("model", {})
        if isinstance(model_cfg, dict):
            return str(model_cfg.get("base_url", "")).strip()
    except Exception:
        pass
    return ""


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


def detect_provider_for_model(
    model_name: str,
    current_provider: str,
) -> Optional[tuple[str, str]]:
    """Auto-detect the best provider for a model name.

    Returns ``(provider_id, model_name)`` — the model name may be remapped
    (e.g. bare ``deepseek-chat`` → ``deepseek/deepseek-chat`` for OpenRouter).
    Returns ``None`` when no confident match is found.

    Priority:
    1. Direct provider with credentials (highest)
    2. Direct provider without credentials → remap to OpenRouter slug
    3. OpenRouter catalog match
    """
    name = (model_name or "").strip()
    if not name:
        return None

    name_lower = name.lower()

    # Aggregators list other providers' models — never auto-switch TO them
    _AGGREGATORS = {"nous", "openrouter"}

    # If the model belongs to the current provider's catalog, don't suggest switching
    current_models = _PROVIDER_MODELS.get(current_provider, [])
    if any(name_lower == m.lower() for m in current_models):
        return None

    # --- Step 1: check static provider catalogs for a direct match ---
    direct_match: Optional[str] = None
    for pid, models in _PROVIDER_MODELS.items():
        if pid == current_provider or pid in _AGGREGATORS:
            continue
        if any(name_lower == m.lower() for m in models):
            direct_match = pid
            break

    if direct_match:
        # Check if we have credentials for this provider
        has_creds = False
        try:
            from hermes_cli.auth import PROVIDER_REGISTRY
            pconfig = PROVIDER_REGISTRY.get(direct_match)
            if pconfig:
                import os
                for env_var in pconfig.api_key_env_vars:
                    if os.getenv(env_var, "").strip():
                        has_creds = True
                        break
        except Exception:
            pass

        if has_creds:
            return (direct_match, name)

        # No direct creds — try to find this model on OpenRouter instead
        or_slug = _find_openrouter_slug(name)
        if or_slug:
            return ("openrouter", or_slug)
        # Still return the direct provider — credential resolution will
        # give a clear error rather than silently using the wrong provider
        return (direct_match, name)

    # --- Step 2: check OpenRouter catalog ---
    # First try exact match (handles provider/model format)
    or_slug = _find_openrouter_slug(name)
    if or_slug:
        if current_provider != "openrouter":
            return ("openrouter", or_slug)
        # Already on openrouter, just return the resolved slug
        if or_slug != name:
            return ("openrouter", or_slug)
        return None  # already on openrouter with matching name

    return None


def _find_openrouter_slug(model_name: str) -> Optional[str]:
    """Find the full OpenRouter model slug for a bare or partial model name.

    Handles:
    - Exact match: ``anthropic/claude-opus-4.6`` → as-is
    - Bare name: ``deepseek-chat`` → ``deepseek/deepseek-chat``
    - Bare name: ``claude-opus-4.6`` → ``anthropic/claude-opus-4.6``
    """
    name_lower = model_name.strip().lower()
    if not name_lower:
        return None

    # Exact match (already has provider/ prefix)
    for mid, _ in OPENROUTER_MODELS:
        if name_lower == mid.lower():
            return mid

    # Try matching just the model part (after the /)
    for mid, _ in OPENROUTER_MODELS:
        if "/" in mid:
            _, model_part = mid.split("/", 1)
            if name_lower == model_part.lower():
                return mid

    return None


def normalize_provider(provider: Optional[str]) -> str:
    """Normalize provider aliases to Hermes' canonical provider ids.

    Note: ``"auto"`` passes through unchanged — use
    ``hermes_cli.auth.resolve_provider()`` to resolve it to a concrete
    provider based on credentials and environment.
    """
    normalized = (provider or "openrouter").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def provider_label(provider: Optional[str]) -> str:
    """Return a human-friendly label for a provider id or alias."""
    original = (provider or "openrouter").strip()
    normalized = original.lower()
    if normalized == "auto":
        return "Auto"
    normalized = normalize_provider(normalized)
    return _PROVIDER_LABELS.get(normalized, original or "OpenRouter")


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
    if normalized == "ai-gateway":
        live = _fetch_ai_gateway_models()
        if live:
            return live
    if normalized == "custom":
        base_url = _get_custom_base_url()
        if base_url:
            # Try common API key env vars for custom endpoints
            api_key = (
                os.getenv("CUSTOM_API_KEY", "")
                or os.getenv("OPENAI_API_KEY", "")
                or os.getenv("OPENROUTER_API_KEY", "")
            )
            live = fetch_api_models(api_key, base_url)
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
        from agent.anthropic_adapter import _COMMON_BETAS, _OAUTH_ONLY_BETAS
        headers["anthropic-beta"] = ",".join(_COMMON_BETAS + _OAUTH_ONLY_BETAS)
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


def probe_api_models(
    api_key: Optional[str],
    base_url: Optional[str],
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Probe an OpenAI-compatible ``/models`` endpoint with light URL heuristics."""
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return {
            "models": None,
            "probed_url": None,
            "resolved_base_url": "",
            "suggested_base_url": None,
            "used_fallback": False,
        }

    if normalized.endswith("/v1"):
        alternate_base = normalized[:-3].rstrip("/")
    else:
        alternate_base = normalized + "/v1"

    candidates: list[tuple[str, bool]] = [(normalized, False)]
    if alternate_base and alternate_base != normalized:
        candidates.append((alternate_base, True))

    tried: list[str] = []
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for candidate_base, is_fallback in candidates:
        url = candidate_base.rstrip("/") + "/models"
        tried.append(url)
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                return {
                    "models": [m.get("id", "") for m in data.get("data", [])],
                    "probed_url": url,
                    "resolved_base_url": candidate_base.rstrip("/"),
                    "suggested_base_url": alternate_base if alternate_base != candidate_base else normalized,
                    "used_fallback": is_fallback,
                }
        except Exception:
            continue

    return {
        "models": None,
        "probed_url": tried[-1] if tried else normalized.rstrip("/") + "/models",
        "resolved_base_url": normalized,
        "suggested_base_url": alternate_base if alternate_base != normalized else None,
        "used_fallback": False,
    }


def _fetch_ai_gateway_models(timeout: float = 5.0) -> Optional[list[str]]:
    """Fetch available language models with tool-use from AI Gateway."""
    api_key = os.getenv("AI_GATEWAY_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = os.getenv("AI_GATEWAY_BASE_URL", "").strip()
    if not base_url:
        from hermes_constants import AI_GATEWAY_BASE_URL
        base_url = AI_GATEWAY_BASE_URL

    url = base_url.rstrip("/") + "/models"
    headers: dict[str, str] = {"Authorization": f"Bearer {api_key}"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return [
                m["id"]
                for m in data.get("data", [])
                if m.get("id")
                and m.get("type") == "language"
                and "tool-use" in (m.get("tags") or [])
            ]
    except Exception:
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
    return probe_api_models(api_key, base_url, timeout=timeout).get("models")


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

    if normalized == "custom":
        probe = probe_api_models(api_key, base_url)
        api_models = probe.get("models")
        if api_models is not None:
            if requested in set(api_models):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }

            suggestions = get_close_matches(requested, api_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)

            message = (
                f"Note: `{requested}` was not found in this custom endpoint's model listing "
                f"({probe.get('probed_url')}). It may still work if the server supports hidden or aliased models."
                f"{suggestion_text}"
            )
            if probe.get("used_fallback"):
                message += (
                    f"\n  Endpoint verification succeeded after trying `{probe.get('resolved_base_url')}`. "
                    f"Consider saving that as your base URL."
                )

            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": message,
            }

        message = (
            f"Note: could not reach this custom endpoint's model listing at `{probe.get('probed_url')}`. "
            f"Hermes will still save `{requested}`, but the endpoint should expose `/models` for verification."
        )
        if probe.get("suggested_base_url"):
            message += f"\n  If this server expects `/v1`, try base URL: `{probe.get('suggested_base_url')}`"

        return {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": message,
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
            # API responded but model is not listed.  Accept anyway —
            # the user may have access to models not shown in the public
            # listing (e.g. Z.AI Pro/Max plans can use glm-5 on coding
            # endpoints even though it's not in /models).  Warn but allow.
            suggestions = get_close_matches(requested, api_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)

            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": (
                    f"Note: `{requested}` was not found in this provider's model listing. "
                    f"It may still work if your plan supports it."
                    f"{suggestion_text}"
                ),
            }

    # api_models is None — couldn't reach API.  Accept and persist,
    # but warn so typos don't silently break things.
    provider_label = _PROVIDER_LABELS.get(normalized, normalized)
    return {
        "accepted": True,
        "persist": True,
        "recognized": False,
        "message": (
            f"Could not reach the {provider_label} API to validate `{requested}`. "
            f"If the service isn't down, this model may not be valid."
        ),
    }
