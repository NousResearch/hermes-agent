"""Shared model-switching logic for CLI and gateway /model commands.

Both the CLI (cli.py) and gateway (gateway/run.py) /model handlers
share the same core pipeline:

  parse flags -> alias resolution -> provider resolution ->
  credential resolution -> normalize model name ->
  metadata lookup -> build result

This module ties together the foundation layers:

- ``agent.models_dev``            -- models.dev catalog, ModelInfo, ProviderInfo
- ``hermes_cli.providers``        -- canonical provider identity + overlays
- ``hermes_cli.model_normalize``  -- per-provider name formatting

Provider switching uses the ``--provider`` flag exclusively.
No colon-based ``provider:model`` syntax — colons are reserved for
OpenRouter variant suffixes (``:free``, ``:extended``, ``:fast``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional

from hermes_cli.providers import (
    ALIASES,
    LABELS,
    TRANSPORT_TO_API_MODE,
    determine_api_mode,
    get_label,
    get_provider,
    is_aggregator,
    normalize_provider,
    resolve_provider_full,
)
from hermes_cli.model_normalize import (
    detect_vendor,
    normalize_model_for_provider,
)
from agent.models_dev import (
    ModelCapabilities,
    ModelInfo,
    get_model_capabilities,
    get_model_info,
    list_provider_models,
    search_models_dev,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model aliases -- short names -> (vendor, family) with NO version numbers.
# Resolved dynamically against the live models.dev catalog.
# ---------------------------------------------------------------------------

class ModelIdentity(NamedTuple):
    """Vendor slug and family prefix used for catalog resolution."""
    vendor: str
    family: str


MODEL_ALIASES: dict[str, ModelIdentity] = {
    # Anthropic
    "sonnet":    ModelIdentity("anthropic", "claude-sonnet"),
    "opus":      ModelIdentity("anthropic", "claude-opus"),
    "haiku":     ModelIdentity("anthropic", "claude-haiku"),
    "claude":    ModelIdentity("anthropic", "claude"),

    # OpenAI
    "gpt5":      ModelIdentity("openai", "gpt-5"),
    "gpt":       ModelIdentity("openai", "gpt"),
    "codex":     ModelIdentity("openai", "codex"),
    "o3":        ModelIdentity("openai", "o3"),
    "o4":        ModelIdentity("openai", "o4"),

    # Google
    "gemini":    ModelIdentity("google", "gemini"),

    # DeepSeek
    "deepseek":  ModelIdentity("deepseek", "deepseek-chat"),

    # X.AI
    "grok":      ModelIdentity("x-ai", "grok"),

    # Meta
    "llama":     ModelIdentity("meta-llama", "llama"),

    # Qwen / Alibaba
    "qwen":      ModelIdentity("qwen", "qwen"),

    # MiniMax
    "minimax":   ModelIdentity("minimax", "minimax"),

    # Nvidia
    "nemotron":  ModelIdentity("nvidia", "nemotron"),

    # Moonshot / Kimi
    "kimi":      ModelIdentity("moonshotai", "kimi"),

    # Z.AI / GLM
    "glm":       ModelIdentity("z-ai", "glm"),

    # StepFun
    "step":      ModelIdentity("stepfun", "step"),

    # Xiaomi
    "mimo":      ModelIdentity("xiaomi", "mimo"),

    # Arcee
    "trinity":   ModelIdentity("arcee-ai", "trinity"),
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelSwitchResult:
    """Result of a model switch attempt."""

    success: bool
    new_model: str = ""
    target_provider: str = ""
    provider_changed: bool = False
    api_key: str = ""
    base_url: str = ""
    api_mode: str = ""
    error_message: str = ""
    warning_message: str = ""
    provider_label: str = ""
    resolved_via_alias: str = ""
    capabilities: Optional[ModelCapabilities] = None
    model_info: Optional[ModelInfo] = None
    is_global: bool = False


@dataclass
class CustomAutoResult:
    """Result of switching to bare 'custom' provider with auto-detect."""

    success: bool
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    error_message: str = ""


# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------

def parse_model_flags(raw_args: str) -> tuple[str, str, bool]:
    """Parse --provider and --global flags from /model command args.

    Returns (model_input, explicit_provider, is_global).

    Examples::

        "sonnet"                         -> ("sonnet", "", False)
        "sonnet --global"                -> ("sonnet", "", True)
        "sonnet --provider anthropic"    -> ("sonnet", "anthropic", False)
        "--provider my-ollama"           -> ("", "my-ollama", False)
        "sonnet --provider anthropic --global" -> ("sonnet", "anthropic", True)
    """
    is_global = False
    explicit_provider = ""

    # Extract --global
    if "--global" in raw_args:
        is_global = True
        raw_args = raw_args.replace("--global", "").strip()

    # Extract --provider <name>
    parts = raw_args.split()
    i = 0
    filtered: list[str] = []
    while i < len(parts):
        if parts[i] == "--provider" and i + 1 < len(parts):
            explicit_provider = parts[i + 1]
            i += 2
        else:
            filtered.append(parts[i])
            i += 1

    model_input = " ".join(filtered).strip()
    return (model_input, explicit_provider, is_global)


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------

def resolve_alias(
    raw_input: str,
    current_provider: str,
) -> Optional[tuple[str, str, str]]:
    """Resolve a short alias against the current provider's catalog.

    Looks up *raw_input* in :data:`MODEL_ALIASES`, then searches the
    current provider's models.dev catalog for the first model whose ID
    starts with ``vendor/family`` (or just ``family`` for non-aggregator
    providers).

    Returns:
        ``(provider, resolved_model_id, alias_name)`` if a match is
        found on the current provider, or ``None`` if the alias doesn't
        exist or no matching model is available.
    """
    key = raw_input.strip().lower()
    identity = MODEL_ALIASES.get(key)
    if identity is None:
        return None

    vendor, family = identity

    # Search the provider's catalog from models.dev
    catalog = list_provider_models(current_provider)
    if not catalog:
        return None

    # For aggregators, models are vendor/model-name format
    aggregator = is_aggregator(current_provider)

    for model_id in catalog:
        mid_lower = model_id.lower()
        if aggregator:
            # Match vendor/family prefix -- e.g. "anthropic/claude-sonnet"
            prefix = f"{vendor}/{family}".lower()
            if mid_lower.startswith(prefix):
                return (current_provider, model_id, key)
        else:
            # Non-aggregator: bare names -- e.g. "claude-sonnet-4-6"
            family_lower = family.lower()
            if mid_lower.startswith(family_lower):
                return (current_provider, model_id, key)

    return None


def _resolve_alias_fallback(
    raw_input: str,
    fallback_providers: tuple[str, ...] = ("openrouter", "nous"),
) -> Optional[tuple[str, str, str]]:
    """Try to resolve an alias on fallback providers."""
    for provider in fallback_providers:
        result = resolve_alias(raw_input, provider)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# Core model-switching pipeline
# ---------------------------------------------------------------------------

def switch_model(
    raw_input: str,
    current_provider: str,
    current_model: str,
    current_base_url: str = "",
    current_api_key: str = "",
    is_global: bool = False,
    explicit_provider: str = "",
    user_providers: dict = None,
) -> ModelSwitchResult:
    """Core model-switching pipeline shared between CLI and gateway.

    Resolution chain:

      If --provider given:
        a. Resolve provider via resolve_provider_full()
        b. Resolve credentials
        c. If model given, resolve alias on target provider or use as-is
        d. If no model, auto-detect from endpoint

      If no --provider:
        a. Try alias resolution on current provider
        b. If alias exists but not on current provider -> fallback
        c. On aggregator, try vendor/model slug conversion
        d. Aggregator catalog search
        e. detect_provider_for_model() as last resort
        f. Resolve credentials
        g. Normalize model name for target provider

      Finally:
        h. Get full model metadata from models.dev
        i. Build result

    Args:
        raw_input: The model name (after flag parsing).
        current_provider: The currently active provider.
        current_model: The currently active model name.
        current_base_url: The currently active base URL.
        current_api_key: The currently active API key.
        is_global: Whether to persist the switch.
        explicit_provider: From --provider flag (empty = no explicit provider).
        user_providers: The ``providers:`` dict from config.yaml (for user endpoints).

    Returns:
        ModelSwitchResult with all information the caller needs.
    """
    from hermes_cli.models import (
        detect_provider_for_model,
        validate_requested_model,
        opencode_model_api_mode,
    )
    from hermes_cli.runtime_provider import resolve_runtime_provider

    resolved_alias = ""
    new_model = raw_input.strip()
    target_provider = current_provider

    # =================================================================
    # PATH A: Explicit --provider given
    # =================================================================
    if explicit_provider:
        # Resolve the provider
        pdef = resolve_provider_full(explicit_provider, user_providers)
        if pdef is None:
            return ModelSwitchResult(
                success=False,
                is_global=is_global,
                error_message=(
                    f"Unknown provider '{explicit_provider}'. "
                    f"Check 'hermes model' for available providers, or define it "
                    f"in config.yaml under 'providers:'."
                ),
            )

        target_provider = pdef.id

        # If no model specified, try auto-detect from endpoint
        if not new_model:
            if pdef.base_url:
                from hermes_cli.runtime_provider import _auto_detect_local_model
                detected = _auto_detect_local_model(pdef.base_url)
                if detected:
                    new_model = detected
                else:
                    return ModelSwitchResult(
                        success=False,
                        target_provider=target_provider,
                        provider_label=pdef.name,
                        is_global=is_global,
                        error_message=(
                            f"No model detected on {pdef.name} ({pdef.base_url}). "
                            f"Specify the model explicitly: /model <model-name> --provider {explicit_provider}"
                        ),
                    )
            else:
                return ModelSwitchResult(
                    success=False,
                    target_provider=target_provider,
                    provider_label=pdef.name,
                    is_global=is_global,
                    error_message=(
                        f"Provider '{pdef.name}' has no base URL configured. "
                        f"Specify a model: /model <model-name> --provider {explicit_provider}"
                    ),
                )

        # Resolve alias on the TARGET provider
        alias_result = resolve_alias(new_model, target_provider)
        if alias_result is not None:
            _, new_model, resolved_alias = alias_result

    # =================================================================
    # PATH B: No explicit provider — resolve from model input
    # =================================================================
    else:
        # --- Step a: Try alias resolution on current provider ---
        alias_result = resolve_alias(raw_input, current_provider)

        if alias_result is not None:
            target_provider, new_model, resolved_alias = alias_result
            logger.debug(
                "Alias '%s' resolved to %s on %s",
                resolved_alias, new_model, target_provider,
            )
        else:
            # --- Step b: Alias exists but not on current provider -> fallback ---
            key = raw_input.strip().lower()
            if key in MODEL_ALIASES:
                fallback_result = _resolve_alias_fallback(raw_input)
                if fallback_result is not None:
                    target_provider, new_model, resolved_alias = fallback_result
                    logger.debug(
                        "Alias '%s' resolved via fallback to %s on %s",
                        resolved_alias, new_model, target_provider,
                    )
                else:
                    identity = MODEL_ALIASES[key]
                    return ModelSwitchResult(
                        success=False,
                        is_global=is_global,
                        error_message=(
                            f"Alias '{key}' maps to {identity.vendor}/{identity.family} "
                            f"but no matching model was found in any provider catalog. "
                            f"Try specifying the full model name."
                        ),
                    )
            else:
                # --- Step c: On aggregator, convert vendor:model to vendor/model ---
                colon_pos = raw_input.find(":")
                if colon_pos > 0 and is_aggregator(current_provider):
                    left = raw_input[:colon_pos].strip().lower()
                    right = raw_input[colon_pos + 1:].strip()
                    if left and right:
                        # Colons become slashes for aggregator slugs
                        new_model = f"{left}/{right}"
                        logger.debug(
                            "Converted vendor:model '%s' to aggregator slug '%s'",
                            raw_input, new_model,
                        )

        # --- Step d: Aggregator catalog search ---
        if is_aggregator(target_provider) and not resolved_alias:
            catalog = list_provider_models(target_provider)
            if catalog:
                new_model_lower = new_model.lower()
                for mid in catalog:
                    if mid.lower() == new_model_lower:
                        new_model = mid
                        break
                else:
                    for mid in catalog:
                        if "/" in mid:
                            _, bare = mid.split("/", 1)
                            if bare.lower() == new_model_lower:
                                new_model = mid
                                break

        # --- Step e: detect_provider_for_model() as last resort ---
        _base = current_base_url or ""
        is_custom = current_provider in ("custom", "local") or (
            "localhost" in _base or "127.0.0.1" in _base
        )

        if (
            target_provider == current_provider
            and not is_custom
            and not resolved_alias
        ):
            detected = detect_provider_for_model(new_model, current_provider)
            if detected:
                target_provider, new_model = detected

    # =================================================================
    # COMMON PATH: Resolve credentials, normalize, get metadata
    # =================================================================

    provider_changed = target_provider != current_provider
    provider_label = get_label(target_provider)

    # --- Resolve credentials ---
    api_key = current_api_key
    base_url = current_base_url
    api_mode = ""

    if provider_changed or explicit_provider:
        try:
            runtime = resolve_runtime_provider(requested=target_provider)
            api_key = runtime.get("api_key", "")
            base_url = runtime.get("base_url", "")
            api_mode = runtime.get("api_mode", "")
        except Exception as e:
            return ModelSwitchResult(
                success=False,
                target_provider=target_provider,
                provider_label=provider_label,
                is_global=is_global,
                error_message=(
                    f"Could not resolve credentials for provider "
                    f"'{provider_label}': {e}"
                ),
            )
    else:
        try:
            runtime = resolve_runtime_provider(requested=current_provider)
            api_key = runtime.get("api_key", "")
            base_url = runtime.get("base_url", "")
            api_mode = runtime.get("api_mode", "")
        except Exception:
            pass

    # --- Normalize model name for target provider ---
    new_model = normalize_model_for_provider(new_model, target_provider)

    # --- Validate ---
    try:
        validation = validate_requested_model(
            new_model,
            target_provider,
            api_key=api_key,
            base_url=base_url,
        )
    except Exception:
        validation = {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": None,
        }

    if not validation.get("accepted"):
        msg = validation.get("message", "Invalid model")
        return ModelSwitchResult(
            success=False,
            new_model=new_model,
            target_provider=target_provider,
            provider_label=provider_label,
            is_global=is_global,
            error_message=msg,
        )

    # --- OpenCode api_mode override ---
    if target_provider in {"opencode-zen", "opencode-go", "opencode", "opencode-go"}:
        api_mode = opencode_model_api_mode(target_provider, new_model)

    # --- Determine api_mode if not already set ---
    if not api_mode:
        api_mode = determine_api_mode(target_provider, base_url)

    # --- Get capabilities (legacy) ---
    capabilities = get_model_capabilities(target_provider, new_model)

    # --- Get full model info from models.dev ---
    model_info = get_model_info(target_provider, new_model)

    # --- Build result ---
    return ModelSwitchResult(
        success=True,
        new_model=new_model,
        target_provider=target_provider,
        provider_changed=provider_changed,
        api_key=api_key,
        base_url=base_url,
        api_mode=api_mode,
        warning_message=validation.get("message") or "",
        provider_label=provider_label,
        resolved_via_alias=resolved_alias,
        capabilities=capabilities,
        model_info=model_info,
        is_global=is_global,
    )


# ---------------------------------------------------------------------------
# Authenticated providers listing (for /model no-args display)
# ---------------------------------------------------------------------------

def list_authenticated_providers(
    current_provider: str = "",
    user_providers: dict = None,
    max_models: int = 8,
) -> List[dict]:
    """Detect which providers have credentials and list their curated models.

    Uses the curated model lists from hermes_cli/models.py (OPENROUTER_MODELS,
    _PROVIDER_MODELS) — NOT the full models.dev catalog.  These are hand-picked
    agentic models that work well as agent backends.

    Returns a list of dicts, each with:
      - slug: str — the --provider value to use
      - name: str — display name
      - is_current: bool
      - is_user_defined: bool
      - models: list[str] — curated model IDs (up to max_models)
      - total_models: int — total curated count
      - source: str — "built-in", "models.dev", "user-config"

    Only includes providers that have API keys set or are user-defined endpoints.
    """
    import os
    from agent.models_dev import (
        PROVIDER_TO_MODELS_DEV,
        fetch_models_dev,
        get_provider_info as _mdev_pinfo,
    )
    from hermes_cli.models import OPENROUTER_MODELS, _PROVIDER_MODELS

    results: List[dict] = []
    seen_slugs: set = set()

    data = fetch_models_dev()

    # Build curated model lists keyed by hermes provider ID
    curated: dict[str, list[str]] = dict(_PROVIDER_MODELS)
    curated["openrouter"] = [mid for mid, _ in OPENROUTER_MODELS]
    # "nous" shares OpenRouter's curated list if not separately defined
    if "nous" not in curated:
        curated["nous"] = curated["openrouter"]

    # --- 1. Check Hermes-mapped providers ---
    for hermes_id, mdev_id in PROVIDER_TO_MODELS_DEV.items():
        pdata = data.get(mdev_id)
        if not isinstance(pdata, dict):
            continue

        env_vars = pdata.get("env", [])
        if not isinstance(env_vars, list):
            continue

        # Check if any env var is set
        has_creds = any(os.environ.get(ev) for ev in env_vars)
        if not has_creds:
            continue

        # Use curated list, falling back to models.dev if no curated list
        model_ids = curated.get(hermes_id, [])
        total = len(model_ids)
        top = model_ids[:max_models]

        slug = hermes_id
        pinfo = _mdev_pinfo(mdev_id)
        display_name = pinfo.name if pinfo else mdev_id

        results.append({
            "slug": slug,
            "name": display_name,
            "is_current": slug == current_provider or mdev_id == current_provider,
            "is_user_defined": False,
            "models": top,
            "total_models": total,
            "source": "built-in",
        })
        seen_slugs.add(slug)

    # --- 2. Check Hermes-only providers (nous, openai-codex, copilot) ---
    from hermes_cli.providers import HERMES_OVERLAYS
    for pid, overlay in HERMES_OVERLAYS.items():
        if pid in seen_slugs:
            continue
        # Check if credentials exist
        has_creds = False
        if overlay.extra_env_vars:
            has_creds = any(os.environ.get(ev) for ev in overlay.extra_env_vars)
        if overlay.auth_type in ("oauth_device_code", "oauth_external", "external_process"):
            # These use auth stores, not env vars — check for auth.json entries
            try:
                from hermes_cli.auth import _read_auth_store
                store = _read_auth_store()
                if store and pid in store:
                    has_creds = True
            except Exception:
                pass
        if not has_creds:
            continue

        # Use curated list
        model_ids = curated.get(pid, [])
        total = len(model_ids)
        top = model_ids[:max_models]

        results.append({
            "slug": pid,
            "name": get_label(pid),
            "is_current": pid == current_provider,
            "is_user_defined": False,
            "models": top,
            "total_models": total,
            "source": "hermes",
        })
        seen_slugs.add(pid)

    # --- 3. User-defined endpoints from config ---
    if user_providers and isinstance(user_providers, dict):
        for ep_name, ep_cfg in user_providers.items():
            if not isinstance(ep_cfg, dict):
                continue
            display_name = ep_cfg.get("name", "") or ep_name
            api_url = ep_cfg.get("api", "") or ep_cfg.get("url", "") or ""
            default_model = ep_cfg.get("default_model", "")

            models_list = []
            if default_model:
                models_list.append(default_model)

            # Try to probe /v1/models if URL is set (but don't block on it)
            # For now just show what we know from config
            results.append({
                "slug": ep_name,
                "name": display_name,
                "is_current": ep_name == current_provider,
                "is_user_defined": True,
                "models": models_list,
                "total_models": len(models_list) if models_list else 0,
                "source": "user-config",
                "api_url": api_url,
            })

    # Sort: current provider first, then by model count descending
    results.sort(key=lambda r: (not r["is_current"], -r["total_models"]))

    return results


# ---------------------------------------------------------------------------
# Fuzzy suggestions
# ---------------------------------------------------------------------------

def suggest_models(raw_input: str, limit: int = 3) -> List[str]:
    """Return fuzzy model suggestions for a (possibly misspelled) input."""
    query = raw_input.strip()
    if not query:
        return []

    results = search_models_dev(query, limit=limit)
    suggestions: list[str] = []
    for r in results:
        mid = r.get("model_id", "")
        if mid:
            suggestions.append(mid)

    return suggestions[:limit]


# ---------------------------------------------------------------------------
# Custom provider switch
# ---------------------------------------------------------------------------

def switch_to_custom_provider() -> CustomAutoResult:
    """Handle bare '/model --provider custom' — resolve endpoint and auto-detect model."""
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider,
        _auto_detect_local_model,
    )

    try:
        runtime = resolve_runtime_provider(requested="custom")
    except Exception as e:
        return CustomAutoResult(
            success=False,
            error_message=f"Could not resolve custom endpoint: {e}",
        )

    cust_base = runtime.get("base_url", "")
    cust_key = runtime.get("api_key", "")

    if not cust_base or "openrouter.ai" in cust_base:
        return CustomAutoResult(
            success=False,
            error_message=(
                "No custom endpoint configured. "
                "Set model.base_url in config.yaml, or set OPENAI_BASE_URL "
                "in .env, or run: hermes setup -> Custom OpenAI-compatible endpoint"
            ),
        )

    detected_model = _auto_detect_local_model(cust_base)
    if not detected_model:
        return CustomAutoResult(
            success=False,
            base_url=cust_base,
            api_key=cust_key,
            error_message=(
                f"Custom endpoint at {cust_base} is reachable but no single "
                f"model was auto-detected. Specify the model explicitly: "
                f"/model <model-name> --provider custom"
            ),
        )

    return CustomAutoResult(
        success=True,
        model=detected_model,
        base_url=cust_base,
        api_key=cust_key,
    )
