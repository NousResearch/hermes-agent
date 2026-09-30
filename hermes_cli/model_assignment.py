"""Main-slot model assignment: normalize a (provider, model) pick, validate it through
``switch_model``, and apply it to a ``model:`` config block.

Shared by the dashboard (``POST /api/model/set``, the Config page, custom endpoints) and profile
provisioning, so nothing here knows about HTTP: a rejected pick raises
:class:`ModelAssignmentRejected` and each surface maps it (the dashboard to a 400).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Optional

from hermes_cli.model_switch import ModelSwitchResult
from hermes_cli.config import custom_endpoint_key_env, get_env_value, save_env_value

# The dashboard's logger, as before the move: hermes_logging routes it to gui.log.
logger = logging.getLogger("hermes_cli.web_server")


class ModelAssignmentRejected(ValueError):
    """``switch_model`` refused the pick; the message says why."""


def normalize_main_model_assignment(provider: str, model: str) -> tuple[str, str]:
    """Normalize a main-slot (provider, model) pair before persisting.

    The per-card "Use as → Main model" menu can send the model's VENDOR prefix as the
    provider (analytics rows with no ``billing_provider``), producing e.g.
    ``provider: anthropic`` + ``default: anthropic/claude-opus-4.6`` — an aggregator slug on
    the native provider, which 400s. Two repairs at this single chokepoint:

    1. Vendor-name → Hermes-provider: when the provider is not a known provider/alias but the
       model is a vendor-prefixed slug, keep the user's CURRENT aggregator if on one, else
       openrouter. User-declared ``providers:``/``custom_providers:`` entries resolve first,
       and durable named-custom slugs (``custom`` / ``custom:<name>``) are excluded —
       ``_KNOWN_PROVIDER_NAMES`` lists only the bare ``custom`` bucket, so without this a
       LiteLLM proxy serving ``ollama/glm-5.2`` would be silently reassigned to openrouter.
       Matching only that syntax (not ``startswith("custom")``) avoids swallowing
       unconfigured vendors like ``customproxy``.
    2. Model-format normalization for the resolved provider via
       ``normalize_model_for_provider`` (custom/user providers keep the model verbatim).
    """
    from hermes_cli.config import get_compatible_custom_providers, load_config
    from hermes_cli.models import _AGGREGATOR_PROVIDERS, _KNOWN_PROVIDER_NAMES, normalize_provider
    from hermes_cli.model_normalize import normalize_model_for_provider
    from hermes_cli.providers import resolve_custom_provider, resolve_user_provider

    prov_in = (provider or "").strip()
    model_in = (model or "").strip()
    canonical = normalize_provider(prov_in)

    try:
        cfg = load_config()
    except Exception:  # an unreadable config normalizes against no declared providers
        logger.debug("config unreadable while normalizing %s/%s", prov_in, model_in, exc_info=True)
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    user_providers = cfg.get("providers")
    declared = resolve_user_provider(
        prov_in, user_providers if isinstance(user_providers, dict) else {}
    ) or resolve_custom_provider(prov_in, get_compatible_custom_providers(cfg))
    if declared is not None:
        return declared.id, model_in

    is_custom_provider_slug = canonical == "custom" or canonical.startswith("custom:")
    if canonical not in _KNOWN_PROVIDER_NAMES and not is_custom_provider_slug and "/" in model_in:
        cur_cfg = cfg.get("model", {})
        cur_provider = str(cur_cfg.get("provider", "") or "").strip().lower() if isinstance(cur_cfg, dict) else ""
        if cur_provider and normalize_provider(cur_provider) in _AGGREGATOR_PROVIDERS:
            canonical = normalize_provider(cur_provider)
            prov_in = cur_provider
        else:
            from hermes_cli.models_detect import provider_has_credentials

            # Only guess OpenRouter when the user actually holds a key for it; otherwise keep the
            # pair as sent rather than persisting a provider they never selected.
            if provider_has_credentials("openrouter"):
                canonical = prov_in = "openrouter"

    if canonical in _KNOWN_PROVIDER_NAMES and not canonical.startswith("custom"):
        try:
            model_in = normalize_model_for_provider(model_in, canonical) or model_in
        except Exception:
            logger.debug("model normalization failed for %s/%s", prov_in, model_in, exc_info=True)

    return prov_in, model_in


BARE_CUSTOM_SELECTORS = frozenset({"custom", "local"})


def _same_endpoint(a: Any, b: Any) -> bool:
    """Trailing-slash and case insensitive URL equality (``_save_custom_provider`` dedups the same way)."""
    return str(a or "").strip().rstrip("/").lower() == str(b or "").strip().rstrip("/").lower()


def entry_points_elsewhere(entry: Any, base_url: str) -> bool:
    """A configured provider block that names a DIFFERENT host than the submitted endpoint. Its
    credentials belong to that host; a block for the same host (e.g. its ``key_env``) still applies."""
    return isinstance(entry, dict) and not _same_endpoint(entry.get("base_url"), base_url)


def is_explicit_custom_endpoint(provider: str, base_url: str) -> bool:
    """A bare ``custom`` / ``local`` pick that carries its own endpoint URL defines a route; it is
    never a reference to an already-configured provider."""
    return provider.strip().lower() in BARE_CUSTOM_SELECTORS and bool((base_url or "").strip())


def validated_main_model_selection(
    cfg: dict, provider: str, model: str, base_url: str = "", api_key: str = "",
    explicit_endpoint: Optional[bool] = None,
) -> ModelSwitchResult:
    """Route a main-slot pick through ``switch_model`` (catalog/alias/credential
    validation) seeded with the configured route, exactly like a ``/model <model> --provider
    <provider> --global``. A bare ``custom`` target carries the submitted endpoint as the current
    one, which is how ``switch_model`` binds a custom base_url/key. A rejection raises
    :class:`ModelAssignmentRejected`."""
    from hermes_cli.config import get_compatible_custom_providers
    from hermes_cli.model_switch import switch_model

    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    provider_norm = provider.strip().lower()
    is_bare_custom = provider_norm in BARE_CUSTOM_SELECTORS
    is_custom_endpoint = is_bare_custom or provider_norm.startswith("custom:")
    # An explicit endpoint must not resolve through the configured providers: the legacy
    # bare-``custom`` recovery in ``resolve_custom_provider`` would bind the FIRST named entry, and
    # a ``providers.custom`` / ``providers.local`` block (same key as the bare selector) would win
    # the credential step and validate ITS host with ITS key. The dashboard's ``_prepare_main_assignment``
    # passes whether the URL was SUBMITTED; a URL it filled from that block is a selection, not a route.
    if explicit_endpoint is None:
        explicit_endpoint = is_explicit_custom_endpoint(provider, base_url)
    user_providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    if explicit_endpoint:
        user_providers = {k: v for k, v in user_providers.items()
                          if not (str(k).strip().lower() in BARE_CUSTOM_SELECTORS
                                  and entry_points_elsewhere(v, base_url))}
    # The submitted endpoint is seeded as a current bare-``custom`` route so the credential step
    # keeps it: arriving from another provider it re-resolves ``custom`` from config and would
    # validate the saved ``model.base_url`` (and its key) instead of the endpoint being saved.
    switch_provider = "custom" if provider_norm.startswith("custom:") else provider
    result = switch_model(
        raw_input=model, explicit_provider=switch_provider, is_global=True,
        current_provider="custom" if explicit_endpoint else str(model_cfg.get("provider") or ""),
        current_model=str(model_cfg.get("default") or ""),
        current_base_url=base_url if is_custom_endpoint else str(model_cfg.get("base_url") or ""),
        current_api_key=api_key if is_custom_endpoint else "",
        user_providers=user_providers,
        custom_providers=[] if explicit_endpoint else get_compatible_custom_providers(cfg))
    if not result.success:
        raise ModelAssignmentRejected(result.error_message or "model switch rejected")
    if is_custom_endpoint and base_url.strip():
        # The submitted endpoint IS the route this pick asked for; the credential step may have
        # re-resolved the bare target onto an env/config endpoint (CUSTOM_BASE_URL, a stale
        # model.base_url, the OPENROUTER_BASE_URL mirror). Restore the submitted endpoint AND the
        # wire protocol it mandates: ``model.base_url`` and ``model.api_mode`` are persisted
        # together, so a mode derived from the displaced host would route the submitted endpoint
        # over the wrong wire.
        from hermes_cli.providers import determine_api_mode
        url = base_url.strip()
        result = replace(result, base_url=url,
                         api_mode=determine_api_mode(result.target_provider, url))
    return result


def apply_main_model_assignment(
    model_cfg: Any, result: ModelSwitchResult, api_key: str = "", key_env: str = "",
    provider: str = "",
) -> dict:
    """Apply a main-slot selection, preserving credential bindings and remote masks.

    Submitted custom-endpoint secrets are persisted before this call and carried as ``key_env``.
    Endpoint changes clear old references; same-endpoint model selections preserve them.
    Returns a new dict.
    """
    from hermes_cli.config_backend import supports_file_tooling
    from hermes_cli.model_switch import apply_model_selection, remote_model_selection_guard

    before = model_cfg
    previous = model_cfg if isinstance(model_cfg, dict) else {}
    old_provider = str(previous.get("provider") or "").strip().lower()
    old_url = str(previous.get("base_url") or "").strip().rstrip("/")
    new_url = str(result.base_url or "").strip().rstrip("/")
    endpoint_changed = bool(new_url and old_url and new_url != old_url)
    if provider and provider != result.target_provider:
        result = replace(result, target_provider=provider)
    # Remote Config: the caller saves the effective block as a whole document, so a cleared
    # inherited key must be masked (null), and a locked route key refuses the whole assignment.
    model_cfg = apply_model_selection(model_cfg, result, mask_cleared=not supports_file_tooling())
    if key_env.strip():
        model_cfg["key_env"] = key_env.strip()
        drop_model_key(model_cfg, "api_key_env")
        drop_model_key(model_cfg, "api_key")
        drop_model_key(model_cfg, "api")
    elif api_key.strip():
        model_cfg["api_key"] = api_key.strip()
        drop_model_key(model_cfg, "api")
        drop_model_key(model_cfg, "key_env")
        drop_model_key(model_cfg, "api_key_env")
    elif old_provider != result.target_provider.strip().lower() or endpoint_changed:
        for key in ("api_key", "api", "key_env", "api_key_env"):
            drop_model_key(model_cfg, key)
    remote_model_selection_guard(before, model_cfg)
    return model_cfg


def drop_model_key(model_cfg: dict, key: str) -> None:
    """Remove *key* from a ``model`` block an assignment is building. Under Remote Config the block
    is the EFFECTIVE one, so a key it holds may be inherited from an upper level that a profile
    removal cannot reach (contract §5.4); store null ("no value") instead, the same mask
    :func:`hermes_cli.model_switch.apply_model_selection` applies. A key the block lacks stays
    absent. File backend: a plain removal."""
    from hermes_cli.config_backend import supports_file_tooling
    if key not in model_cfg:
        return
    if supports_file_tooling():
        model_cfg.pop(key, None)
    else:
        model_cfg[key] = None


def persist_custom_endpoint_secret(provider: str, base_url: str, api_key: str) -> str:
    """Write a bare/named custom or local API key to ``.env``; return its ``key_env``.

    Empty string means this assignment has no custom-endpoint secret to stash
    (other providers, missing URL, or empty key). Raises if the ``.env`` write
    cannot be verified — callers must not persist a ``key_env`` pointer to a
    secret that is not actually on disk.
    """
    normalized_provider = provider.strip().lower()
    if normalized_provider not in {"custom", "local"} and not normalized_provider.startswith("custom:"):
        return ""
    if not base_url or not api_key.strip():
        return ""
    secret = api_key.strip()
    env_var = custom_endpoint_key_env(base_url)
    save_env_value(env_var, secret)
    if (get_env_value(env_var) or "").strip() != secret:
        raise RuntimeError(f"failed to persist {env_var} to .env")
    return env_var

