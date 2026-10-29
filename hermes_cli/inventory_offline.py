"""Offline provider/model inventory: local config and bundled catalogs, never the network.

``hermes providers list --offline`` reads this instead of
:func:`hermes_cli.inventory.build_models_payload`, which may consult models.dev, remote model
catalogs and configured custom endpoints. Rows use the picker payload's provider-row keys so one
serializer handles both. Only environment variable NAMES are read for readiness, and the
credential pool is never opened.
"""

from __future__ import annotations

import os

from hermes_cli.inventory import ConfigContext


def _config_model_ids(*values: object) -> list[str]:
    """Model ids from the supported local config shapes (str, list, dict keyed by id, ``{id|name}``)."""
    ids: list[str] = []
    for value in values:
        candidates = list(value) if isinstance(value, (dict, list, tuple)) else [value]
        for item in candidates:
            if isinstance(item, dict):
                item = item.get("id") or item.get("name")
            if isinstance(item, str) and item.strip() and item.strip() not in ids:
                ids.append(item.strip())
    return ids


def _env_var_names(slug: str) -> tuple[str, ...]:
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.providers import get_provider

    config = PROVIDER_REGISTRY.get(slug) or get_provider(slug, allow_network=False)
    raw = getattr(config, "api_key_env_vars", ()) if config else ()
    return tuple(name for name in raw if isinstance(name, str) and name)


def _auth_type(slug: str) -> str:
    from hermes_cli.auth import PROVIDER_REGISTRY

    return getattr(PROVIDER_REGISTRY.get(slug), "auth_type", "api_key")


def _row(slug: str, name: str, models: list[str], *, is_current: bool, user_defined: bool, source: str,
         authenticated: bool = True, auth_type: str = "api_key", key_env: str = "") -> dict:
    return {"slug": slug, "name": name, "is_current": is_current, "is_user_defined": user_defined,
            "models": models, "total_models": len(models), "source": source,
            "authenticated": authenticated, "auth_type": auth_type, "key_env": key_env}


def _hidden_identities(ctx: ConfigContext) -> set[str]:
    """What the picker hides: ``model_catalog.excluded_providers`` and every ``providers.<key>`` entry
    with ``enabled: false`` (:func:`hermes_cli.config.is_provider_enabled`), under each key alias."""
    from hermes_cli.config import is_provider_enabled
    from hermes_cli.providers import custom_provider_aliases

    hidden = {str(slug).strip().lower() for slug in ctx.excluded_providers or [] if str(slug).strip()}
    for key, value in ctx.user_providers.items():
        if isinstance(value, dict) and not is_provider_enabled(value):
            hidden |= custom_provider_aliases(str(key))
    return hidden


def _canonical_rows(ctx: ConfigContext, current: str, include_unconfigured: bool, hidden: set[str]) -> list[dict]:
    from hermes_cli.models import _PROVIDER_MODELS, OPENROUTER_MODELS
    from hermes_cli.models_catalog_static import listed_canonical_providers

    configured = {str(slug).strip().lower() for slug, value in ctx.user_providers.items()
                  if isinstance(value, dict) and str(slug).strip()} | ({current} if current else set())
    rows = []
    for entry in listed_canonical_providers():
        slug = entry.slug.lower()
        env_vars = _env_var_names(slug)
        is_configured = slug in configured or any(os.environ.get(name, "").strip() for name in env_vars)
        if slug in hidden or not (include_unconfigured or is_configured):
            continue
        user_row = ctx.user_providers.get(entry.slug)
        models = _config_model_ids(*(user_row.get(k) for k in ("default_model", "model", "models"))) \
            if isinstance(user_row, dict) else []
        if slug == current and ctx.current_model:
            models.insert(0, ctx.current_model)
        static = [mid for mid, _ in OPENROUTER_MODELS] if slug == "openrouter" else list(_PROVIDER_MODELS.get(slug, ()))
        rows.append(_row(entry.slug, entry.label, list(dict.fromkeys([*models, *static])), is_current=slug == current,
                         user_defined=False, source="static", authenticated=is_configured,
                         auth_type=_auth_type(slug), key_env=env_vars[0] if env_vars else ""))
    return rows


def _norm_url(value: object) -> str:
    return str(value or "").strip().rstrip("/").lower()


def _endpoint_rows(ctx: ConfigContext, current: str, seen: set[str], hidden: set[str]) -> list[dict]:
    """Configured endpoints, declared models only, never probed.

    Reads ``ctx.custom_providers``: the normalized, enabled, deduplicated view
    (:func:`hermes_cli.config.get_compatible_custom_providers`) of both ``providers:`` and legacy
    ``custom_providers:``, so camelCase keys, ``api_key_env`` and API-mode aliases are interpreted
    once, by the runtime's own normalizer. A keyed entry keeps its config key as the slug (as the live
    picker does); a legacy entry gets :func:`hermes_cli.providers.custom_provider_slug`. Entries that
    share a slug (one legacy entry per model) merge into one row. A row is current under any
    :func:`hermes_cli.providers.custom_provider_aliases` identity, so a saved ``custom:<key>`` marks it.
    """
    from hermes_cli.providers import custom_provider_aliases, custom_provider_slug

    current_url = _norm_url(ctx.current_base_url)
    rows: dict[str, dict] = {}
    for entry in ctx.custom_providers:
        if not isinstance(entry, dict) or not str(entry.get("name") or "").strip():
            continue
        name, key = str(entry["name"]).strip(), str(entry.get("provider_key") or "").strip()
        slug, aliases = key or custom_provider_slug(name), custom_provider_aliases(name, key)
        row = rows.get(slug.lower())
        if row is None:
            if slug.lower() in seen or aliases & hidden:  # a canonical row has it, or the user hid it
                continue
            row = rows[slug.lower()] = _row(slug, name, [], is_current=False, user_defined=True,
                                            source="user-config", key_env=str(entry.get("key_env") or ""))
        url = _norm_url(entry.get("base_url"))
        row["is_current"] = row["is_current"] or current in aliases or (
            current == "custom" and bool(current_url) and url == current_url)
        row["models"] = list(dict.fromkeys([*row["models"], *_config_model_ids(entry.get("model"), entry.get("models"))]))
    for row in rows.values():
        if row["is_current"] and ctx.current_model and ctx.current_model not in row["models"]:
            row["models"].insert(0, ctx.current_model)
        row["total_models"] = len(row["models"])
    seen.update(rows)
    return list(rows.values())


def build_offline_models_payload(ctx: ConfigContext, *, include_unconfigured: bool = False) -> dict:
    """``{providers, model, provider}`` from config + bundled catalogs; no network, no credential pool."""
    current = str(ctx.current_provider or "").strip().lower()
    hidden = _hidden_identities(ctx)
    rows = _canonical_rows(ctx, current, include_unconfigured, hidden)
    seen = {row["slug"].lower() for row in rows}
    rows += _endpoint_rows(ctx, current, seen, hidden)
    # A saved provider no row represents (removed from config, or bare ``custom`` + base_url) —
    # unless the user hid it.
    if current and current not in seen | hidden and not any(row["is_current"] for row in rows):
        models = [ctx.current_model] if ctx.current_model else []
        rows.append(_row(ctx.current_provider, ctx.current_provider, models, is_current=True, user_defined=True,
                         source="model-config"))
    return {"providers": rows, "model": ctx.current_model, "provider": ctx.current_provider}
