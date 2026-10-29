"""``hermes providers list``: the inference-provider inventory without opening the picker.

The interactive ``hermes model`` picker, the dashboard and the TUI share one inventory substrate
(:mod:`hermes_cli.inventory`); this command prints the same rows for scripts, without a TTY or a
running gateway. ``--offline`` reads local config and bundled catalogs only
(:mod:`hermes_cli.inventory_offline`) and never contacts a provider.
"""

from __future__ import annotations

from typing import Any

from hermes_cli.cli_json import json_error, print_json_document


def cmd_providers(args) -> int:
    """Handle ``hermes providers list``."""
    payload = _inventory_payload(args)
    providers = _filter_providers(payload["providers"], getattr(args, "provider", None))
    if args.json:
        print_json_document({
            "offline": bool(args.offline),
            "current": {"provider": payload["provider"], "model": payload["model"]},
            "providers": providers,
        })
    else:
        _render_providers(providers)
    return 0


def _inventory_payload(args) -> dict[str, Any]:
    from hermes_cli.inventory import build_models_payload, load_picker_context

    context = load_picker_context()
    if args.offline:
        from hermes_cli.inventory_offline import build_offline_models_payload

        raw = build_offline_models_payload(context, include_unconfigured=bool(args.all))
    else:
        raw = build_models_payload(
            context, include_unconfigured=bool(args.all), picker_hints=True, canonical_order=True,
            probe_custom_providers=False, probe_current_custom_provider=True,
        )
    key_envs = _endpoint_key_envs(context.custom_providers)
    return {
        "providers": [_serialize_provider(row, key_envs) for row in raw["providers"]],
        "model": str(raw.get("model") or ""),
        "provider": str(raw.get("provider") or ""),
    }


def _endpoint_key_envs(custom_providers: list) -> dict[str, str]:
    """``key_env`` of each configured endpoint under every identity its picker row may carry: live
    endpoint rows do not repeat it, so it is read from the normalized provider view."""
    from hermes_cli.providers import custom_provider_aliases

    key_envs: dict[str, str] = {}
    for entry in custom_providers:
        if isinstance(entry, dict) and entry.get("key_env"):
            for alias in custom_provider_aliases(str(entry.get("name") or ""), str(entry.get("provider_key") or "")):
                key_envs.setdefault(alias, str(entry["key_env"]))
    return key_envs


def _serialize_provider(row: dict[str, Any], key_envs: dict[str, str]) -> dict[str, Any]:
    """Picker row -> the documented public row. Variable NAMES only, never values. Live built-in rows
    carry no ``auth_type``: the registry's is the provider's, as for unconfigured rows."""
    from hermes_cli.inventory import _provider_auth_hint

    slug = str(row.get("slug") or "")
    models = [str(model) for model in row.get("models") or []]
    key_env = str(row.get("key_env") or key_envs.get(slug.lower(), ""))
    return {
        "id": slug,
        "name": str(row.get("name") or slug),
        "auth_state": "configured" if row.get("authenticated") or row.get("is_user_defined") else "unconfigured",
        "auth_type": str(row.get("auth_type") or _provider_auth_hint(slug)[0]),
        "env_vars": _provider_env_vars(slug, key_env),
        "is_current": bool(row.get("is_current")),
        "is_user_defined": bool(row.get("is_user_defined")),
        "models": models,
        "model_count": int(row.get("total_models") or len(models)),
        "source": str(row.get("source") or "unknown"),
    }


def _provider_env_vars(slug: str, key_env: str) -> list[str]:
    from hermes_cli.auth import PROVIDER_REGISTRY

    config = PROVIDER_REGISTRY.get(slug)
    names = getattr(config, "api_key_env_vars", ()) if config else ()
    if not names and key_env:
        names = (key_env,)
    return [name for name in names if isinstance(name, str) and name]


def _filter_providers(providers: list[dict[str, Any]], requested: str | None) -> list[dict[str, Any]]:
    if not requested:
        return providers
    wanted = requested.strip().lower()
    filtered = [provider for provider in providers if provider["id"].lower() == wanted]
    if not filtered:
        json_error(f"unknown or unavailable provider: {requested} (pass --all to include unconfigured providers)")
    return filtered


def _render_providers(providers: list[dict[str, Any]]) -> None:
    if not providers:
        print("No providers found. Configure a provider or pass --all.")
        return
    width = max(len(provider["id"]) for provider in providers)
    print(f"{'PROVIDER':<{width}}  STATE         MODELS  CURRENT")
    for provider in providers:
        current = "yes" if provider["is_current"] else ""
        print(f"{provider['id']:<{width}}  {provider['auth_state']:<13} {provider['model_count']:>6}  {current}")
