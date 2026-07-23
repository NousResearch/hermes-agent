"""Non-interactive provider and model inventory commands.

The interactive ``hermes model`` picker, dashboard, and TUI already share the
same inventory substrate.  This module exposes that data to scripts without
requiring a TTY or a running gateway.
"""

from __future__ import annotations

import json
import sys
from typing import Any


def build_inventory_parsers(subparsers, *, cmd_models, cmd_providers) -> None:
    """Register the scriptable ``models`` and ``providers`` command groups."""
    models_parser = subparsers.add_parser(
        "models",
        help="List available models without opening the interactive picker",
        description="List configured provider models in a scriptable format.",
    )
    models_subparsers = models_parser.add_subparsers(dest="models_action")
    models_subparsers.required = True
    models_list = models_subparsers.add_parser(
        "list", aliases=["ls"], help="List models grouped by provider"
    )
    _add_inventory_filters(models_list)

    models_status = models_subparsers.add_parser(
        "status",
        help="Show configured-provider and model-catalog status",
    )
    _add_inventory_filters(models_status, include_offline=False)
    status_mode = models_status.add_mutually_exclusive_group()
    status_mode.add_argument(
        "--offline",
        "--no-live",
        dest="offline",
        action="store_true",
        help="Use local configuration and bundled catalogs only; never contact a provider",
    )
    status_mode.add_argument(
        "--probe",
        action="store_true",
        help="Refresh provider model catalogs before reporting their status",
    )
    models_list.set_defaults(func=cmd_models)
    models_status.set_defaults(func=cmd_models)

    providers_parser = subparsers.add_parser(
        "providers",
        help="List configured inference providers without opening the picker",
        description="List inference providers in a scriptable format.",
    )
    providers_subparsers = providers_parser.add_subparsers(dest="providers_action")
    providers_subparsers.required = True
    providers_list = providers_subparsers.add_parser(
        "list", aliases=["ls"], help="List inference providers"
    )
    _add_inventory_filters(providers_list)
    providers_list.set_defaults(func=cmd_providers)


def _add_inventory_filters(parser, *, include_offline: bool = True) -> None:
    parser.add_argument(
        "--provider",
        metavar="NAME",
        help="Limit output to one provider slug",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include supported but unconfigured providers",
    )
    if include_offline:
        parser.add_argument(
            "--offline",
            "--no-live",
            dest="offline",
            action="store_true",
            help="Use local configuration and bundled catalogs only; never contact a provider",
        )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a stable machine-readable JSON document",
    )


def cmd_models(args) -> None:
    """Handle ``hermes models list`` and ``hermes models status``."""
    action = getattr(args, "models_action", "list")
    payload = _inventory_payload(
        args, probe=action == "status" and getattr(args, "probe", False)
    )
    providers = _filter_providers(payload["providers"], getattr(args, "provider", None))

    if action == "status":
        document = {
            "schema_version": 1,
            "offline": bool(args.offline),
            "probe_requested": bool(getattr(args, "probe", False)),
            "providers": providers,
        }
        _print_document(document, as_json=args.json, render=_render_provider_status)
        return

    models = [
        {
            "id": model_id,
            "provider": provider["id"],
            "provider_name": provider["name"],
            "is_current": bool(provider["is_current"] and model_id == payload["model"]),
        }
        for provider in providers
        for model_id in provider["models"]
    ]
    document = {
        "schema_version": 1,
        "offline": bool(args.offline),
        "models": models,
    }
    _print_document(document, as_json=args.json, render=_render_models)


def cmd_providers(args) -> None:
    """Handle ``hermes providers list``."""
    payload = _inventory_payload(args)
    document = {
        "schema_version": 1,
        "offline": bool(args.offline),
        "providers": _filter_providers(payload["providers"], getattr(args, "provider", None)),
    }
    _print_document(document, as_json=args.json, render=_render_provider_status)


def _inventory_payload(args, *, probe: bool = False) -> dict[str, Any]:
    from hermes_cli.inventory import (
        build_models_payload,
        build_offline_models_payload,
        load_picker_context,
    )

    context = load_picker_context()
    if args.offline:
        raw = build_offline_models_payload(
            context, include_unconfigured=bool(args.all)
        )
    else:
        raw = build_models_payload(
            context,
            include_unconfigured=bool(args.all),
            picker_hints=True,
            canonical_order=True,
            refresh=probe,
            probe_custom_providers=probe,
            probe_current_custom_provider=not probe,
        )
    return {
        "providers": [_serialize_provider(row) for row in raw["providers"]],
        "model": str(raw.get("model") or ""),
    }


def _serialize_provider(row: dict[str, Any]) -> dict[str, Any]:
    """Convert picker rows into the stable public CLI JSON contract."""
    slug = str(row.get("slug") or "")
    env_vars = _provider_env_vars(slug, row)
    is_configured = bool(row.get("authenticated") or row.get("is_user_defined"))
    return {
        "id": slug,
        "name": str(row.get("name") or slug),
        "auth_state": "configured" if is_configured else "unconfigured",
        "auth_type": str(row.get("auth_type") or "api_key"),
        "env_vars": env_vars,
        "is_current": bool(row.get("is_current")),
        "models": [str(model) for model in row.get("models") or []],
        "model_count": int(row.get("total_models") or 0),
        "source": str(row.get("source") or "unknown"),
    }


def _provider_env_vars(slug: str, row: dict[str, Any]) -> list[str]:
    """Return variable *names* only; inventory commands never expose values."""
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.providers import get_provider

        config = PROVIDER_REGISTRY.get(slug) or get_provider(slug)
        env_vars = getattr(config, "api_key_env_vars", ()) if config else ()
        return [name for name in env_vars if isinstance(name, str) and name]
    except Exception:
        key_env = row.get("key_env")
        return [key_env] if isinstance(key_env, str) and key_env else []


def _filter_providers(providers: list[dict[str, Any]], requested: str | None) -> list[dict[str, Any]]:
    if not requested:
        return providers
    normalized = requested.strip().lower()
    filtered = [provider for provider in providers if provider["id"].lower() == normalized]
    if filtered:
        return filtered
    print(f"Unknown or unavailable provider: {requested}", file=sys.stderr)
    raise SystemExit(2)


def _print_document(document: dict[str, Any], *, as_json: bool, render) -> None:
    if as_json:
        print(json.dumps(document, indent=2, sort_keys=True))
        return
    render(document)


def _render_models(document: dict[str, Any]) -> None:
    models = document["models"]
    if not models:
        print("No models found. Configure a provider or pass --all.")
        return
    for model in models:
        current = " *" if model["is_current"] else ""
        print(f"{model['provider']}: {model['id']}{current}")


def _render_provider_status(document: dict[str, Any]) -> None:
    providers = document["providers"]
    if not providers:
        print("No providers found. Configure a provider or pass --all.")
        return
    name_width = max(len(provider["id"]) for provider in providers)
    print(f"{'PROVIDER':<{name_width}}  STATE         MODELS  CURRENT")
    for provider in providers:
        current = "yes" if provider["is_current"] else ""
        print(
            f"{provider['id']:<{name_width}}  {provider['auth_state']:<13} "
            f"{provider['model_count']:>6}  {current}"
        )
