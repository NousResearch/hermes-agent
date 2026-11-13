"""A profile's configured default model named as an alias (``model.default: myalias``).

``model_aliases:`` / ``model.aliases`` entries carry their own provider, endpoint and key. CLI startup,
``-m`` and ``/model`` expand them through ``resolve_startup_model_route``; every surface that runs the
CONFIG default instead (messaging gateway, cron, TUI agent factory, API server) must do the same, or
the literal alias goes to ``model.provider``'s endpoint and the provider answers HTTP 400 (#154786).
Only names that ARE configured aliases expand here: a plain id or a ``provider:model`` default keeps
the route the default provider resolution already gives it.
"""
from __future__ import annotations

from typing import Any, Optional


def _model_section(config: Any) -> dict:
    section = config.get("model") if isinstance(config, dict) else None
    return section if isinstance(section, dict) else {}


def configured_default_model(config: dict) -> str:
    """``model.default`` (or ``model.model`` / the ``model: <id>`` shorthand), stripped; "" when unset."""
    model_cfg = config.get("model") if isinstance(config, dict) else None
    if isinstance(model_cfg, dict):
        return str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
    return model_cfg.strip() if isinstance(model_cfg, str) else ""


def is_configured_alias(name: str, config: dict) -> bool:
    """Whether *name* is a key of the config's ``model_aliases`` or ``model.aliases`` (case-insensitive)."""
    wanted = str(name or "").strip().lower()
    if not wanted or not isinstance(config, dict):
        return False
    model_cfg = _model_section(config)
    return any(isinstance(group, dict) and any(str(key).strip().lower() == wanted for key in group)
               for group in (config.get("model_aliases"), model_cfg.get("aliases")))


def alias_route(name: str, config: dict) -> Optional[Any]:
    """The whole ``StartupModelRoute`` (model, provider, base_url, api_key) of alias *name* in *config*,
    resolved by the same owner as CLI startup; None when *name* is not a configured alias."""
    if not is_configured_alias(name, config):
        return None
    from hermes_cli.model_switch import resolve_startup_model_route
    model_cfg = _model_section(config)
    return resolve_startup_model_route(
        str(name).strip(), current_provider=str(model_cfg.get("provider") or "").strip().lower(),
        alias_config=config)


def default_alias_route(config: dict) -> Optional[Any]:
    """``alias_route`` for the configured default model."""
    return alias_route(configured_default_model(config), config)


def runtime_request(route: Any) -> dict:
    """``resolve_runtime_provider`` / ``resolve_runtime_with_fallback`` kwargs for an alias *route*."""
    return {"requested": route.provider or None, "target_model": route.model,
            "explicit_base_url": route.base_url or None, "explicit_api_key": route.api_key or None}


def expand_alias_model(name: str, config: dict) -> str:
    """The backing model id of alias *name*, else *name* unchanged (for comparisons with ``agent.model``)."""
    route = alias_route(name, config)
    return route.model if route is not None and route.model else name


def alias_runtime_request(name: str, config: Any) -> Optional[tuple[str, dict]]:
    """``(backing model, runtime kwargs)`` when *name* is a configured alias in *config*, else None."""
    route = alias_route(name, config) if isinstance(config, dict) else None
    return (route.model, runtime_request(route)) if route is not None else None


def resolve_default_runtime(config: dict) -> tuple[dict, Optional[dict], Optional[Any]]:
    """``(runtime, fallback_entry, alias_route)`` for a profile's config default: the shared
    ``resolve_runtime_with_fallback`` walk, on the alias entry's own model/endpoint/key when
    ``model.default`` names one (resolved against *config* itself), else the plain default route.
    ``alias_route`` is the alias's ``StartupModelRoute`` (None when the default is not an alias), so
    callers name the alias's model and provider, not the bare alias, in notices."""
    from hermes_cli.runtime_provider import frozen_runtime_config, resolve_runtime_with_fallback
    route = default_alias_route(config)
    if route is None:
        runtime, fallback_entry = resolve_runtime_with_fallback(config)
        return runtime, fallback_entry, None
    with frozen_runtime_config(config):
        runtime, fallback_entry = resolve_runtime_with_fallback(config, **runtime_request(route))
    return runtime, fallback_entry, route
