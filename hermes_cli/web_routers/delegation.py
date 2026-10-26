"""Delegation routing routes: which ``delegation.backend`` runs subagents, and each plugin backend's
routes under ``delegation.backends.<name>``.

The settings shapes are the ones the official advanced-delegation plugin reads (``model_map``:
named model routes; ``profile_routes``: Hermes profiles subagents may run as). Writes replace one
backend's block wholesale through the same profile scope + config lock every config route uses, so
deleting a route really deletes it (``PUT /api/config`` deep-merges and cannot drop a key).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any, Callable, Mapping, Optional

from fastapi import APIRouter, HTTPException

from hermes_cli.web_deps import late
from hermes_cli.web_models import DelegationBackendSelect, DelegationModelMapUpdate, DelegationProfileRoutesUpdate
from hermes_cli.web_routers._common import (
    _profile_scope, config_write_scope, is_redacted_credential_preview, redacted_credential_preview,
)

router = APIRouter()
_log = logging.getLogger(__name__)

load_config = late("load_config", "hermes_cli.config")
save_config = late("save_config", "hermes_cli.config")

DEFAULT_BACKEND = "default"
MODEL_MAP = "model_map"
PROFILE_ROUTES = "profile_routes"
_OPTION_TEXT_KEYS = ("name", "description", "provider", "model", "base_url", "api_mode")


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _managed_install(message: str) -> HTTPException:
    """A package-manager-managed install's config.yaml is not ours to write (``save_config`` would skip it and
    report nothing): 409 with core's one-line reason, which the page shows as the error. Not 403, which the
    Desktop transport reads as an auth rejection and answers with a re-login."""
    return HTTPException(status_code=409, detail=" ".join(message.split()))


def _mapping(parent: Any, key: str) -> dict:
    value = parent.get(key) if isinstance(parent, Mapping) else None
    return value if isinstance(value, dict) else {}


def _section(config: dict, key: str) -> dict:
    value = config.get(key)
    if not isinstance(value, dict):
        value = {}
        config[key] = value
    return value


def _registered_backends() -> list[dict[str, Any]]:
    """``default`` plus every plugin-registered backend of the active profile, with availability."""
    from agent.delegation_backends import list_backends
    from agent.delegation_registry import DEFAULT_BACKEND as builtin
    from hermes_cli.plugins import discover_plugins

    discover_plugins()  # idempotent; a plugin enabled after boot loads on the next scan
    # ``description`` is the provider's ``summary``: the one line a backend picker shows for it.
    return [{**asdict(b), "builtin": b.name == builtin} for b in list_backends()]


def _redacted_options(options: Any) -> list[dict[str, Any]]:
    """Model routes as stored, with an ``api_key`` reduced to a display-only preview."""
    if not isinstance(options, list):
        return []
    out = []
    for entry in options:
        if not isinstance(entry, dict):
            continue
        row = dict(entry)
        if row.get("api_key"):
            row["api_key"] = redacted_credential_preview(row["api_key"])
        out.append(row)
    return out


def _profile_names() -> list[str]:
    from hermes_cli.profiles import list_profile_names
    return list_profile_names()


@router.get("/api/delegation/routing")
async def get_delegation_routing(profile: Optional[str] = None):
    """The profile's delegation backend (``configured``) and, when it cannot run here, why
    (``blocked_reason``, ``None`` when it runs: core fails closed, so subagents are refused until it is fixed), the registered
    backends, both route blocks and the profiles a profile route may name."""
    def _read():
        from agent.delegation_registry import backend_status, configured_backend
        with _profile_scope(profile):
            delegation = _mapping(load_config(), "delegation")
            backends = _registered_backends()
            configured = configured_backend(delegation)
            blocks = _mapping(delegation, "backends")
            model_map, routes = _mapping(blocks, MODEL_MAP), _mapping(blocks, PROFILE_ROUTES)
            return {
                "configured": configured,
                "blocked_reason": backend_status(configured)[1] or None,
                "backends": backends,
                MODEL_MAP: {"options": _redacted_options(model_map.get("options")),
                            "default": model_map.get("default") or None},
                PROFILE_ROUTES: {"profiles": [p for p in routes.get("profiles") or [] if isinstance(p, dict)],
                                 "default": routes.get("default") or None},
                "profiles": _profile_names(),
            }

    return await asyncio.to_thread(_read)


@router.put("/api/delegation/backend")
async def select_delegation_backend(body: DelegationBackendSelect, profile: Optional[str] = None):
    """Persist ``delegation.backend`` through core's ``set_global_backend``: only ``default`` or a backend
    that can run for this profile (core fails closed on any other, refusing every subagent)."""
    name = (body.backend or "").strip().lower() or DEFAULT_BACKEND

    def _run():
        from agent.delegation_backends import set_global_backend
        from hermes_cli.plugins import discover_plugins
        with config_write_scope(body.profile or profile):
            discover_plugins()
            try:
                set_global_backend(name)
            except PermissionError as exc:
                raise _managed_install(str(exc)) from exc
            except ValueError as exc:
                raise _bad_request(str(exc)) from exc

    await asyncio.to_thread(_run)
    return {"ok": True, "backend": name}


def _text(entry: dict, key: str) -> str:
    value = entry.get(key)
    return value.strip() if isinstance(value, str) else ""


def _check_default(default: Optional[str], names: list[str], what: str) -> Optional[str]:
    value = (default or "").strip()
    if value and value not in names:
        raise _bad_request(f"Default {what} {value!r} is not in the list.")
    return value or None


def _clean_model_option(index: int, entry: dict, stored: dict[str, dict]) -> dict[str, Any]:
    from agent.delegation_provider import ROUTE_KEYS
    from agent.fast_mode import STATIC_TIERS, parse_exact_service_tier
    from hermes_constants import VALID_REASONING_EFFORTS, parse_reasoning_effort

    row = {key: _text(entry, key) for key in _OPTION_TEXT_KEYS if _text(entry, key)}
    label = f"Route {index + 1}"
    if not row.get("name"):
        raise _bad_request(f"{label} needs a name.")
    label = f"Route {row['name']!r}"
    if not row.get("description"):
        raise _bad_request(f"{label} needs a description: it is what the agent reads to choose it.")
    if not row.get("model"):
        raise _bad_request(f"{label} needs a model.")
    if not (row.get("provider") or row.get("base_url")):
        raise _bad_request(f"{label} needs a provider.")
    effort = _text(entry, "reasoning_effort")
    if effort:
        if parse_reasoning_effort(effort) is None:
            raise _bad_request(f"{label}: unknown reasoning effort {effort!r}. "
                               f"Use none or one of: {', '.join(VALID_REASONING_EFFORTS)}.")
        row["reasoning_effort"] = effort
    tier = _text(entry, "service_tier")
    if tier:
        try:
            parsed = parse_exact_service_tier(tier)
        except ValueError:
            parsed = None
        if parsed is None or (parsed and parsed not in STATIC_TIERS):
            raise _bad_request(f"{label}: unknown service tier {tier!r}. Use normal, fast or ultrafast.")
        row["service_tier"] = tier
    if isinstance(entry.get("fallback_providers"), list):
        row["fallback_providers"] = entry["fallback_providers"]
    # Route fields the editor never touches (request_overrides, label, profile) round-trip as stored.
    validated = {*_OPTION_TEXT_KEYS, "api_key", "reasoning_effort", "service_tier", "fallback_providers"}
    for field in sorted(ROUTE_KEYS - validated):
        if entry.get(field) not in (None, "", [], {}):
            row[field] = entry[field]
    key = entry.get("api_key")
    if is_redacted_credential_preview(key):
        # The renderer only ever saw a preview: keep the stored key of the route it came from (a rename names
        # it in original_name; once that save landed, the route is stored under its new name).
        source = stored.get(_text(entry, "original_name")) or stored.get(row["name"]) or {}
        key = source.get("api_key")
    if isinstance(key, str) and key.strip():
        row["api_key"] = key.strip()
    return row


def _write_backend_block(profile: Optional[str], backend: str, build: Callable[[dict], dict]) -> dict:
    from hermes_cli.config import format_managed_message, is_managed
    if is_managed():
        raise _managed_install(format_managed_message(f"save delegation.backends.{backend}"))
    with config_write_scope(profile):
        config = load_config()
        blocks = _section(_section(config, "delegation"), "backends")
        block = build(_mapping(blocks, backend))
        blocks[backend] = block
        save_config(config)
        return block


@router.put("/api/delegation/backends/model_map")
async def save_model_routes(body: DelegationModelMapUpdate, profile: Optional[str] = None):
    """Replace ``delegation.backends.model_map`` (``options`` + ``default``) after validating every route."""
    def _build(current: dict) -> dict:
        stored = {_text(o, "name"): o for o in current.get("options") or [] if isinstance(o, dict)}
        options = [_clean_model_option(i, dict(o), stored) for i, o in enumerate(body.options)]
        names = [o["name"] for o in options]
        duplicate = next((n for n in names if names.count(n) > 1), None)
        if duplicate:
            raise _bad_request(f"Two routes are named {duplicate!r}; route names must be unique.")
        block: dict[str, Any] = {"options": options}
        default = _check_default(body.default, names, "route")
        if default:
            block["default"] = default
        return block

    block = await asyncio.to_thread(_write_backend_block, body.profile or profile, MODEL_MAP, _build)
    return {"ok": True, MODEL_MAP: {"options": _redacted_options(block["options"]),
                                    "default": block.get("default")}}


@router.put("/api/delegation/backends/profile_routes")
async def save_profile_routes(body: DelegationProfileRoutesUpdate, profile: Optional[str] = None):
    """Replace ``delegation.backends.profile_routes`` (``profiles`` + ``default``); every name must be an
    existing profile."""
    def _build(_current: dict) -> dict:
        known = set(_profile_names())
        profiles: list[dict[str, str]] = []
        names: list[str] = []
        for entry in body.profiles:
            name = _text(entry, "name")
            if not name:
                raise _bad_request("A profile route needs a profile name.")
            if name not in known:
                raise _bad_request(f"Profile {name!r} does not exist.")
            if name in names:
                raise _bad_request(f"Profile {name!r} is listed twice.")
            names.append(name)
            description = _text(entry, "description")
            profiles.append({"name": name, **({"description": description} if description else {})})
        block: dict[str, Any] = {"profiles": profiles}
        default = _check_default(body.default, names, "profile")
        if default:
            block["default"] = default
        return block

    block = await asyncio.to_thread(_write_backend_block, body.profile or profile, PROFILE_ROUTES, _build)
    return {"ok": True, PROFILE_ROUTES: {"profiles": block["profiles"], "default": block.get("default")}}
