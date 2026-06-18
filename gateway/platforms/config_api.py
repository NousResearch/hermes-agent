"""
Config + profile HTTP endpoints for the OpenAI-compatible API server.

Exposes read/write access to a profile's ``config.yaml`` and the
profile lifecycle (list/create) under the API server's existing bearer
auth, so a headless control plane (the "master console") can drive
capability provisioning through ONE authenticated surface instead of
falling back to the dashboard server's session auth.

Design constraints (mirror ``kanban_api`` / ``actions_api``):

- **Additive.** This module only adds handlers; it changes no existing
  route, the CLI, or the gateway. ``api_server`` imports it lazily and
  registers the routes in ``connect()``.
- **Reuse the real code paths.** Writes go through the same
  ``hermes_cli.config.save_config`` the dashboard and CLI use; profile
  creation goes through ``hermes_cli.profiles.create_profile``. We never
  hand-roll YAML or directory layout.
- **Profile-scoped, with opt-in cross-profile targeting.** With no
  ``?profile=`` the endpoint reads/writes the api_server's *own* profile
  (the common case: the console addresses each agent on its own port).
  ``?profile=<name>`` retargets a *named* sibling profile via a
  context-local ``HERMES_HOME`` override (per-request safe, never mutates
  ``os.environ``) — this is what lets the console configure a freshly
  created profile before its gateway is running.

The handlers are plain ``async def f(adapter, request)`` functions so the
whole surface lives in one module; the adapter still owns auth
(``_check_auth``) and body parsing (``_read_json_body``).
"""

import asyncio
import contextlib
import logging
import os
import subprocess
import sys
from typing import Any, Dict, Optional

from aiohttp import web

from gateway.platforms.api_server import _openai_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _err(message: str, *, status: int, code: str, param: Optional[str] = None) -> "web.Response":
    return web.json_response(_openai_error(message, param=param, code=code), status=status)


class _ProfileTargetError(Exception):
    """A ``?profile=`` value that can't be resolved to a writable home."""

    def __init__(self, message: str, *, status: int = 400, code: str = "invalid_profile"):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


@contextlib.contextmanager
def _profile_home(profile: Optional[str]):
    """Scope ``load_config``/``save_config`` to ``profile`` for one call.

    Yields the resolved profile label. ``None``/``""``/``"current"`` means the
    api_server's own profile (no override). A named profile sets a
    context-local ``HERMES_HOME`` override that ``load_config``/``save_config``
    resolve at call time (same seam the dashboard's ``_profile_scope`` uses for
    its config reads/writes). We refuse to retarget ``default`` by name because
    its home is the root ``~/.hermes`` rather than a ``profiles/<name>`` dir —
    address the default profile's own API server instead.
    """
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.profiles import (
        normalize_profile_name,
        validate_profile_name,
        get_profile_dir,
    )

    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        yield "current"
        return

    canon = normalize_profile_name(requested)
    if canon == "default":
        raise _ProfileTargetError(
            "Refusing to target the 'default' profile by name; call its own "
            "API server (omit ?profile=) instead."
        )
    try:
        validate_profile_name(canon)
    except Exception as exc:  # validate_profile_name raises on bad names
        raise _ProfileTargetError(f"Invalid profile name: {exc}") from exc

    profile_dir = get_profile_dir(canon)
    if not profile_dir.is_dir():
        raise _ProfileTargetError(
            f"Profile '{canon}' does not exist", status=404, code="profile_not_found"
        )

    token = set_hermes_home_override(str(profile_dir))
    try:
        yield canon
    finally:
        reset_hermes_home_override(token)


def _profile_summary(info: Any) -> Dict[str, Any]:
    """Serialize a ``hermes_cli.profiles.ProfileInfo`` to JSON-safe fields."""
    return {
        "name": info.name,
        "path": str(info.path),
        "is_default": info.is_default,
        "gateway_running": info.gateway_running,
        "model": info.model,
        "provider": info.provider,
        "skill_count": info.skill_count,
        "description": info.description,
        "distribution_name": info.distribution_name,
        "distribution_version": info.distribution_version,
    }


# ---------------------------------------------------------------------------
# Config: GET / PUT
# ---------------------------------------------------------------------------


async def handle_get_config(adapter, request: "web.Request") -> "web.Response":
    """GET /api/config[?profile=<name>] — return a profile's parsed config.yaml."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    profile = request.query.get("profile")

    def _read():
        from hermes_cli.config import load_config
        with _profile_home(profile) as resolved:
            return load_config(), resolved

    try:
        config, resolved = await asyncio.to_thread(_read)
    except _ProfileTargetError as exc:
        return _err(exc.message, status=exc.status, code=exc.code, param="profile")
    except Exception:
        logger.exception("GET /api/config failed")
        return _err("Failed to load config", status=500, code="server_error")

    return web.json_response({"profile": resolved, "config": config})


async def handle_put_config(adapter, request: "web.Request") -> "web.Response":
    """PUT /api/config[?profile=<name>] — replace a profile's config.yaml.

    Body: ``{"config": {<full config dict>}}``. This is a **full replace**
    (parity with the dashboard's ``PUT /api/config``), so callers read the
    current config, modify it, and write the whole object back. Config changes
    take effect on the next gateway start — see ``reload_required`` in the
    response and ``POST /api/gateway/restart``.
    """
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    body, err = await adapter._read_json_body(request)
    if err:
        return err

    config = body.get("config")
    if not isinstance(config, dict) or not config:
        return _err(
            "'config' must be a non-empty object",
            status=400,
            code="invalid_config",
            param="config",
        )

    profile = request.query.get("profile")

    def _write():
        from hermes_cli.config import save_config
        with _profile_home(profile) as resolved:
            save_config(config)
            return resolved

    try:
        resolved = await asyncio.to_thread(_write)
    except _ProfileTargetError as exc:
        return _err(exc.message, status=exc.status, code=exc.code, param="profile")
    except Exception:
        logger.exception("PUT /api/config failed")
        return _err("Failed to save config", status=500, code="server_error")

    return web.json_response({"ok": True, "profile": resolved, "reload_required": True})


# ---------------------------------------------------------------------------
# Profiles: GET (list) / POST (create)
# ---------------------------------------------------------------------------


async def handle_list_profiles(adapter, request: "web.Request") -> "web.Response":
    """GET /api/profiles — list profiles on this host (for the fleet view)."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    def _list():
        from hermes_cli.profiles import list_profiles
        return [_profile_summary(p) for p in list_profiles()]

    try:
        profiles = await asyncio.to_thread(_list)
    except Exception:
        logger.exception("GET /api/profiles failed")
        return _err("Failed to list profiles", status=500, code="server_error")

    return web.json_response({"count": len(profiles), "profiles": profiles})


async def handle_create_profile(adapter, request: "web.Request") -> "web.Response":
    """POST /api/profiles — create a new profile directory.

    Body: ``{"name": <str>, "clone_from"?: <str>, "clone_config"?: <bool>,
    "no_skills"?: <bool>, "no_alias"?: <bool>, "description"?: <str>}``.
    Wraps ``hermes_cli.profiles.create_profile`` so directory layout, name
    validation, and skill seeding match a CLI-created profile exactly. Set the
    new profile's tools/skills/model afterward via ``PUT /api/config?profile=``.
    """
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    body, err = await adapter._read_json_body(request)
    if err:
        return err

    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        return _err("'name' is required", status=400, code="missing_name", param="name")

    clone_from = body.get("clone_from")
    if clone_from is not None and not isinstance(clone_from, str):
        return _err("'clone_from' must be a string", status=400, code="invalid_clone_from", param="clone_from")

    description = body.get("description")
    if description is not None and not isinstance(description, str):
        return _err("'description' must be a string", status=400, code="invalid_description", param="description")

    clone_config = bool(body.get("clone_config", False))
    no_skills = bool(body.get("no_skills", False))
    no_alias = bool(body.get("no_alias", False))

    def _create():
        from hermes_cli.profiles import create_profile, normalize_profile_name
        path = create_profile(
            name,
            clone_from=clone_from,
            clone_config=clone_config,
            no_skills=no_skills,
            no_alias=no_alias,
            description=description,
        )
        return normalize_profile_name(name), path

    try:
        canon, path = await asyncio.to_thread(_create)
    except FileExistsError as exc:
        return _err(str(exc), status=409, code="profile_exists", param="name")
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc), status=400, code="invalid_profile", param="name")
    except Exception:
        logger.exception("POST /api/profiles failed")
        return _err("Failed to create profile", status=500, code="server_error")

    return web.json_response({"ok": True, "profile": canon, "path": str(path)}, status=201)


# ---------------------------------------------------------------------------
# Reload: POST /api/gateway/restart
# ---------------------------------------------------------------------------


async def handle_restart_gateway(adapter, request: "web.Request") -> "web.Response":
    """POST /api/gateway/restart — restart this gateway so a config write takes effect.

    Spawns a detached ``hermes gateway restart`` using the running
    interpreter's ``hermes_cli.main`` (same mechanism the dashboard uses), so
    the in-flight request returns before the gateway bounces.
    """
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    def _spawn() -> int:
        proc = subprocess.Popen(
            [sys.executable, "-m", "hermes_cli.main", "gateway", "restart"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "HERMES_NONINTERACTIVE": "1"},
        )
        return proc.pid

    try:
        pid = await asyncio.to_thread(_spawn)
    except Exception:
        logger.exception("POST /api/gateway/restart failed")
        return _err("Failed to restart gateway", status=500, code="server_error")

    return web.json_response({"ok": True, "pid": pid, "name": "gateway-restart"})
