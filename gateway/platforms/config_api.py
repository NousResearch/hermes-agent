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
import shutil
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


def _resolve_named_profile(name: object):
    """Resolve a named sibling profile to ``(canon, profile_dir)``.

    Shared by gateway start/stop + archive. Refuses the ``default`` profile (its
    home is the root ``~/.hermes``, not ``profiles/<name>``, so it must never be
    started/stopped/torn down by these routes) and 404s on a missing dir —
    mirrors ``_profile_home``. Raises ``_ProfileTargetError`` on any rejection.
    """
    from hermes_cli.profiles import (
        normalize_profile_name,
        validate_profile_name,
        get_profile_dir,
    )

    if not isinstance(name, str) or not name.strip():
        raise _ProfileTargetError("'profile' is required", status=400, code="missing_profile")
    canon = normalize_profile_name(name)
    if canon == "default":
        raise _ProfileTargetError(
            "Refusing to target the 'default' profile by name; address its own "
            "API server instead."
        )
    try:
        validate_profile_name(canon)
    except Exception as exc:  # validate_profile_name raises on bad names
        raise _ProfileTargetError(
            f"Invalid profile name: {exc}", status=400, code="invalid_profile"
        ) from exc
    profile_dir = get_profile_dir(canon)
    if not profile_dir.is_dir():
        raise _ProfileTargetError(
            f"Profile '{canon}' does not exist", status=404, code="profile_not_found"
        )
    return canon, profile_dir


def _sibling_gateway_env(profile_dir) -> dict:
    """Env for spawning gateway CLI ops (start/stop/uninstall) on a SIBLING profile.

    Drops ``_HERMES_GATEWAY`` — set in THIS running gateway's own environment —
    so the CLI's "refuse to stop/restart from inside the gateway process" guard
    (anti restart-loop, gateway.py) doesn't false-positive. Safe here because
    these routes only ever target a *named sibling* profile (``default`` is
    refused by ``_resolve_named_profile``), never the profile hosting this API
    server, so there is no self-kill / restart-loop risk.
    """
    env = {k: v for k, v in os.environ.items() if k != "_HERMES_GATEWAY"}
    env["HERMES_NONINTERACTIVE"] = "1"
    env["HERMES_HOME"] = str(profile_dir)
    return env


def _seed_api_server_env(profile_path: Any, port: int, key: str) -> None:
    """Append API_SERVER_* settings to a freshly created profile's ``.env``.

    ``create_profile`` seeds an empty, 0600 ``.env``; we add the three lines that
    bring up the profile's own bearer API server on ``port`` with the shared
    ``key``. Any pre-existing ``API_SERVER_*`` lines are dropped first so a retry
    is idempotent rather than appending duplicates.
    """
    from pathlib import Path

    env_path = Path(profile_path) / ".env"
    existing = ""
    if env_path.exists():
        existing = env_path.read_text(encoding="utf-8")
    kept = [
        ln for ln in existing.splitlines()
        if not ln.lstrip().startswith(("API_SERVER_ENABLED", "API_SERVER_PORT", "API_SERVER_KEY"))
    ]
    body = "\n".join(kept).rstrip("\n")
    block = (
        "# API server — seeded at profile creation so the control plane can reach it.\n"
        "API_SERVER_ENABLED=true\n"
        f"API_SERVER_PORT={port}\n"
        f"API_SERVER_KEY={key}\n"
    )
    env_path.write_text((body + "\n\n" if body else "") + block, encoding="utf-8")
    os.chmod(str(env_path), 0o600)


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
    "no_skills"?: <bool>, "no_alias"?: <bool>, "description"?: <str>,
    "api_server_port"?: <int>}``. Wraps ``hermes_cli.profiles.create_profile``
    so directory layout, name validation, and skill seeding match a
    CLI-created profile exactly. Set the new profile's tools/skills/model
    afterward via ``PUT /api/config?profile=``.

    When ``api_server_port`` is given, the new profile's ``.env`` is seeded with
    ``API_SERVER_ENABLED``/``API_SERVER_PORT``/``API_SERVER_KEY`` (the shared key
    of this running server) so ``POST /api/gateway/start`` can bring up a gateway
    the control plane can reach on that port.
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

    # Optional: seed the new profile's own API server so a control plane can
    # reach it on a dedicated port right after creation. Without this the
    # profile's .env has no API_SERVER_* and its gateway would expose nothing.
    api_server_port = body.get("api_server_port")
    if api_server_port is not None:
        if isinstance(api_server_port, bool) or not isinstance(api_server_port, int):
            return _err("'api_server_port' must be an integer", status=400,
                        code="invalid_api_server_port", param="api_server_port")
        if not (1024 <= api_server_port <= 65535):
            return _err("'api_server_port' must be in 1024-65535", status=400,
                        code="invalid_api_server_port", param="api_server_port")
        if api_server_port == adapter._port:
            return _err(f"'api_server_port' {api_server_port} collides with this "
                        "server's port", status=400,
                        code="invalid_api_server_port", param="api_server_port")

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

    result: Dict[str, Any] = {"ok": True, "profile": canon, "path": str(path)}

    if api_server_port is not None:
        # Append the API-server settings to the profile's .env (create_profile
        # seeds an empty, 0600 .env). The shared bearer key is read from THIS
        # running server (adapter._api_key) so the whole fleet authenticates
        # with one key — same value the console already holds.
        try:
            await asyncio.to_thread(
                _seed_api_server_env, path, api_server_port, adapter._api_key
            )
            result["api_server_port"] = api_server_port
        except Exception:
            logger.exception("Seeding API-server env for '%s' failed", canon)
            return _err("Profile created but seeding its API-server env failed",
                        status=500, code="env_seed_failed")

    return web.json_response(result, status=201)


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


async def handle_start_gateway(adapter, request: "web.Request") -> "web.Response":
    """POST /api/gateway/start — start a *named* sibling profile's gateway as a service.

    Body: ``{"profile": <name>}``. Brings up a different profile's gateway by
    running ``gateway start`` (with ``--profile <name>``), which installs + loads
    the profile's launchd/systemd **service** — the same way every real agent
    runs (``ai.hermes.gateway-<profile>``). This is how a freshly-created profile
    whose gateway has never run gets a managed, login-surviving, cleanly-
    stoppable gateway (a bare ``gateway run`` double-spawns and can't be stopped
    by ``gateway stop``). Idempotent (restart-if-running). Refuses ``default``
    (manage its own server) and 404s on an unknown profile.

    Starting is asynchronous — the service takes a moment to bind; the caller
    polls ``/health/detailed`` on the seeded ``API_SERVER_PORT``.
    """
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    body, err = await adapter._read_json_body(request)
    if err:
        return err

    try:
        canon, profile_dir = await asyncio.to_thread(_resolve_named_profile, body.get("profile"))
    except _ProfileTargetError as exc:
        return _err(exc.message, status=exc.status, code=exc.code, param="profile")

    def _start() -> int:
        # `--profile <name> gateway start` installs + loads the launchd/systemd
        # service for the profile. The CLI matches the profile by the `--profile`
        # flag (not HERMES_HOME), so start/stop/uninstall must all pass it.
        proc = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "--profile", canon, "gateway", "start"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_sibling_gateway_env(profile_dir),
            timeout=60,
        )
        return proc.returncode

    try:
        rc = await asyncio.to_thread(_start)
    except Exception:
        logger.exception("POST /api/gateway/start failed")
        return _err("Failed to start gateway", status=500, code="server_error")

    # The service is loading; the caller polls /health/detailed for readiness.
    return web.json_response({"ok": True, "profile": canon, "started": rc == 0})


async def handle_stop_gateway(adapter, request: "web.Request") -> "web.Response":
    """POST /api/gateway/stop — stop a *named* sibling profile's gateway (teardown).

    Body: ``{"profile": <name>}``. Runs ``--profile <name> gateway stop`` to boot
    out the profile's launchd/systemd service (the CLI matches the running
    gateway by the ``--profile`` flag, so it must mirror how ``start`` launched
    it). Idempotent: stopping an already-stopped gateway returns 200 with
    ``stopped: true``. Refuses ``default``, 404s on unknown.
    """
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    body, err = await adapter._read_json_body(request)
    if err:
        return err

    try:
        canon, profile_dir = await asyncio.to_thread(_resolve_named_profile, body.get("profile"))
    except _ProfileTargetError as exc:
        return _err(exc.message, status=exc.status, code=exc.code, param="profile")

    def _stop() -> bool:
        import time
        from gateway.status import get_running_pid

        # Pass `--profile <name>`: `gateway stop` matches the running gateway by
        # the `--profile` flag in its cmdline (not HERMES_HOME), so this must
        # mirror how `start` launched it or stop finds "no gateway running". We
        # confirm from ground truth (the profile's gateway.pid) rather than the
        # exit code, and BLOCK until the process is gone so a follow-up archive
        # doesn't race.
        subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "--profile", canon, "gateway", "stop"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_sibling_gateway_env(profile_dir),
            timeout=30,
        )
        for _ in range(20):  # up to ~10s for the process to exit + pid file clear
            if not get_running_pid(profile_dir / "gateway.pid", cleanup_stale=True):
                return True
            time.sleep(0.5)
        return False

    try:
        stopped = await asyncio.to_thread(_stop)
    except Exception:
        logger.exception("POST /api/gateway/stop failed")
        return _err("Failed to stop gateway", status=500, code="server_error")

    # `stopped` is ground truth (pid gone). An already-stopped gateway also
    # returns True here, so teardown is idempotent.
    return web.json_response({"ok": True, "profile": canon, "stopped": stopped})


async def handle_archive_profile(adapter, request: "web.Request") -> "web.Response":
    """POST /api/profiles/{name}/archive — move a profile dir aside (reversible teardown).

    Moves ``$HERMES_HOME/profiles/<name>`` to
    ``$HERMES_HOME/profiles/.archived/<name>-<UTC-ts>`` (reverse by moving it
    back). Refuses ``default``, 404s on unknown, and 409s if the gateway is still
    running (the caller is expected to ``/api/gateway/stop`` first).
    """
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    try:
        canon, profile_dir = await asyncio.to_thread(
            _resolve_named_profile, request.match_info.get("name")
        )
    except _ProfileTargetError as exc:
        return _err(exc.message, status=exc.status, code=exc.code, param="name")

    def _archive():
        from datetime import datetime, timezone
        from gateway.status import get_running_pid

        pid = get_running_pid(profile_dir / "gateway.pid", cleanup_stale=True)
        if pid:
            raise _ProfileTargetError(
                f"Profile '{canon}' gateway is still running (pid {pid}); stop it first.",
                status=409, code="gateway_running",
            )
        # Remove the launchd/systemd service definition so no orphan plist points
        # at the archived dir (best-effort — the gateway is already stopped).
        subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "--profile", canon, "gateway", "uninstall"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_sibling_gateway_env(profile_dir),
            timeout=30,
        )
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived_root = profile_dir.parent / ".archived"
        archived_root.mkdir(parents=True, exist_ok=True)
        dest = archived_root / f"{canon}-{ts}"
        shutil.move(str(profile_dir), str(dest))
        return dest

    try:
        dest = await asyncio.to_thread(_archive)
    except _ProfileTargetError as exc:
        return _err(exc.message, status=exc.status, code=exc.code, param="name")
    except Exception:
        logger.exception("POST /api/profiles/%s/archive failed", canon)
        return _err("Failed to archive profile", status=500, code="server_error")

    return web.json_response(
        {"ok": True, "profile": canon, "archived": True, "path": str(dest)}, status=200
    )


# ---------------------------------------------------------------------------
# Versioned write path — fleet-state snapshot trigger (master_console P8 M0)
# ---------------------------------------------------------------------------

import re as _re

# Profile names are passed straight to the snapshot binary's argv; constrain to
# the Hermes profile-id charset as defence-in-depth (subprocess uses the list
# form, so this is belt-and-braces, not the only guard against injection).
_SNAPSHOT_PROFILE_RE = _re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _snapshot_bin() -> Optional[str]:
    """Resolve the host `fleet-state-snapshot` script (installed to ~/.local/bin).

    Overridable via FLEET_STATE_SNAPSHOT_BIN; falls back to PATH lookup."""
    override = os.environ.get("FLEET_STATE_SNAPSHOT_BIN")
    if override and os.path.isfile(override):
        return override
    candidate = os.path.expanduser("~/.local/bin/fleet-state-snapshot")
    if os.path.isfile(candidate):
        return candidate
    return shutil.which("fleet-state-snapshot")


async def handle_snapshot(adapter, request: "web.Request") -> "web.Response":
    """POST /api/snapshot — commit a fleet-state snapshot of the agents' brains.

    Body: ``{"reason": <str>, "profile": <name|null>}``. Shells out to the host
    ``fleet-state-snapshot`` script (the P8 M0 versioned write path), which
    captures each profile's editable, non-secret state (config / SOUL / routines /
    skills) into the ``~/fleet-state`` git repo so the change is diffable and
    revertible. The console calls this right AFTER an apply (provision / config /
    SOUL / routine edit) so console-initiated changes are committed immediately
    with a precise reason; a launchd cadence is the safety net for everything else.

    Runs host-native like the gateway start/stop seams — the API server lives in a
    host gateway process and can spawn host scripts (the containerised console
    can't). Idempotent: a no-op snapshot returns ``committed: false``.
    """
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    body, err = await adapter._read_json_body(request)
    if err:
        return err

    reason = str(body.get("reason") or "console snapshot").strip()[:200]
    profile = body.get("profile")
    if profile is not None:
        profile = str(profile).strip()
        if not _SNAPSHOT_PROFILE_RE.match(profile):
            return _err("invalid profile", status=400, code="invalid_profile", param="profile")

    binary = _snapshot_bin()
    if not binary:
        return _err("fleet-state-snapshot not installed on host", status=503,
                    code="snapshot_unavailable")

    def _run() -> "subprocess.CompletedProcess":
        cmd = [binary, "-m", reason]
        if profile:
            cmd += ["--profile", profile]
        return subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )

    try:
        proc = await asyncio.to_thread(_run)
    except Exception:
        logger.exception("POST /api/snapshot failed")
        return _err("Failed to run snapshot", status=500, code="server_error")

    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        return _err(f"snapshot failed: {(proc.stderr or out).strip()[:300]}",
                    status=500, code="snapshot_error")
    committed = "no changes" not in out
    # The script prints "committed <sha> — <reason>" on a commit.
    sha = None
    m = _re.search(r"committed (\w+)", out)
    if m:
        sha = m.group(1)
    return web.json_response(
        {"ok": True, "committed": committed, "sha": sha, "reason": reason,
         "profile": profile, "output": out}, status=200)
