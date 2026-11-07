"""Read-only profile roster for external clients (``GET /v1/profiles``).

Since the unified gateway runtime, a profile's roster (bot titles, shapes, colours,
avatars) is reachable only through the dashboard's WebSocket, which redeems tickets minted
on an owner-only control socket. A client that holds just the API-server credential (the
Hermes Browser Extension, Open WebUI, scripts) can reach ``/p/<profile>/`` for each profile
but has no way to learn which profiles exist or how to draw them.

This module serves that roster over the credential those clients already hold. It is
read-only, lists only profiles this listener serves (the same set ``/p/<profile>/``
accepts), and never exposes filesystem paths, secrets or sessions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Avatars are small icons; a hard cap keeps a mistaken multi-megabyte file off the wire.
_AVATAR_MAX_BYTES = 2_000_000
# A roster is a handful of profiles; a bound keeps a pathological profile directory from
# turning one authenticated GET into an unbounded scan, and ``ui_meta`` (free-form, written by
# Desktop and the extension) is dropped from a row when it is larger than a client could use.
_ROSTER_MAX_PROFILES = 1000
_UI_META_MAX_BYTES = 65_536
_AVATAR_TYPES = (("png", "image/png"), ("jpg", "image/jpeg"), ("webp", "image/webp"))


def _http_routes(self) -> list[tuple[str, str, Any]]:
    async def profiles(request: web.Request) -> web.Response:
        return await _handle_profiles(self, request)

    async def profile_avatar(request: web.Request) -> web.Response:
        return await _handle_profile_avatar(self, request)

    return [
        ("GET", "/v1/profiles", profiles),
        ("GET", "/v1/profiles/{name}/avatar", profile_avatar)]


def _json_error(message: str, status: int, code: str) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": "invalid_request_error", "code": code}}, status=status)


def _served(self) -> dict[str, Path]:
    """``{profile name: home}`` for the profiles this listener serves."""
    from hermes_cli.profiles import profiles_to_serve
    runner = getattr(self, "gateway_runner", None)
    multiplex = bool(getattr(getattr(runner, "config", None), "multiplex_profiles", False))
    served = {name: Path(home) for name, home in profiles_to_serve(multiplex=multiplex)}
    # A request made through ``/p/<profile>/`` with that profile's own key sees only that
    # profile; the listener's owner/default credential sees the whole roster.
    from gateway.platforms.api_server import _api_request_profile
    scoped = _api_request_profile.get()
    if scoped and scoped != "default":
        return {name: home for name, home in served.items() if name == scoped}
    return served


def _ui_meta(home: Path) -> tuple[dict, dict]:
    """``(ui_meta, ui_meta_revisions)`` from ``profile.yaml``; empty when absent or unreadable."""
    import hermes_yaml as yaml
    path = home / "profile.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}, {}
    data = data if isinstance(data, dict) else {}
    ui_meta = data.get("ui_meta") if isinstance(data.get("ui_meta"), dict) else {}
    raw = data.get("_ui_meta_revisions")
    revisions = ({str(k): max(0, v) for k, v in raw.items() if type(v) is int}
                 if isinstance(raw, dict) else {})
    return ui_meta, revisions


def _avatar_file(home: Path) -> tuple[Path, str] | None:
    for ext, mime in _AVATAR_TYPES:
        candidate = home / "assets" / f"avatar.{ext}"
        if candidate.is_file():
            return candidate, mime
    return None


def _fits(ui_meta: dict) -> bool:
    try:
        return len(json.dumps(ui_meta, default=str)) <= _UI_META_MAX_BYTES
    except (TypeError, ValueError):
        return False


def _build_roster(self) -> list[dict[str, Any]]:
    from hermes_cli.profiles import list_profiles
    served = _served(self)
    rows: list[dict[str, Any]] = []
    for info in list_profiles(lazy_skill_count=True):
        home = served.get(info.name)
        if home is None:
            continue
        ui_meta, revisions = _ui_meta(home)
        row: dict[str, Any] = {
            "name": info.name,
            "is_default": bool(info.is_default),
            "model": info.model or "",
            "provider": info.provider or "",
            "description": info.description or "",
            "display_name": info.display_name or "",
            "skill_count": int(info.skill_count or 0),
            "has_avatar": _avatar_file(home) is not None,
            "ui_meta_revisions": revisions}
        if ui_meta and _fits(ui_meta):
            row["ui_meta"] = ui_meta
        rows.append(row)
        if len(rows) >= _ROSTER_MAX_PROFILES:
            break
    return rows


async def _handle_profiles(self, request: web.Request) -> web.Response:
    """GET /v1/profiles — the served profiles with their bot metadata."""
    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err
    try:
        rows = await asyncio.to_thread(_build_roster, self)
    except Exception:
        logger.exception("GET /v1/profiles failed")
        return _json_error("Failed to list profiles.", 500, "profiles_failed")
    return web.json_response({"object": "list", "profiles": rows})


async def _handle_profile_avatar(self, request: web.Request) -> web.Response:
    """GET /v1/profiles/{name}/avatar — the profile's avatar image, if it has one."""
    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err
    from hermes_cli.profiles import normalize_profile_name
    try:
        name = normalize_profile_name(request.match_info.get("name", ""))
    except ValueError:
        return _json_error("Unknown profile.", 404, "profile_not_found")
    home = _served(self).get(name)
    if home is None:
        return _json_error("Unknown profile.", 404, "profile_not_found")

    def read() -> tuple[bytes, str] | None:
        found = _avatar_file(home)
        if found is None:
            return None
        path, mime = found
        if path.stat().st_size > _AVATAR_MAX_BYTES:
            return None
        return path.read_bytes(), mime

    try:
        result = await asyncio.to_thread(read)
    except OSError:
        result = None
    if result is None:
        return _json_error("Profile has no avatar.", 404, "avatar_not_found")
    body, mime = result
    return web.Response(body=body, content_type=mime, headers={"Cache-Control": "private, max-age=60"})
