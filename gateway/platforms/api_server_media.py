"""Image read for API-server clients (``GET /api/media?path=...``).

The dashboard already serves stored images at ``GET /api/media``, gated by the
dashboard session token. A client that only holds ``API_SERVER_KEY`` — the
Hermes Browser Extension, scripts, another control plane — cannot redeem that
token, so pictures a user attached to a turn stay as filenames.

This serves the same ``{"data_url": "data:image/...;base64,..."}`` shape over
the credential those clients already hold. The path must be an image file under
the Hermes media roots (``images``, ``screenshots``, ``cache``), resolved
symlink-safe and size-capped, matching the dashboard route's allowlist.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - aiohttp is a hard gateway dependency
    web = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_MAX_BYTES = 25 * 1024 * 1024
_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
    ".ico": "image/x-icon",
}


def _http_routes(self) -> list[tuple[str, str, Any]]:
    async def media(request: web.Request) -> web.Response:
        return await _handle_media(self, request)

    return [("GET", "/api/media", media)]


def _json_error(message: str, status: int, code: str) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": "invalid_request_error", "code": code}},
        status=status,
    )


def _media_roots() -> list[Path]:
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    roots: list[Path] = []
    for name in ("images", "screenshots", "cache"):
        try:
            roots.append((home / name).resolve())
        except (OSError, RuntimeError):
            continue
    return roots


def _read_encoded(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _resolve_target(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def _stat_image(path: Path) -> tuple[bool, int]:
    """File existence and size, off the event loop. Missing is not an error."""
    try:
        return path.is_file(), path.stat().st_size
    except OSError:
        return False, 0


async def _handle_media(self, request: web.Request) -> web.Response:
    """GET /api/media?path=<image file> -> ``{"data_url": ...}``."""
    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err
    raw = str(request.query.get("path") or "").strip()
    if not raw:
        return _json_error("A path is required.", 400, "invalid_path")
    try:
        target = await asyncio.to_thread(_resolve_target, raw)
    except (OSError, RuntimeError, ValueError):
        return _json_error("Invalid path.", 400, "invalid_path")
    content_type = _CONTENT_TYPES.get(target.suffix.lower())
    if not content_type:
        return _json_error("Unsupported media type.", 415, "unsupported_media")
    roots = await asyncio.to_thread(_media_roots)
    if not any(target == root or root in target.parents for root in roots):
        return _json_error("Path is outside the media folders.", 403, "path_not_allowed")
    try:
        is_file, size = await asyncio.to_thread(_stat_image, target)
        if not is_file:
            return _json_error("File not found.", 404, "not_found")
        if size > _MAX_BYTES:
            return _json_error("File is too large.", 413, "media_too_large")
        encoded = await asyncio.to_thread(_read_encoded, target)
    except OSError:
        logger.exception("GET /api/media failed")
        return _json_error("File could not be read.", 500, "media_unreadable")
    return web.json_response({"data_url": f"data:{content_type};base64,{encoded}"})


def _core_routes(self) -> list[tuple]:
    """The adapter's own HTTP rows. Lives here so adding a route does not grow the facade."""
    from gateway.platforms.api_server import _CRON_AVAILABLE

    routes: list[tuple] = [
        ("GET", "/health", self._handle_health),
        ("GET", "/health/detailed", self._handle_health_detailed),
        ("GET", "/v1/health", self._handle_health),
        ("GET", "/v1/models", self._handle_models),
        ("GET", "/api/model/options", self._handle_model_options),
        ("GET", "/v1/capabilities", self._handle_capabilities),
        # Browser-control (gated on browser.extension_control.enabled + API key): POST
        # mints a short-lived ticket, WS consumes it; artifacts are bounded + scope-bound.
        ("POST", "/v1/browser-control/register", self._handle_browser_control_register),
        ("GET", "/v1/browser-control/ws", self._handle_browser_control_ws),
        ("POST", "/v1/artifacts/upload", self._handle_artifact_upload),
        ("GET", "/v1/artifacts/download/{artifact_id}", self._handle_artifact_download),
        ("GET", "/v1/skills", self._handle_skills),
        ("GET", "/v1/toolsets", self._handle_toolsets),
        ("GET", "/api/sessions", self._handle_list_sessions),
        ("POST", "/api/sessions", self._handle_create_session),
        ("GET", "/api/sessions/{session_id}", self._handle_get_session),
        ("PATCH", "/api/sessions/{session_id}", self._handle_patch_session),
        ("DELETE", "/api/sessions/{session_id}", self._handle_delete_session),
        ("GET", "/api/sessions/{session_id}/messages", self._handle_session_messages),
        ("POST", "/api/sessions/{session_id}/fork", self._handle_fork_session),
        ("POST", "/api/sessions/{session_id}/chat", self._handle_session_chat),
        ("POST", "/api/sessions/{session_id}/chat/stream", self._handle_session_chat_stream),
        ("POST", "/api/sessions/{session_id}/model", self._handle_session_model_lock),
        ("POST", "/v1/chat/completions", self._handle_chat_completions),
        ("POST", "/v1/responses", self._handle_responses),
        ("GET", "/v1/responses/{response_id}", self._handle_get_response),
        ("DELETE", "/v1/responses/{response_id}", self._handle_delete_response),
        # Platform event ingress: authenticated by the target adapter's own verifier,
        # NOT API_SERVER_KEY (external platforms hold no API server key).
        ("POST", "/api/platforms/{platform}/events", self._handle_platform_event_callback),
        ("GET", "/api/jobs", self._handle_list_jobs),
        ("POST", "/api/jobs", self._handle_create_job),
        ("GET", "/api/jobs/{job_id}", self._handle_get_job),
        ("PATCH", "/api/jobs/{job_id}", self._handle_update_job),
        ("DELETE", "/api/jobs/{job_id}", self._handle_delete_job),
        ("POST", "/api/jobs/{job_id}/pause", self._handle_pause_job),
        ("POST", "/api/jobs/{job_id}/resume", self._handle_resume_job),
        ("POST", "/api/jobs/{job_id}/run", self._handle_run_job),
    ]
    if _CRON_AVAILABLE:
        # Chronos fire webhook (NAS -> agent): authenticated by a NAS-minted JWT.
        routes.append(("POST", "/api/cron/fire", self._handle_cron_fire))
    return routes
