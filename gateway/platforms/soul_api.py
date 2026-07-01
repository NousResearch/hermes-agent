"""
SOUL.md HTTP endpoints for the OpenAI-compatible API server.

Exposes read/write access to a profile's ``SOUL.md`` (the agent's identity /
persona prompt) under the API server's existing bearer auth, so the master
console can project the Phase-6 capability-growth directive into a live agent's
SOUL through the same authenticated surface it uses for ``config.yaml``.

Design constraints (mirror ``config_api`` / ``kanban_api`` / ``actions_api``):

- **Additive.** Only adds handlers; ``api_server`` imports it lazily and
  registers the routes in ``connect()``.
- **Profile-scoped, opt-in cross-profile targeting.** With no ``?profile=`` the
  endpoint reads/writes the api_server's *own* profile; ``?profile=<name>``
  retargets a named sibling via the same context-local ``HERMES_HOME`` override
  ``config_api`` uses (refuses ``default`` by name; 404 on a missing profile).
- **Reversible writes.** PUT backs up the existing ``SOUL.md`` to
  ``SOUL.md.bak-<timestamp>`` before replacing it, and re-applies owner-only
  perms via the same ``_secure_file`` the seeder uses — so a bad projection is
  one ``mv`` away from undone.

The console composes the SOUL text client-side (insert/replace its managed
``capability-growth`` block, leaving hand-authored identity untouched) and PUTs
the full result here — this seam stays a dumb, auditable file read/write.
"""

import asyncio
import logging
import time
from typing import Optional

from aiohttp import web

# Reuse config_api's auth-scoped profile targeting + error helper verbatim so
# the two seams behave identically (same ?profile= semantics, same 400/404).
from gateway.platforms.config_api import _err, _profile_home, _ProfileTargetError

logger = logging.getLogger(__name__)

_MAX_SOUL_BYTES = 256 * 1024  # generous cap; a persona is a few KB, guards abuse


def _soul_path():
    """Resolve ``SOUL.md`` under the (possibly overridden) HERMES_HOME."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "SOUL.md"


async def handle_get_soul(adapter, request: "web.Request") -> "web.Response":
    """GET /api/soul[?profile=<name>] — return a profile's SOUL.md text."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    profile = request.query.get("profile")

    def _read():
        with _profile_home(profile) as resolved:
            path = _soul_path()
            if not path.exists():
                return resolved, str(path), None
            return resolved, str(path), path.read_text(encoding="utf-8")

    try:
        resolved, path, text = await asyncio.to_thread(_read)
    except _ProfileTargetError as exc:
        return _err(exc.message, status=exc.status, code=exc.code, param="profile")
    except Exception:
        logger.exception("GET /api/soul failed")
        return _err("Failed to read SOUL.md", status=500, code="server_error")

    return web.json_response({
        "profile": resolved,
        "path": path,
        "exists": text is not None,
        "soul": text or "",
    })


async def handle_put_soul(adapter, request: "web.Request") -> "web.Response":
    """PUT /api/soul[?profile=<name>] — replace a profile's SOUL.md.

    Body: ``{"soul": "<full markdown>"}``. Full replace (the console does the
    read-compose-write so identity outside the managed block is preserved).
    Backs the prior file up to ``SOUL.md.bak-<ts>`` first. Takes effect for
    sessions created after the next gateway restart (SOUL is injected at
    session creation) — see ``reload_required`` + ``POST /api/gateway/restart``.
    """
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    body, err = await adapter._read_json_body(request)
    if err:
        return err

    soul = body.get("soul")
    if not isinstance(soul, str) or not soul.strip():
        return _err("'soul' must be a non-empty string", status=400,
                    code="invalid_soul", param="soul")
    if len(soul.encode("utf-8")) > _MAX_SOUL_BYTES:
        return _err(f"SOUL.md exceeds {_MAX_SOUL_BYTES} bytes", status=400,
                    code="soul_too_large", param="soul")

    profile = request.query.get("profile")

    def _write():
        from hermes_cli.config import _secure_file
        with _profile_home(profile) as resolved:
            path = _soul_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            backup = None
            if path.exists():
                backup = path.with_name(f"SOUL.md.bak-{time.strftime('%Y%m%d_%H%M%S')}")
                backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
                _secure_file(backup)
            path.write_text(soul, encoding="utf-8")
            _secure_file(path)
            return resolved, str(path), (str(backup) if backup else None)

    try:
        resolved, path, backup = await asyncio.to_thread(_write)
    except _ProfileTargetError as exc:
        return _err(exc.message, status=exc.status, code=exc.code, param="profile")
    except Exception:
        logger.exception("PUT /api/soul failed")
        return _err("Failed to write SOUL.md", status=500, code="server_error")

    return web.json_response({
        "ok": True,
        "profile": resolved,
        "path": path,
        "backup": backup,
        "reload_required": True,
    })
