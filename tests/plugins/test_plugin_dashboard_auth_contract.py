"""Guardrail: dashboard plugins must NOT read the session token directly.

The dashboard host exposes a sanctioned, gated-mode-aware auth surface on the
plugin SDK (``window.__HERMES_PLUGIN_SDK__``): ``fetchJSON`` (JSON REST),
``authedFetch`` (uploads / blob downloads), and ``buildWsUrl`` /
``buildWsAuthParam`` (WebSockets). These handle BOTH dashboard auth modes —
loopback (``X-Hermes-Session-Token`` header) and gated OAuth
(``hermes_session_at`` cookie / single-use ``?ticket=``).

Plugins that hand-roll ``fetch`` / ``WebSocket`` and read
``window.__HERMES_SESSION_TOKEN__`` directly send an empty token in gated mode
and 401/1008. That bug shipped in the kanban and achievements plugins and was
invisible until the dashboard ran gated on hosted Fly agents.

This test fails if any bundled plugin's frontend reads the token global
directly, forcing new/edited plugins through the SDK surface instead. It is
the enforcement half of the "single sanctioned auth surface" design — the SDK
helpers are the carrot, this test is the stick.

If you have a legitimate reason to reference the token name (e.g. a comment
explaining why NOT to use it), add the file to ``_ALLOWED_FILES`` with a note.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# Repo root: tests/plugins/<this file> → ../../
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGINS_DIR = _REPO_ROOT / "plugins"

# The forbidden global. Reading it directly bypasses the gated-mode auth path.
_FORBIDDEN = "__HERMES_SESSION_TOKEN__"

# Files explicitly allowed to mention the token (none today). Map path →
# reason so the allowance is self-documenting if one is ever needed.
_ALLOWED_FILES: dict[str, str] = {}


def _plugin_frontend_bundles() -> list[Path]:
    """Every plugin-shipped JS bundle the dashboard loads into the browser."""
    if not _PLUGINS_DIR.is_dir():
        return []
    # Plugin dashboards live at plugins/<name>/dashboard/dist/*.js
    return sorted(_PLUGINS_DIR.glob("*/dashboard/dist/*.js"))


def test_there_are_plugin_bundles_to_check() -> None:
    """Sanity: the glob actually finds the bundles, so a future layout change
    doesn't silently turn this guard into a no-op."""
    bundles = _plugin_frontend_bundles()
    names = {b.parent.parent.parent.name for b in bundles}
    # kanban + hermes-achievements are bundled today; assert at least one is
    # found so the guard can't pass vacuously.
    assert bundles, "no plugin dashboard bundles found — glob/layout drift?"
    assert names, "could not resolve plugin names from bundle paths"


@pytest.mark.parametrize(
    "bundle",
    _plugin_frontend_bundles(),
    ids=lambda p: str(p.relative_to(_REPO_ROOT)),
)
def test_plugin_bundle_does_not_read_session_token(bundle: Path) -> None:
    rel = str(bundle.relative_to(_REPO_ROOT))
    text = bundle.read_text(encoding="utf-8", errors="replace")

    if rel in _ALLOWED_FILES:
        return  # explicitly allowed (with a documented reason)

    # Only flag CODE reads of the token global, not mentions in ``//`` comments
    # (e.g. a comment explaining why the SDK helper is used instead). A line is
    # a code read if it contains the global and the global appears before any
    # ``//`` comment marker on that line.
    offending: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        idx = line.find(_FORBIDDEN)
        if idx == -1:
            continue
        comment_idx = line.find("//")
        in_comment = comment_idx != -1 and comment_idx < idx
        if not in_comment:
            offending.append(f"  {i}: {line.strip()}")

    if not offending:
        return

    pytest.fail(
        f"{rel} reads {_FORBIDDEN} directly — this bypasses gated-mode auth "
        f"and 401/1008s on OAuth-gated dashboards. Use the plugin SDK instead: "
        f"SDK.fetchJSON (JSON), SDK.authedFetch (uploads/downloads), or "
        f"SDK.buildWsUrl (WebSockets). Offending lines:\n" + "\n".join(offending)
    )


# ── Backend auth contract: mounted plugin behind dashboard auth ──────────────


def _workflows_plugin_router():
    """Load the workflows plugin router without hitting the full dashboard."""
    import importlib.util
    import sys

    plugin_file = _REPO_ROOT / "plugins" / "workflows" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_workflows_auth_test", plugin_file
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


def test_mounted_workflows_plugin_rejects_unauthenticated_definitions():
    """Unauthenticated GET /api/plugins/workflows/definitions returns 401
    when the plugin is mounted behind the dashboard's auth layer."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient
    from starlette.middleware.base import BaseHTTPMiddleware

    _TEST_TOKEN = "test-session-token-for-auth-contract"

    class SessionAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            token = request.headers.get("X-Hermes-Session-Token")
            if token != _TEST_TOKEN:
                return JSONResponse(status_code=401, content={"detail": "unauthenticated"})
            return await call_next(request)

    app = FastAPI()
    app.add_middleware(SessionAuthMiddleware)
    app.include_router(_workflows_plugin_router(), prefix="/api/plugins/workflows")

    with TestClient(app) as client:
        r = client.get("/api/plugins/workflows/definitions")
        assert r.status_code == 401

        r = client.get(
            "/api/plugins/workflows/definitions",
            headers={"X-Hermes-Session-Token": _TEST_TOKEN},
        )
        assert r.status_code == 200
        assert "definitions" in r.json()
