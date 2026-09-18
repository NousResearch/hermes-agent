"""End-to-end: the REAL dashboard app, the REAL plugin discovery path, a REAL plugin page.

Not a unit test with a fake router. This writes a small fixture plugin into a temp
``HERMES_HOME``, runs Hermes' real plugin discovery so the plugin registers its page through
``ctx.register_dashboard_page``, boots ``hermes_cli.web_server`` with its real SPA build mounted,
then drives ``/p/<slug>`` over HTTP and WebSocket — including two concurrent sockets each holding
a real PTY child, which is the shape a terminal-pane plugin needs.

Lives in ``scripts/e2e/`` rather than ``tests/``: it needs a built ``hermes_cli/web_dist`` (without
the SPA the catch-all check cannot detect the mount-order hazard it exists to catch) and it spawns
real processes, so it is run deliberately, not by the pytest sweep.

Run (no external plugin required — the fixture is generated):

    HERMES_HOME=$(mktemp -d) .venv/bin/python scripts/e2e/dashboard_plugin_page.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

FAILURES: list[str] = []

SLUG = "e2epage"

#: A minimal plugin that exercises everything a real page needs: an HTML document that uses the
#: core-provided bootstrap instead of hardcoding ``/p/<slug>``, a subresource (the case the page
#: cookie exists for), a JSON endpoint, and a WebSocket holding a real PTY child.
FIXTURE_PLUGIN = '''
"""Fixture plugin for scripts/e2e/dashboard_plugin_page.py."""
import json

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import HTMLResponse, PlainTextResponse

_children = []


def build_router():
    router = APIRouter()

    @router.get("/")
    async def index(request: Request):
        from hermes_cli.dashboard_pages import bootstrap_script, page_base_path

        base = page_base_path(request)
        return HTMLResponse(
            "<!doctype html><html><head>"
            + bootstrap_script(request)
            + f'<script src="{base}/app.js"></script>'
            + "</head><body>pane host</body></html>")

    @router.get("/app.js")
    async def app_js():
        return PlainTextResponse("/* fixture */", media_type="text/javascript")

    @router.get("/api/thing")
    async def thing():
        return {"thing": "ok"}

    @router.websocket("/ws")
    async def socket(ws: WebSocket):
        import ptyprocess

        await ws.accept()
        child = ptyprocess.PtyProcessUnicode.spawn(
            ["/bin/sh", "-c", "echo pane-ready; cat"], dimensions=(24, 80))
        _children.append(child)
        await ws.send_text(json.dumps({"type": "opened", "pane": child.pid}))
        try:
            while True:
                msg = await ws.receive_text()
                frame = json.loads(msg)
                if frame.get("type") == "kill":
                    break
                if frame.get("type") == "read":
                    await ws.send_text(json.dumps({"type": "data", "data": child.read(64)}))
        finally:
            child.terminate(force=True)

    return router


def shutdown():
    for child in _children:
        try:
            child.terminate(force=True)
        except Exception:
            pass


def register(ctx):
    ctx.register_dashboard_page(slug="e2epage", title="E2E Page", router=build_router())
'''

FIXTURE_MANIFEST = """
name: e2e-dashboard-page
version: 1.0.0
description: Fixture plugin serving a dashboard page for the e2e run.
"""


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'OK  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def _install_fixture_plugin(home: Path) -> None:
    plugin_dir = home / "plugins" / "e2e-dashboard-page"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "__init__.py").write_text(FIXTURE_PLUGIN, encoding="utf-8")
    (plugin_dir / "plugin.yaml").write_text(FIXTURE_MANIFEST, encoding="utf-8")
    # plugins.enabled is the trust gate for user plugins; without it register() never runs.
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - e2e-dashboard-page\n", encoding="utf-8")


def _open_pane(sock) -> int | None:
    """Wait for the fixture's ``opened`` frame and return the child pid."""
    deadline = time.time() + 60
    while time.time() < deadline:
        msg = sock.receive()
        if msg.get("type") == "websocket.disconnect":
            return None
        if msg.get("text"):
            frame = json.loads(msg["text"])
            if frame.get("type") == "opened":
                return frame["pane"]
    return None


def main() -> int:
    home = Path(os.environ["HERMES_HOME"])
    _install_fixture_plugin(home)

    from starlette.testclient import TestClient

    from hermes_cli.plugins import discover_plugins
    import hermes_cli.dashboard_pages as dp
    import hermes_cli.web_server as ws

    discover_plugins(force=True)

    pages = {p.slug: p for p in dp.list_pages()}
    check("plugin registered the page through real discovery", SLUG in pages,
          f"pages={sorted(pages)}")
    if SLUG not in pages:
        return 1
    check("page title carried through", pages[SLUG].title == "E2E Page", pages[SLUG].title)
    check("page attributed to the plugin", pages[SLUG].plugin == "e2e-dashboard-page",
          pages[SLUG].plugin)

    ws.app.state.bound_host = ""          # not listening: WS host guard passes in-process
    ws.app.state.auth_required = False
    client = TestClient(ws.app)
    token = ws._SESSION_TOKEN

    # --- the SPA is really mounted (the ordering hazard is real, not hypothetical) ---
    spa = client.get("/settings")
    check("SPA catch-all is live", spa.status_code == 200 and "<html" in spa.text.lower(),
          f"status={spa.status_code}")

    # --- unauthenticated ---
    check("unauthenticated page is 401", client.get(f"/p/{SLUG}/").status_code == 401)
    check("unauthenticated asset is 401", client.get(f"/p/{SLUG}/app.js").status_code == 401)
    check("unauthenticated api is 401", client.get(f"/p/{SLUG}/api/thing").status_code == 401)

    # --- authenticated page render + cookie handshake ---
    page = client.get(f"/p/{SLUG}/?token={token}")
    check("page renders", page.status_code == 200, f"status={page.status_code}")
    check("bootstrap injected", "__HERMES_PAGE_BASE__" in page.text)
    check("base path resolves to the page prefix", f'__HERMES_PAGE_BASE__="/p/{SLUG}"' in page.text,
          page.text[page.text.find("__HERMES_PAGE_BASE__"):][:60])
    check("assets rewritten to the page base", f'src="/p/{SLUG}/app.js"' in page.text)
    check("cookie issued for subresources", "set-cookie" in page.headers)

    # Now the client jar carries the cookie: subresources must load with NO token in the URL.
    check("app.js loads via cookie", client.get(f"/p/{SLUG}/app.js").status_code == 200)
    thing = client.get(f"/p/{SLUG}/api/thing")
    check("api route via cookie", thing.status_code == 200 and thing.json() == {"thing": "ok"})

    # --- slug isolation ---
    check("unknown slug 404s", client.get(f"/p/nope/?token={token}").status_code == 404)

    # --- a real PTY child over the real WebSocket ---
    try:
        with client.websocket_connect(f"/p/{SLUG}/ws?token={token}") as sock:
            pane = _open_pane(sock)
            check("websocket opened a PTY pane", bool(pane), f"pane={pane}")
            if pane:
                sock.send_text(json.dumps({"type": "read"}))
                msg = sock.receive()
                data = json.loads(msg["text"]).get("data", "") if msg.get("text") else ""
                check("pane produced terminal output", "pane-ready" in data, repr(data[:40]))
            sock.send_text(json.dumps({"type": "kill"}))
    except Exception as exc:  # noqa: BLE001 - report, don't mask
        check("websocket pane", False, repr(exc))

    # --- two concurrent sockets (a pane-per-socket plugin needs this) ---
    try:
        with client.websocket_connect(f"/p/{SLUG}/ws?token={token}") as a, \
             client.websocket_connect(f"/p/{SLUG}/ws?token={token}") as b:
            ids = [pid for pid in (_open_pane(a), _open_pane(b)) if pid]
            check("two concurrent sockets, two distinct panes",
                  len(ids) == 2 and ids[0] != ids[1], str(ids))
            for sock in (a, b):
                sock.send_text(json.dumps({"type": "kill"}))
    except Exception as exc:  # noqa: BLE001
        check("two concurrent sockets", False, repr(exc))

    # --- unload removes the route ---
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    for reg in list(manager._ownership_ledger.get("e2e-dashboard-page", [])):
        if reg.kind == "dashboard_page":
            reg.dispose()
    check("unload removes the page", dp.get_page(SLUG) is None)
    check("route is gone after unload",
          client.get(f"/p/{SLUG}/?token={token}").status_code == 404)

    # --- clean up any PTY child the run left behind ---
    for mod in list(sys.modules.values()):
        shutdown = getattr(mod, "shutdown", None) if mod else None
        if shutdown and getattr(mod, "__name__", "").endswith("e2e-dashboard-page"):
            shutdown()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("All dashboard plugin page end-to-end checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
