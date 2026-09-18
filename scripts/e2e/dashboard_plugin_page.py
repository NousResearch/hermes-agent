"""WB-618 end-to-end: the REAL dashboard app, the REAL plugin discovery path, the REAL page.

Not a unit test with a fake router — this boots hermes_cli.web_server (SPA build included),
runs plugin discovery so a plugin registers its page through ``ctx.register_dashboard_page``,
then drives /p/<slug> over HTTP and WebSocket and spawns an actual PTY pane.

Lives in scripts/e2e/ rather than tests/: it needs a real plugin checkout and a built
``hermes_cli/web_dist`` (without the SPA the catch-all check cannot detect the mount-order
hazard it exists to catch), so it is run deliberately, not by the pytest sweep.

Run:
    HERMES_HOME=<temp> WB618_PLUGIN_DIR=<plugin checkout> \\
        .venv/bin/python scripts/e2e/dashboard_plugin_page.py
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


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'OK  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def main() -> int:
    plugin_src = Path(os.environ["WB618_PLUGIN_DIR"])
    home = Path(os.environ["HERMES_HOME"])
    (home / "plugins").mkdir(parents=True, exist_ok=True)
    link = home / "plugins" / "hermes-i3"
    if not link.exists():
        link.symlink_to(plugin_src)
    # plugins.enabled is the trust gate for user plugins; without it register() never runs.
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - hermes-i3\n", encoding="utf-8")

    from starlette.testclient import TestClient

    from hermes_cli.plugins import discover_plugins
    import hermes_cli.dashboard_pages as dp
    import hermes_cli.web_server as ws

    discover_plugins(force=True)

    pages = {p.slug: p for p in dp.list_pages()}
    check("plugin registered /p/i3 through real discovery", "i3" in pages,
          f"pages={sorted(pages)}")
    if "i3" not in pages:
        return 1
    check("page title carried through", pages["i3"].title == "Terminals", pages["i3"].title)
    check("page attributed to the plugin", pages["i3"].plugin == "hermes-i3",
          pages["i3"].plugin)

    ws.app.state.bound_host = ""          # not listening: WS host guard passes in-process
    ws.app.state.auth_required = False
    client = TestClient(ws.app)
    token = ws._SESSION_TOKEN

    # --- the SPA is really mounted (the ordering hazard is real, not hypothetical) ---
    spa = client.get("/settings")
    check("SPA catch-all is live", spa.status_code == 200 and "<html" in spa.text.lower(),
          f"status={spa.status_code}")

    # --- unauthenticated ---
    check("unauthenticated /p/i3 is 401", client.get("/p/i3/").status_code == 401)
    check("unauthenticated asset is 401", client.get("/p/i3/app.js").status_code == 401)
    check("unauthenticated api is 401", client.get("/p/i3/api/panes").status_code == 401)

    # --- authenticated page render + cookie handshake ---
    page = client.get(f"/p/i3/?token={token}")
    check("page renders", page.status_code == 200, f"status={page.status_code}")
    check("bootstrap injected", "__HERMES_PAGE_BASE__" in page.text)
    check("base path is /p/i3", '__HERMES_PAGE_BASE__="/p/i3"' in page.text,
          page.text[page.text.find("__HERMES_PAGE_BASE__"):][:60])
    check("assets rewritten to the page base", 'src="/p/i3/app.js"' in page.text)
    check("cookie issued for subresources", "set-cookie" in page.headers)

    # Now the client jar carries the cookie: assets must load with NO token in the URL.
    check("app.js loads via cookie", client.get("/p/i3/app.js").status_code == 200)
    check("app.css loads via cookie", client.get("/p/i3/app.css").status_code == 200)
    vendor = client.get("/p/i3/vendor/xterm.mjs")
    check("xterm vendored", vendor.status_code == 200, f"status={vendor.status_code}")

    panes = client.get("/p/i3/api/panes")
    check("api/panes via cookie", panes.status_code == 200 and "panes" in panes.json())
    profiles = client.get("/p/i3/api/profiles")
    check("api/profiles via cookie", profiles.status_code == 200)

    # --- slug isolation ---
    check("unknown slug 404s", client.get(f"/p/nope/?token={token}").status_code == 404)

    # --- the real PTY pane over the real WebSocket ---
    pane_id = None
    output = b""
    try:
        with client.websocket_connect(f"/p/i3/ws?token={token}") as sock:
            sock.send_text(json.dumps({"type": "open", "cols": 100, "rows": 30}))
            deadline = time.time() + 120
            while time.time() < deadline:
                msg = sock.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text"):
                    frame = json.loads(msg["text"])
                    if frame.get("type") == "opened":
                        pane_id = frame["pane"]
                    elif frame.get("type") in ("error", "exit"):
                        print("   pane frame:", frame)
                        break
                elif msg.get("bytes"):
                    output += msg["bytes"]
                    if len(output) > 2000:
                        break
            if pane_id:
                sock.send_text(json.dumps({"type": "resize", "cols": 120, "rows": 40}))
                sock.send_text(json.dumps({"type": "kill"}))
    except Exception as exc:  # noqa: BLE001 - report, don't mask
        check("websocket pane", False, repr(exc))
    else:
        check("websocket opened a pane", bool(pane_id), f"pane={pane_id}")
        check("pane produced terminal output", len(output) > 200, f"{len(output)} bytes")

    # --- two concurrent sockets (i3 needs one per pane) ---
    try:
        with client.websocket_connect(f"/p/i3/ws?token={token}") as a, \
             client.websocket_connect(f"/p/i3/ws?token={token}") as b:
            a.send_text(json.dumps({"type": "open", "cols": 80, "rows": 24}))
            b.send_text(json.dumps({"type": "open", "cols": 80, "rows": 24}))
            ids = []
            for sock in (a, b):
                deadline = time.time() + 120
                while time.time() < deadline:
                    msg = sock.receive()
                    if msg.get("text"):
                        frame = json.loads(msg["text"])
                        if frame.get("type") == "opened":
                            ids.append(frame["pane"])
                            break
                        if frame.get("type") in ("error", "exit"):
                            break
                    elif msg.get("type") == "websocket.disconnect":
                        break
            check("two concurrent sockets, two distinct panes",
                  len(ids) == 2 and ids[0] != ids[1], str(ids))
            for sock in (a, b):
                sock.send_text(json.dumps({"type": "kill"}))
    except Exception as exc:  # noqa: BLE001
        check("two concurrent sockets", False, repr(exc))

    # --- unload removes the route ---
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    for reg in list(manager._ownership_ledger.get("hermes-i3", [])):
        if reg.kind == "dashboard_page":
            reg.dispose()
    check("unload removes the page", dp.get_page("i3") is None)
    check("route is gone after unload",
          client.get(f"/p/i3/?token={token}").status_code == 404)

    # --- clean up any pane the run left behind ---
    page_mod = sys.modules.get("hermes_plugins.hermes-i3.page")
    try:
        from hermes_plugins import __dict__ as _  # noqa: F401
    except Exception:
        pass
    for mod in list(sys.modules.values()):
        inst = getattr(mod, "_page", None) if mod else None
        if inst is not None and hasattr(inst, "panes"):
            inst.shutdown()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("All WB-618 end-to-end checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
