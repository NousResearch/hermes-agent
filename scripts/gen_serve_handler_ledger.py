#!/usr/bin/env -S bash -c 'exec "$BASH" "$(dirname "$0")/run-in-hermes-env" python3 "$0" "$@"'
"""Regenerate website/docs/developer-guide/serve-handler-ledger.json from the code.

One row per handler ``hermes serve`` exposes: every HTTP/WebSocket route of the shared FastAPI app
(``hermes_cli.web_server.app``, imported the way ``serve`` imports it, ``HERMES_SERVE_HEADLESS=1``)
and every JSON-RPC method a ``/api/ws`` client can call (the legacy ``tui_gateway.server._methods``
table plus the verbs the session authority serves). Dashboard-plugin API routes depend on which
plugins are installed and enabled, so they are one row: the ``/api/plugins/{name}/{path}`` mount.

Columns are derived from code, never judged:

* RPC ``current_path`` uses the tables ``/api/ws`` dispatch reads (``tui_gateway/ws.py``): served by
  the authority (``AuthorityConnection.handlers`` + group verbs + ``profiles.list``), else legacy
  fallback or refused per ``tui_gateway/ws_legacy_fallback.py``. Without an authority (standalone
  ``serve``/``dashboard``) every legacy method runs in-process on ``server.dispatch``.
* HTTP ``current_path`` is the endpoint plus any ``route_on`` surface gate
  (``hermes_cli/web_server_surface.py``). Whether an endpoint reaches the authority is not derived.
* ``decision`` is ``authority`` for a verb the authority serves; everything else is ``undecided``
  until a reviewed entry lands in ``DECISIONS`` below. ``ref`` names the open #145171 decision a
  row waits on.

``--check`` writes nothing and exits 1 on drift (``tests/hermes_cli/test_serve_handler_ledger.py``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "website" / "docs" / "developer-guide" / "serve-handler-ledger.json"
DECISION_VALUES = ("authority", "keep-host-wide", "delete", "undecided")
# Reviewed owner decisions, keyed by row ``handler``: ``(decision, ref)``. A row not listed here
# keeps its mechanical default.
DECISIONS: dict[str, tuple[str, str]] = {
    # Creating a profile is host-level work, not session work.
    "POST /api/profiles": ("keep-host-wide", ""),
    "profiles.create": ("keep-host-wide", ""),
}
_PLUGIN_MODULE_PREFIX = "hermes_dashboard_plugin_"


def _surface_gates(route) -> list[str]:
    """``route_on(field)`` dependencies: the route 404s on surfaces whose policy leaves *field* off."""
    gates = []
    for dep in getattr(getattr(route, "dependant", None), "dependencies", ()):
        call = getattr(dep, "call", None)
        if call is not None and call.__name__ == "surface_serves" and call.__closure__:
            gates.append(str(call.__closure__[0].cell_contents))
    return sorted(gates)


def _collect() -> dict:
    """Raw facts from a fresh import (run in a child process: see :func:`_facts`)."""
    sys.path.insert(0, str(ROOT))
    from unittest.mock import MagicMock

    from fastapi.routing import APIRoute, APIWebSocketRoute
    from starlette.routing import Mount

    from gateway.session_controls import AuthorityConnection
    from gateway.session_group_controls import GROUP_METHODS
    from hermes_cli.web_server import app
    from tui_gateway import server
    from tui_gateway.ws_legacy_fallback import _PROCESS_GLOBAL, _SESSION_FALLBACK_ALLOWED, _SESSION_NAMESPACES

    routes = []
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None) or getattr(route, "app", None)
        module = getattr(endpoint, "__module__", "") or ""
        if module.startswith(_PLUGIN_MODULE_PREFIX):
            continue
        name = getattr(endpoint, "__qualname__", None) or type(endpoint).__qualname__
        if isinstance(route, APIWebSocketRoute):
            kind, methods = "websocket", ["WS"]
        elif isinstance(route, Mount):
            kind, methods = "mount", ["*"]
        else:
            kind, methods = "http", sorted(getattr(route, "methods", None) or ["*"])
        routes.append({"path": route.path, "kind": kind, "methods": methods, "source": f"{module}:{name}",
                       "gates": _surface_gates(route) if isinstance(route, (APIRoute, APIWebSocketRoute)) else []})
    return {
        "routes": routes,
        "legacy": sorted(server._methods),
        "served": sorted(set(AuthorityConnection.handlers(MagicMock())) | set(GROUP_METHODS) | {"profiles.list"}),
        "process_global": sorted(_PROCESS_GLOBAL),
        "session_namespaces": list(_SESSION_NAMESPACES),
        "session_fallback_allowed": sorted(_SESSION_FALLBACK_ALLOWED),
    }


def _facts() -> dict:
    """:func:`_collect` in a child interpreter with ``serve``'s import environment and a throwaway
    HERMES_HOME, so the route set never depends on the caller's process state or real profile."""
    with tempfile.TemporaryDirectory(prefix="serve-ledger-") as scratch:
        # The facts go to a file: importing tui_gateway.server takes over stdout for its JSON-RPC.
        home, out = Path(scratch) / "home", Path(scratch) / "facts.json"
        home.mkdir()
        env = {**os.environ, "HERMES_HOME": str(home), "HERMES_SERVE_HEADLESS": "1"}
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--collect", str(out)], cwd=ROOT,
                              env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False,
                              timeout=300)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"handler collection failed:\n{proc.stderr[-4000:]}")
        return json.loads(out.read_text(encoding="utf-8-sig"))


def _rpc_row(method: str, facts: dict) -> dict:
    legacy, served = method in facts["legacy"], method in facts["served"]
    standalone = "; standalone: legacy server.dispatch" if legacy else "; standalone: unknown method"
    if served:
        return {"current_path": "authority" + standalone, "decision": "authority", "ref": ""}
    refused = method in facts["process_global"] or (
        method.startswith(tuple(facts["session_namespaces"])) and method not in facts["session_fallback_allowed"])
    path = "authority refuses (-32601)" if refused else "legacy fallback"
    return {"current_path": path + standalone, "decision": "undecided", "ref": "D3"}


def _http_row(route: dict) -> dict:
    path = route["source"] + ("; surface: " + ", ".join(route["gates"]) if route["gates"] else "")
    return {"current_path": path, "decision": "undecided", "ref": ""}


def rows(facts: dict) -> list[dict]:
    out = [{"handler": f"{' '.join(route['methods'])} {route['path']}", "kind": route["kind"], **_http_row(route)}
           for route in facts["routes"]]
    out += [{"handler": method, "kind": "rpc", **_rpc_row(method, facts)}
            for method in sorted(set(facts["legacy"]) | set(facts["served"]))]
    for row in out:
        if row["handler"] in DECISIONS:
            row["decision"], row["ref"] = DECISIONS[row["handler"]]
    return sorted(out, key=lambda row: (row["kind"] == "rpc", row["handler"]))


def render(facts: dict | None = None) -> str:
    table = rows(facts or _facts())
    counts: dict[str, dict[str, int]] = {}
    for row in table:
        bucket = counts.setdefault(row["kind"], {})
        bucket[row["decision"]] = bucket.get(row["decision"], 0) + 1
    head = json.dumps({"generated_by": "scripts/gen_serve_handler_ledger.py", "refs": "#145171 (EP-1)",
                       "decisions": list(DECISION_VALUES), "counts": counts}, indent=1, sort_keys=True)
    return (head[:-2] + ',\n "handlers": [\n' + ",\n".join("  " + json.dumps(row, sort_keys=True) for row in table)
            + "\n ]\n}\n")


def check(out: Path = OUT) -> int:
    committed = out.read_text(encoding="utf-8-sig") if out.exists() else ""
    if committed == render():
        return 0
    print(f"{out.relative_to(ROOT)} is stale; run scripts/gen_serve_handler_ledger.py", file=sys.stderr)
    return 1


def main(argv) -> int:
    if "--collect" in argv:
        Path(argv[argv.index("--collect") + 1]).write_text(json.dumps(_collect()), encoding="utf-8")
        return 0
    if "--check" in argv:
        return check()
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
