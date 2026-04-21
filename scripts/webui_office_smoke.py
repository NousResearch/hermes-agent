"""Merged hermes-webui + digital office smoke test.

Hits both the classic dashboard endpoints and the new /api/office/* router,
plus the /ws/office WebSocket, to confirm the integration is healthy.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

# Gateway API server (``hermes gateway`` with API_SERVER_ENABLED) — default 8642.
# Legacy hermes-webui on 8643: set HERMES_GATEWAY_OFFICE_SMOKE_URL=http://127.0.0.1:8643
BASE = os.environ.get("HERMES_GATEWAY_OFFICE_SMOKE_URL", "http://127.0.0.1:8642")
_API_KEY = os.environ.get("API_SERVER_KEY", "").strip()


def _req(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if _API_KEY:
        headers["Authorization"] = f"Bearer {_API_KEY}"
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{BASE}{path}", data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            payload = r.read()
            try:
                return r.status, json.loads(payload) if payload else None
            except json.JSONDecodeError:
                return r.status, payload.decode(errors="replace")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, None


def _assert(cond: bool, msg: str) -> None:
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        globals()["_failures"] += 1


_failures = 0


def phase(title: str) -> None:
    print(f"\n── {title} ──")


async def ws_smoke() -> bool:
    try:
        import websockets  # type: ignore
    except ImportError:
        print("  [SKIP] websockets not installed; skipping WS test")
        return True

    from urllib.parse import urlparse

    u = urlparse(BASE)
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if u.scheme == "https" else 80)
    scheme = "wss" if u.scheme == "https" else "ws"
    url = f"{scheme}://{host}:{port}/ws/office"
    try:
        kw: dict = {"open_timeout": 5}
        if _API_KEY:
            kw["additional_headers"] = [("Authorization", f"Bearer {_API_KEY}")]
        async with websockets.connect(url, **kw) as ws:
            await ws.send(json.dumps({"type": "auth", "token": "ignored"}))
            first = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(first)
            return msg.get("kind") == "hello"
    except Exception as exc:
        print(f"  ws error: {exc!r}")
        return False


def main() -> int:
    global _failures

    try:
        ping = urllib.request.Request(
            f"{BASE}/api/health",
            method="GET",
            headers={"Accept": "application/json"},
        )
        urllib.request.urlopen(ping, timeout=5)
    except urllib.error.URLError as exc:
        print(f"\n[smoke] cannot reach {BASE!r}: {exc}", file=sys.stderr)
        print(
            "  Start the user-facing API server:  hermes gateway  "
            "(set API_SERVER_ENABLED=true in ~/.hermes/.env), then re-run.",
            file=sys.stderr,
        )
        print(
            "  Offline equivalent:  pytest tests/gateway/test_embedded_office_e2e.py -v",
            file=sys.stderr,
        )
        return 2

    phase("classic webui endpoints")
    s, body = _req("GET", "/api/health")
    _assert(s == 200 and body.get("status") == "ok", f"GET /api/health -> {s}")

    phase("office: meta")
    s, body = _req("GET", "/api/office/health")
    _assert(s == 200 and body.get("ok"), f"GET /api/office/health -> {s}")
    initial_counts = body

    s, body = _req("GET", "/api/office/capacity")
    _assert(s == 200 and "recommended_concurrency" in body, f"GET /api/office/capacity -> {s}")

    s, body = _req("GET", "/api/office/toolsets")
    _assert(s == 200 and isinstance(body, list) and len(body) > 5, f"GET /api/office/toolsets -> {s} ({len(body) if isinstance(body, list) else 0} items)")

    s, body = _req("GET", "/api/office/presets")
    _assert(s == 200 and isinstance(body, list) and body, f"GET /api/office/presets -> {s}")

    s, body = _req("POST", "/api/office/skills/resolve", {"text": "Write Python unit tests and fix flaky CI"})
    ok = (
        s == 200
        and isinstance(body, dict)
        and isinstance(body.get("recommended_toolsets"), list)
        and body.get("recommended_toolsets")
        and 0.0 <= float(body.get("confidence", 0)) <= 1.0
    )
    _assert(ok, f"POST /api/office/skills/resolve -> {s} (toolsets={body.get('recommended_toolsets')})")

    phase("office: department + employee CRUD")
    dept_name = "Smoke Test Dept"
    s, dept = _req("POST", "/api/office/departments", {"name": dept_name, "mission": "webui smoke", "color": "#14b8a6"})
    _assert(s == 201 and dept.get("id"), f"POST /api/office/departments -> {s}")
    dept_id = dept["id"]

    s, emp = _req("POST", "/api/office/employees", {
        "department_id": dept_id,
        "name": "Smokey",
        "role": "Tester",
        "model": "gemma4-e2b-hermes",
        "enabled_toolsets": ["web", "code_execution"],
        "skills": [],
        "runtime": "simulated",
    })
    _assert(s == 201 and emp.get("id"), f"POST /api/office/employees -> {s}")
    emp_id = emp["id"]

    s, got = _req("GET", f"/api/office/employees/{emp_id}")
    _assert(s == 200 and got.get("cli_command", "").startswith("hermes chat"), f"GET employee + cli_command -> {s}")

    phase("office: task routing")
    s, task = _req("POST", "/api/office/tasks", {"text": "Smoke task for integration", "employee_id": emp_id})
    _assert(s == 201 and task.get("id"), f"POST /api/office/tasks -> {s}")
    task_id = task["id"]

    # Wait for simulated runtime to finish (~2 s)
    import time
    for _ in range(30):
        time.sleep(0.25)
        s, rows = _req("GET", "/api/office/tasks?limit=20")
        row = next((r for r in rows if r.get("id") == task_id), None)
        if row and row.get("status") in ("done", "failed"):
            break
    _assert(
        row and row.get("status") == "done",
        f"simulated task reached 'done' (got {row.get('status') if row else 'missing'})",
    )

    phase("office: cleanup")
    s, body = _req("DELETE", f"/api/office/employees/{emp_id}")
    _assert(s == 200, f"DELETE employee -> {s}")
    s, body = _req("DELETE", f"/api/office/departments/{dept_id}")
    _assert(s == 200, f"DELETE department -> {s}")

    phase("websocket /ws/office")
    ok = asyncio.run(ws_smoke())
    _assert(ok, "ws/office hello received")

    phase("static frontend")
    s, body = _req("GET", "/office")
    _assert(s == 200 and "<div id=\"root\"" in (body if isinstance(body, str) else body.decode()), f"GET /office (SPA) -> {s}")

    print(f"\n── summary: {_failures} failure(s) ──")
    return 0 if _failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
