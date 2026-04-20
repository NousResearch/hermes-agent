#!/usr/bin/env python3
"""End-to-end smoke test for the running Hermes Digital Office server.

Exercises every documented API endpoint plus the live WebSocket against
http://127.0.0.1:8765/ (overridable). Idempotent: every artifact it creates is
also deleted at the end.

Exit code is the number of failed checks (0 == all green).

Usage:
    python scripts/office_e2e_smoke.py
    python scripts/office_e2e_smoke.py --base http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE = "http://127.0.0.1:8765"

PASS = "\u2713"
FAIL = "\u2717"


class Client:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

    def _req(
        self,
        method: str,
        path: str,
        body: Any | None = None,
        expect: tuple[int, ...] = (200, 201),
    ) -> tuple[int, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
                payload: Any = None
                if raw:
                    try:
                        payload = json.loads(raw.decode("utf-8"))
                    except json.JSONDecodeError:
                        payload = raw.decode("utf-8", errors="replace")
                return resp.status, payload
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(payload)
            except Exception:
                pass
            return exc.code, payload

    def get(self, path: str, expect=(200,)) -> tuple[int, Any]:
        return self._req("GET", path, expect=expect)

    def post(self, path: str, body: dict, expect=(200, 201)) -> tuple[int, Any]:
        return self._req("POST", path, body=body, expect=expect)

    def patch(self, path: str, body: dict, expect=(200,)) -> tuple[int, Any]:
        return self._req("PATCH", path, body=body, expect=expect)

    def delete(self, path: str, expect=(200,)) -> tuple[int, Any]:
        return self._req("DELETE", path, expect=expect)


# ─────────────────────────────────────────────────────────────────────────────


class Report:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def ok(self, name: str) -> None:
        self.passed.append(name)
        print(f"  {PASS} {name}")

    def fail(self, name: str, detail: str) -> None:
        self.failed.append((name, detail))
        print(f"  {FAIL} {name}\n      → {detail}")

    def section(self, title: str) -> None:
        print(f"\n— {title} —")

    def summary(self) -> int:
        total = len(self.passed) + len(self.failed)
        print()
        print("=" * 64)
        print(f" PASS {len(self.passed)} / {total}    FAIL {len(self.failed)}")
        if self.failed:
            print(" Failures:")
            for n, d in self.failed:
                print(f"   - {n}: {d}")
        print("=" * 64)
        return len(self.failed)


# ─────────────────────────────────────────────────────────────────────────────


def expect(report: Report, name: str, cond: bool, detail: str = "") -> bool:
    if cond:
        report.ok(name)
        return True
    report.fail(name, detail or "condition false")
    return False


# ─────────────────────────────────────────────────────────────────────────────


async def ws_smoke(base: str, report: Report) -> None:
    """Connect, get hello + at least one event, then close cleanly."""
    try:
        import websockets  # type: ignore
    except ImportError:
        report.fail("WebSocket: websockets lib", "pip install websockets")
        return

    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + "/ws/office"
    try:
        async with websockets.connect(ws_url, open_timeout=5, close_timeout=2) as ws:
            try:
                hello_raw = await asyncio.wait_for(ws.recv(), timeout=3)
            except asyncio.TimeoutError:
                report.fail("WebSocket: hello", "no hello frame within 3s")
                return
            try:
                hello = json.loads(hello_raw)
            except Exception as exc:
                report.fail("WebSocket: hello json", repr(exc))
                return
            expect(report, "WebSocket: hello frame", hello.get("kind") == "hello",
                   f"got {hello!r}")
            # Trigger an event so we can validate live event push.
            client = Client(base)
            _, depts = client.get("/api/departments")
            target_emp = None
            for d in depts or []:
                emps = client.get(f"/api/employees?dept_id={d['id']}")[1] or []
                if emps:
                    target_emp = emps[0]
                    break
            if not target_emp:
                report.fail("WebSocket: trigger task", "no employee available to dispatch a task")
                return
            client.post("/api/tasks", {"employee_id": target_emp["id"], "text": "ws ping"})
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    data = json.loads(msg)
                    if data.get("kind") and data.get("kind") != "hello":
                        report.ok(f"WebSocket: live event ({data['kind']})")
                        return
            except asyncio.TimeoutError:
                report.fail("WebSocket: live event", "no non-hello event within 10s")
    except Exception as exc:
        report.fail("WebSocket: connect", repr(exc))


# ─────────────────────────────────────────────────────────────────────────────


def run(base: str) -> int:
    print(f"Hermes Digital Office e2e smoke against {base}\n")
    c = Client(base)
    r = Report()

    # ── 1. Health & meta ────────────────────────────────────────────────────
    r.section("health & meta")
    code, body = c.get("/api/health")
    expect(r, "GET /api/health 200", code == 200, f"code={code}")
    expect(r, "health.ok=true", isinstance(body, dict) and body.get("ok") is True, str(body))
    expect(r, "health.version present", bool(body.get("version")))

    code, cap = c.get("/api/capacity")
    expect(r, "GET /api/capacity 200", code == 200, f"code={code}")
    expect(r, "capacity.host", isinstance(cap, dict) and "host" in cap, str(cap)[:120])
    expect(r, "capacity.recommended_concurrency >= 1",
           isinstance(cap, dict) and cap.get("recommended_concurrency", 0) >= 1)

    code, ts = c.get("/api/toolsets")
    expect(r, "GET /api/toolsets 200", code == 200, f"code={code}")
    expect(r, "toolsets non-empty list", isinstance(ts, list) and len(ts) > 5, f"len={len(ts) if isinstance(ts, list) else 'NA'}")

    code, sk = c.get("/api/skills")
    expect(r, "GET /api/skills 200", code == 200, f"code={code}")
    expect(r, "skills returns list", isinstance(sk, list))

    code, ps = c.get("/api/presets")
    expect(r, "GET /api/presets 200", code == 200, f"code={code}")
    expect(r, "presets non-empty", isinstance(ps, list) and len(ps) > 0)

    # ── 2. Skill resolver ───────────────────────────────────────────────────
    r.section("skill resolver")
    code, role = c.post("/api/skills/resolve", {"text": "research arxiv papers and write summaries"})
    expect(r, "POST /api/skills/resolve 200", code == 200, f"code={code}")
    expect(r, "resolved.recommended_toolsets is list",
           isinstance(role, dict) and isinstance(role.get("recommended_toolsets"), list),
           f"got {role!r}")
    expect(r, "resolver picked at least one toolset for research prompt",
           isinstance(role, dict) and len(role.get("recommended_toolsets", [])) > 0,
           f"got {role!r}")

    # ── 3. Departments CRUD ─────────────────────────────────────────────────
    r.section("departments CRUD")
    suffix = f"e2e-{int(time.time())}"
    dept_name = f"E2E Dept {suffix}"
    code, dept = c.post("/api/departments",
                        {"name": dept_name, "mission": "smoke", "color": "#888888"})
    expect(r, "POST /api/departments 201", code == 201, f"code={code} body={dept}")
    dept_id = (dept or {}).get("id", "")
    expect(r, "dept id format", dept_id.startswith("dept_"), dept_id)

    code, plist = c.get("/api/departments")
    expect(r, "GET /api/departments includes new", any(d.get("id") == dept_id for d in (plist or [])))

    code, patched = c.patch(f"/api/departments/{dept_id}", {"mission": "smoke (patched)"})
    expect(r, "PATCH /api/departments 200", code == 200, f"code={code}")
    expect(r, "patched mission applied", isinstance(patched, dict) and patched.get("mission") == "smoke (patched)")

    # Negative: delete a missing dept should 404.
    code, _ = c.delete("/api/departments/dept_doesnotexist")
    expect(r, "DELETE missing dept -> 404", code == 404, f"got {code}")

    # ── 4. Employees CRUD + activity ────────────────────────────────────────
    r.section("employees CRUD")
    emp_payload = {
        "department_id": dept_id,
        "name": f"E2E Bot {suffix}",
        "role": "Tester",
        "avatar": {"sprite_id": "robot-1", "hue": 200},
        "model": "gemma4-e2b-hermes",
        "enabled_toolsets": ["file", "todo"],
        "skills": [],
        "system_prompt": "You are a smoke-test bot.",
        "runtime": "simulated",
    }
    code, emp = c.post("/api/employees", emp_payload)
    expect(r, "POST /api/employees 201", code == 201, f"code={code} body={emp}")
    emp_id = (emp or {}).get("id", "")
    expect(r, "emp id format", emp_id.startswith("emp_"), emp_id)

    # Negative: missing department.
    bad = dict(emp_payload, department_id="dept_doesnotexist")
    code, body = c.post("/api/employees", bad, expect=(404, 422))
    expect(r, "POST employee with bad dept -> 4xx", code in (404, 422), f"got {code}")

    # Negative: validation error on bad sprite hue.
    bad2 = dict(emp_payload, avatar={"sprite_id": "robot-1", "hue": 9999})
    code, body = c.post("/api/employees", bad2, expect=(422,))
    expect(r, "POST employee bad hue -> 422", code == 422, f"got {code}")

    # GET single employee and verify cli_command is computed.
    code, single = c.get(f"/api/employees/{emp_id}")
    expect(r, "GET employee includes cli_command",
           isinstance(single, dict) and "cli_command" in single)

    # PATCH employee.
    code, patched = c.patch(f"/api/employees/{emp_id}",
                            {"role": "Senior Tester", "system_prompt": "Patched."})
    expect(r, "PATCH employee 200", code == 200, f"code={code}")
    expect(r, "PATCH role applied",
           isinstance(patched, dict) and patched.get("role") == "Senior Tester")

    # cli-command endpoint.
    code, cli = c.get(f"/api/employees/{emp_id}/cli-command")
    expect(r, "GET cli-command 200", code == 200)
    expect(r, "cli-command starts with hermes chat",
           isinstance(cli, dict) and cli.get("command", "").startswith("hermes chat"))

    # ── 5. Tasks routing ────────────────────────────────────────────────────
    r.section("tasks routing")
    code, _ = c.post("/api/tasks", {"text": "no target"}, expect=(422,))
    expect(r, "POST task w/o target -> 422", code == 422, f"got {code}")

    code, t1 = c.post("/api/tasks",
                      {"employee_id": emp_id, "text": "smoke task A"})
    expect(r, "POST task by employee 201", code == 201, f"code={code}")
    expect(r, "task.id format", isinstance(t1, dict) and (t1.get("id", "")).startswith("task_"))

    code, t2 = c.post("/api/tasks",
                      {"department_id": dept_id, "text": "smoke task B"})
    expect(r, "POST task by department 201", code == 201, f"code={code}")

    code, tasks = c.get("/api/tasks?limit=20")
    expect(r, "GET /api/tasks 200", code == 200)
    expect(r, "tasks list contains both",
           isinstance(tasks, list)
           and any(t.get("id") == t1.get("id") for t in tasks)
           and any(t.get("id") == t2.get("id") for t in tasks))

    # Wait for simulated runtime to flush activity.
    time.sleep(7)

    code, act = c.get(f"/api/employees/{emp_id}/activity?limit=50")
    expect(r, "GET activity 200", code == 200)
    expect(r, "activity has events",
           isinstance(act, dict) and isinstance(act.get("events"), list) and len(act["events"]) > 0,
           f"events={(act or {}).get('events')!r}")

    # Verify task transitioned to terminal state.
    code, tasks_after = c.get("/api/tasks?limit=50")
    last = next((t for t in tasks_after if t.get("id") == t1.get("id")), {})
    expect(r, "task A reached terminal state",
           last.get("status") in ("done", "failed"),
           f"status={last.get('status')!r}")

    # ── 6. Export / import round-trip ───────────────────────────────────────
    r.section("export / import")
    code, exp = c.get("/api/export")
    expect(r, "GET /api/export 200", code == 200)
    expect(r, "export has departments + employees",
           isinstance(exp, dict)
           and isinstance(exp.get("departments"), list)
           and isinstance(exp.get("employees"), list))
    code, imp = c.post("/api/import", exp)
    expect(r, "POST /api/import 200", code == 200, f"code={code} body={str(imp)[:120]}")

    # ── 7. WebSocket ────────────────────────────────────────────────────────
    r.section("websocket")
    asyncio.run(ws_smoke(base, r))

    # ── 8. Static frontend ──────────────────────────────────────────────────
    r.section("static frontend")
    code, html = c.get("/")
    expect(r, "GET / 200", code == 200)
    expect(r, "/ returns html", isinstance(html, str) and "<!doctype html" in html.lower())

    # ── 9. Cleanup ──────────────────────────────────────────────────────────
    r.section("cleanup")
    code, _ = c.delete(f"/api/employees/{emp_id}")
    expect(r, "DELETE employee 200", code == 200)
    code, _ = c.delete(f"/api/departments/{dept_id}")
    expect(r, "DELETE department 200", code == 200)

    return r.summary()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=DEFAULT_BASE)
    args = p.parse_args()
    return run(args.base)


if __name__ == "__main__":
    raise SystemExit(main())
