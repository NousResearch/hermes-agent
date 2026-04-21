"""In-process E2E for Digital Office embedded on the Gateway API server (aiohttp).

Mirrors ``scripts/webui_office_smoke.py`` without requiring ``hermes gateway`` on 8642.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("aiohttp", reason="aiohttp not installed")
pytest.importorskip("httpx", reason="httpx not installed")


@pytest.mark.asyncio
async def test_embedded_office_full_smoke(monkeypatch, tmp_path):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from hermes_office import gateway_http as gh
    from hermes_office.gateway_http import register_digital_office_routes

    gh.get_office_app.cache_clear()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()

    app = web.Application()
    assert register_digital_office_routes(app) is True

    try:
        async with TestClient(TestServer(app)) as client:
            async with client.get("/api/health") as r:
                assert r.status == 200
                assert (await r.json()).get("status") == "ok"

            async with client.get("/api/office/health") as r:
                assert r.status == 200
                body = await r.json()
                assert body.get("ok") is True

            async with client.get("/api/office/capacity") as r:
                assert r.status == 200
                assert "recommended_concurrency" in await r.json()

            async with client.get("/api/office/toolsets") as r:
                assert r.status == 200
                toolsets = await r.json()
                assert isinstance(toolsets, list) and len(toolsets) > 5

            async with client.get("/api/office/presets") as r:
                assert r.status == 200
                presets = await r.json()
                assert isinstance(presets, list) and presets

            async with client.post(
                "/api/office/skills/resolve",
                json={"text": "Write Python unit tests and fix flaky CI"},
            ) as r:
                assert r.status == 200
                resolved = await r.json()
                assert isinstance(resolved.get("recommended_toolsets"), list)
                assert resolved.get("recommended_toolsets")
                assert 0.0 <= float(resolved.get("confidence", 0)) <= 1.0

            dept_name = "E2E Smoke Dept"
            async with client.post(
                "/api/office/departments",
                json={
                    "name": dept_name,
                    "mission": "e2e",
                    "color": "#14b8a6",
                },
            ) as r:
                assert r.status == 201
                dept = await r.json()
            dept_id = dept["id"]

            async with client.post(
                "/api/office/employees",
                json={
                    "department_id": dept_id,
                    "name": "Smokey",
                    "role": "Tester",
                    "model": "gemma4-e2b-hermes",
                    "enabled_toolsets": ["web", "code_execution"],
                    "skills": [],
                    "runtime": "simulated",
                },
            ) as r:
                assert r.status == 201
                emp = await r.json()
            emp_id = emp["id"]

            async with client.get(f"/api/office/employees/{emp_id}") as r:
                assert r.status == 200
                got = await r.json()
                assert got.get("cli_command", "").startswith("hermes chat")

            async with client.post(
                "/api/office/tasks",
                json={"text": "E2E smoke task", "employee_id": emp_id},
            ) as r:
                assert r.status == 201
                task = await r.json()
            task_id = task["id"]

            row = None
            for _ in range(30):
                await asyncio.sleep(0.25)
                async with client.get("/api/office/tasks?limit=20") as r:
                    assert r.status == 200
                    rows = await r.json()
                row = next((x for x in rows if x.get("id") == task_id), None)
                if row and row.get("status") in ("done", "failed"):
                    break
            assert row and row.get("status") == "done"

            async with client.delete(f"/api/office/employees/{emp_id}") as r:
                assert r.status == 200
            async with client.delete(f"/api/office/departments/{dept_id}") as r:
                assert r.status == 200

            ws = await client.ws_connect("/ws/office")
            try:
                first = await asyncio.wait_for(ws.receive_json(), timeout=5)
                assert first.get("kind") == "hello"
            finally:
                await ws.close()

            async with client.get("/office", allow_redirects=True) as r:
                if r.status == 503:
                    pytest.skip("office frontend dist not built")
                assert r.status == 200
                text = await r.text()
                assert 'id="root"' in text
    finally:
        gh.get_office_app.cache_clear()
