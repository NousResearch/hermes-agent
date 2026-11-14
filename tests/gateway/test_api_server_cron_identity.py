"""Gateway clients can link cron sessions to their job and execution without dashboard access."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from cron import jobs
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tests.fakes.cron_owner import admit_fire, cron_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unified", "legacy", "unattributed", "ordinary"])
async def test_list_and_detail_expose_only_known_cron_identity(tmp_path, monkeypatch, kind):
    owner = cron_owner(tmp_path, monkeypatch)
    db = owner.db
    try:
        job = jobs.create_job(prompt="report", schedule="every 1h", deliver="local")["id"]
        if kind == "unified":
            sid, execution = await admit_fire(owner, job)
            expected = {"cron_job_id": job, "cron_execution_id": execution}
        else:
            sid = f"cron_{job}_20261011_120000" if kind == "legacy" else f"local-{kind}"
            db.create_session(sid, source="api" if kind == "ordinary" else "cron")
            expected = {"cron_job_id": job, "cron_execution_id": None} if kind == "legacy" else {}

        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
        adapter._session_db = db
        app = web.Application()
        app.router.add_get("/api/sessions", adapter._handle_list_sessions)
        app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/api/sessions")).status == 401
            headers = {"Authorization": "Bearer test-key"}
            response = await client.get("/api/sessions", headers=headers)
            assert response.status == 200
            row = next(row for row in (await response.json())["data"] if row["id"] == sid)
            response = await client.get(f"/api/sessions/{sid}", headers=headers)
            assert response.status == 200
            detail = (await response.json())["session"]
            for result in (row, detail):
                identity = {key: result[key] for key in ("cron_job_id", "cron_execution_id") if key in result}
                assert identity == expected
                assert "system_prompt" not in result
                assert "model_config" not in result
    finally:
        db.close()
