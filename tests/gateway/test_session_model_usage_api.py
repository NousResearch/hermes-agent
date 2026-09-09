"""Session detail usage must preserve the ledger and its authorization boundary."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_detail_pages_raw_main_and_auxiliary_usage_without_claiming_actual_cost(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "synthetic-key"}))
    adapter._session_db = db
    app = web.Application()
    app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
    try:
        db.create_session("subject", "api_server")
        db.update_token_counts("subject", input_tokens=101, output_tokens=7, model="main-model",
                               estimated_cost_usd=0.2, actual_cost_usd=0.03,
                               billing_base_url="https://private.invalid/secret", cost_status="estimated")
        db.record_auxiliary_usage("subject", "title_generation", model="aux-model", input_tokens=23,
                                  output_tokens=4, estimated_cost_usd=0.01)
        db.record_auxiliary_usage("other", "background_review", model="unrelated", input_tokens=900)
        headers = {"Authorization": "Bearer synthetic-key"}
        async with TestClient(TestServer(app)) as client:
            plain = await client.get("/api/sessions/subject", headers=headers)
            assert "model_usage" not in await plain.json()
            first = await client.get("/api/sessions/subject?include_usage=true&usage_limit=1", headers=headers)
            assert first.status == 200
            payload = await first.json()
            usage = payload["model_usage"]
            assert usage["pagination"] == {"limit": 1, "offset": 0, "has_more": True}
            main = usage["data"][0]
            assert (main["task"], main["model"], main["input_tokens"], main["output_tokens"]) == ("", "main-model", 101, 7)
            assert main["estimated_cost_usd"] == 0.2
            assert main["recorded_actual_cost_usd"] == 0.03
            assert main["actual_cost_usd"] is None
            second = await client.get("/api/sessions/subject?include_usage=true&usage_limit=1&usage_offset=1", headers=headers)
            page = (await second.json())["model_usage"]
            assert page["pagination"]["has_more"] is False
            auxiliary = page["data"][0]
            assert (auxiliary["task"], auxiliary["model"], auxiliary["input_tokens"]) == ("title_generation", "aux-model", 23)
            assert auxiliary["recorded_actual_cost_usd"] == 0
            assert auxiliary["actual_cost_usd"] is None
            assert auxiliary["estimated_cost_usd"] == 0.01
            assert set(main) == set(auxiliary) == {
                "model", "task", "api_call_count", "input_tokens", "output_tokens", "cache_read_tokens",
                "cache_write_tokens", "reasoning_tokens", "estimated_cost_usd", "recorded_actual_cost_usd",
                "actual_cost_usd", "cost_status", "first_seen", "last_seen",
            }
            empty = await client.get("/api/sessions/subject?include_usage=true&usage_offset=100", headers=headers)
            assert (await empty.json())["model_usage"]["data"] == []
            bounded = await client.get("/api/sessions/subject?include_usage=true&usage_limit=999999", headers=headers)
            assert (await bounded.json())["model_usage"]["pagination"]["limit"] == 500
            missing = await client.get("/api/sessions/missing?include_usage=true", headers=headers)
            assert missing.status == 404
            rejected = await client.get("/api/sessions/subject?include_usage=true")
            assert rejected.status == 401
    finally:
        db.close()


@pytest.mark.asyncio
async def test_detail_usage_reads_only_the_selected_profile_database(tmp_path, monkeypatch):
    default_home = tmp_path / "default"
    worker_home = tmp_path / "worker"
    default_home.mkdir()
    worker_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    databases = [SessionDB(home / "state.db") for home in (default_home, worker_home)]
    try:
        for db, model in zip(databases, ("default-model", "worker-model")):
            db.create_session("same-id", "api_server")
            db.record_auxiliary_usage("same-id", "title_generation", model=model, input_tokens=5)
        for home, expected in ((default_home, "default-model"), (worker_home, "worker-model")):
            monkeypatch.setenv("HERMES_HOME", str(home))
            app = web.Application()
            app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
            async with TestClient(TestServer(app)) as client:
                response = await client.get("/api/sessions/same-id?include_usage=true")
                assert response.status == 200
                rows = (await response.json())["model_usage"]["data"]
                assert [row["model"] for row in rows] == [expected]
    finally:
        adapter._close_cached_session_dbs()
        for db in databases:
            db.close()
