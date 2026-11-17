"""POST /api/sessions/{id}/compress: the shared manual core commits, and a running turn refuses."""
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB

SID = "api-compress"


def _seeded(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session(SID, "api_server")
    for i in range(12):
        db.append_message(SID, "user", f"question {i} " + "x" * 400)
        db.append_message(SID, "assistant", f"answer {i} " + "y" * 400)
    return db


def _agent(db, monkeypatch, summary_tail):
    """A real AIAgent on the session; only the summarizer call is stubbed."""
    from agent.context_compressor import SUMMARY_PREFIX
    from run_agent import AIAgent
    agent = AIAgent(api_key="fixture-only", base_url="http://127.0.0.1:9/v1", model="test/model",
                    platform="api_server", quiet_mode=True, session_db=db, session_id=SID,
                    skip_context_files=True, skip_memory=True, enabled_toolsets=[])
    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": SUMMARY_PREFIX + " earlier turns"}, summary_tail]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = compressor._last_aux_model_failure_error = None
    compressor.has_content_to_compress.return_value = True
    agent.context_compressor = compressor
    agent.compression_in_place = True
    agent._compression_feasibility_checked = True
    monkeypatch.setattr(agent, "_build_system_prompt", lambda *a, **kw: "fixture prompt")
    return agent


async def _post(adapter, body):
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path.endswith("/compress"):
            app.router.add_route(method, path, handler)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(f"/api/sessions/{SID}/compress", json=body)
        return resp.status, await resp.json()


@pytest.mark.asyncio
async def test_compress_commits_through_shared_core_and_keeps_archive(tmp_path, monkeypatch):
    db = _seeded(tmp_path)
    before = db.get_messages_as_conversation(SID)
    agent = _agent(db, monkeypatch, before[-1].copy())
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = db
    built = []
    monkeypatch.setattr(adapter, "_create_agent", lambda **kw: built.append(kw) or agent)
    try:
        status, payload = await _post(adapter, {"instructions": "keep the API design"})
        assert status == 200, payload
        assert payload["status"] == "compressed"
        assert payload["session_id"] == SID  # in-place compaction keeps the id
        assert payload["after_tokens"] < payload["before_tokens"]
        assert compressor_focus(agent) == "keep the API design"
        live = db.get_messages_as_conversation(SID)
        assert any("earlier turns" in str(m["content"]) for m in live)
        assert payload["messages_removed"] == len(before) - len(live) > 0
        # The commit soft-archived the originals: a destructive rewrite would have dropped them.
        archived = db.get_messages(SID, include_compacted=True)
        assert any(m["content"].startswith("question 0 ") for m in archived)
        assert db.get_compression_lock_holder(SID) is None
    finally:
        agent.close()
        db.close()


def compressor_focus(agent):
    return agent.context_compressor.compress.call_args.kwargs.get("focus_topic")


@pytest.mark.asyncio
async def test_compress_refuses_while_a_turn_holds_the_session(tmp_path, monkeypatch):
    db = _seeded(tmp_path)
    before = db.get_messages_as_conversation(SID)
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = db
    monkeypatch.setattr(adapter, "_create_agent", lambda **kw: pytest.fail("no agent while a turn runs"))
    assert db.try_acquire_session_turn_lease(SID, "pid=1:turn=live", ttl_seconds=300)
    try:
        status, payload = await _post(adapter, {})
        assert status == 409
        assert payload["error"]["code"] == "session_active_turn"
        assert db.get_messages_as_conversation(SID) == before
    finally:
        db.release_session_turn_lease(SID, "pid=1:turn=live")
        db.close()
