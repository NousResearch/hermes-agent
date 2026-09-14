"""Real HTTP handlers + shared admission/counter: no provider network or credentials."""
import json
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent import free_tier
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import free_tier_usage as usage


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    monkeypatch.setattr(usage, "current_identity", lambda: "fixture-identity")
    monkeypatch.setattr(usage, "_usage_path", lambda: tmp_path / "usage.json")
    monkeypatch.setattr("hermes_cli.onboarding_profile.is_onboarding_profile", lambda: False)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    entered = []

    def create(**kwargs):
        agent = SimpleNamespace(base_url="https://welcome-api.nousresearch.com/v1", provider="nous",
                                model="nous/welcome", session_id=kwargs.get("session_id"),
                                session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0)

        if getattr(adapter, "recovered", False):
            agent.base_url = "http://localhost:11434/v1"
            agent.provider = "custom"
        def run(user_message, conversation_history=None, **_):
            history = list(conversation_history or [])
            with free_tier.admit_turn(agent, history) as blocked:
                if blocked is not None:
                    return blocked
                entered.append(user_message)
                free_tier.record_tool_completion(agent)
                callback = kwargs.get("stream_delta_callback")
                if callback:
                    callback("Finished.")
                return free_tier.finish_turn(agent, {"completed": True, "api_calls": 1,
                    "final_response": "Finished.", "messages": history + [
                        {"role": "user", "content": user_message}, {"role": "assistant", "content": "Finished."}]})
        agent.run_conversation = run
        return agent

    monkeypatch.setattr(adapter, "_create_agent", create)
    adapter.entered = entered
    return adapter


def app_for(adapter):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    return app


def events(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["chat", "responses", "native", "runs"])
@pytest.mark.parametrize("stream", [False, True])
async def test_refusals_are_failed_and_recoverable_without_transcript_mutation(adapter, surface, stream):
    for _ in range(usage.TOOL_CALL_CAP):
        usage.record_completed_tool("fixture-identity")
    db = await adapter._ensure_session_db_async()
    db.create_session(session_id="existing", source="api_server")
    history = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"}]
    db.replace_messages("existing", history)
    before = db.get_messages_as_conversation("existing")
    path, body = {
        "chat": ("/v1/chat/completions", {"messages": [{"role": "user", "content": "pending"}], "stream": stream}),
        "responses": ("/v1/responses", {"input": "pending", "stream": stream, "store": True}),
        "native": ("/api/sessions/existing/chat" + ("/stream" if stream else ""), {"message": "pending"}),
        "runs": ("/v1/runs", {"input": "pending", "session_id": "existing"}),
    }[surface]
    async with TestClient(TestServer(app_for(adapter))) as client:
        response = await client.post(path, json=body)
        if surface == "runs":
            assert response.status == 202
            run_id = (await response.json())["run_id"]
            ev = events(await (await client.get(f"/v1/runs/{run_id}/events")).text())
            payload = await (await client.get(f"/v1/runs/{run_id}")).json()
            assert payload["status"] == "failed"
            assert ev[-1]["event"] == "run.failed"
            meta = payload
            assert payload["output"] == usage.LIMIT_NOTICE
        elif stream:
            ev = events(await response.text())
            if surface == "chat":
                assert "".join(e["choices"][0]["delta"].get("content", "") for e in ev) == usage.LIMIT_NOTICE
                assert ev[-1]["choices"][0]["finish_reason"] == "error"
                meta = ev[-1]["hermes"]
            elif surface == "responses":
                payload = ev[-1]["response"]
                assert ev[-1]["type"] == "response.failed"
                assert payload["output"][-1]["content"][0]["text"] == usage.LIMIT_NOTICE
                meta = payload["hermes"]
                stored = adapter._response_store.get(payload["id"])
                assert stored["conversation_history"] == []
                assert stored["pending_prompt"] == "pending"
            else:
                assert not any(e.get("completed") is True for e in ev)
                failed = next(e for e in ev if e.get("event") == "run.failed" or e.get("code") == "free_tier_limit")
                meta = failed
                assert any(e.get("content") == usage.LIMIT_NOTICE for e in ev)
        else:
            assert response.status == 403
            payload = await response.json()
            meta = payload["hermes"]
            assert payload["error"]["code"] == "free_tier_limit"
            assert payload["error"]["message"] == usage.LIMIT_NOTICE
        assert meta["code"] == "free_tier_limit"
        if surface == "chat" and stream:
            assert meta["error_code"] == "free_tier_limit"
        if surface == "native" and stream:
            status = await (await client.get(f"/v1/runs/{meta['run_id']}")).json()
            assert status["status"] == "failed" and status["code"] == "free_tier_limit"
        assert meta["completed"] is False and meta["failed"] is True
        assert meta["retryable"] is False and meta["retryable_after_recovery"] is True
        assert meta["pending_prompt"] == "pending"
        assert adapter.entered == []
        assert db.get_messages_as_conversation("existing") == before
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_explicit_resend_after_recovery_is_not_stuck_in_idempotency_cache(adapter):
    for _ in range(usage.TOOL_CALL_CAP):
        usage.record_completed_tool("fixture-identity")
    async with TestClient(TestServer(app_for(adapter))) as client:
        body = {"messages": [{"role": "user", "content": "retry exactly"}]}
        headers = {"Idempotency-Key": "recoverable-request"}
        refused = await client.post("/v1/chat/completions", json=body, headers=headers)
        assert refused.status == 403
        adapter.recovered = True
        assert adapter.entered == []
        response = await client.post("/v1/chat/completions", json=body, headers=headers)
        assert response.status == 200
        assert (await response.json())["choices"][0]["message"]["content"] == "Finished."
        assert adapter.entered == ["retry exactly"]
        await client.post("/v1/chat/completions", json=body, headers=headers)
        assert adapter.entered == ["retry exactly"]  # completed work still deduplicates
    await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["chat", "responses", "native", "runs"])
async def test_crossing_notice_streams_once_without_replaying_the_answer(adapter, surface):
    for _ in range(usage.TOOL_CALL_CAP - 1):
        usage.record_completed_tool("fixture-identity")
    db = await adapter._ensure_session_db_async()
    db.create_session(session_id="existing", source="api_server")
    path, body = {
        "chat": ("/v1/chat/completions", {"messages": [{"role": "user", "content": "finish"}]}),
        "responses": ("/v1/responses", {"input": "finish"}),
        "native": ("/api/sessions/existing/chat/stream", {"message": "finish"}),
        "runs": ("/v1/runs", {"input": "finish", "session_id": "existing"}),
    }[surface]
    async with TestClient(TestServer(app_for(adapter))) as client:
        response = await client.post(path, json={**body, "stream": True})
        if surface == "runs":
            run_id = (await response.json())["run_id"]
            ev = events(await (await client.get(f"/v1/runs/{run_id}/events")).text())
            assert ev[-1]["event"] == "run.completed"
            assert ev[-1]["free_tier"]["capped"] is True
            text = ev[-1]["output"]
        else:
            ev = events(await response.text())
            if surface == "chat":
                text = "".join(e["choices"][0]["delta"].get("content", "") for e in ev)
                assert ev[-1]["choices"][0]["finish_reason"] == "stop"
                assert ev[-1]["hermes"]["free_tier"]["capped"] is True
            elif surface == "responses":
                text = "".join(e.get("delta", "") for e in ev if e["type"] == "response.output_text.delta")
                assert ev[-1]["type"] == "response.completed"
                stored = adapter._response_store.get(ev[-1]["response"]["id"])
                assert stored["conversation_history"][-1]["content"] == "Finished."
            else:
                completed = next(e for e in ev if e.get("content"))
                text = completed["content"]
                assert completed["completed"] is True and completed["free_tier"]["capped"] is True
        assert text == "Finished.\n\n" + usage.LIMIT_NOTICE
        assert adapter.entered == ["finish"]
    await adapter.disconnect()
