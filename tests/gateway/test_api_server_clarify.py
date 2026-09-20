"""Run-bound clarify transport, using the real registry and HTTP handlers."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp import StreamReader, web
from aiohttp.test_utils import make_mocked_request

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tools import clarify_gateway as clarify


@pytest.fixture
def adapter():
    instance = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    yield instance
    for session in list(instance._run_approval_sessions.values()):
        clarify.clear_session(session)
    instance._run_idempotency_store.close()
    instance._close_cached_session_dbs()


def app_for(adapter):
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return app


class InProcessClient:
    """Real aiohttp routing and JSON parsing without binding a listening socket."""

    def __init__(self, app):
        self.app = app

    async def __aenter__(self):
        self.app.freeze()
        return self

    async def __aexit__(self, *args):
        await self.app.cleanup()

    async def post(self, path, *, json=None, data=None, headers=None):
        import json as json_module

        payload = StreamReader(MagicMock(_reading_paused=False), 65536)
        payload.feed_data((data if data is not None else json_module.dumps(json)).encode())
        payload.feed_eof()
        request = make_mocked_request(
            "POST", path, headers={"Content-Type": "application/json", **(headers or {})},
            app=self.app, payload=payload, loop=asyncio.get_running_loop())
        return await self.app._handle(request)


def claim(adapter, run_id):
    request = SimpleNamespace(headers={})
    adapter._run_owners[run_id] = adapter._run_idempotency_scope(request)
    adapter._run_approval_sessions[run_id] = run_id
    adapter._run_streams[run_id] = asyncio.Queue()
    adapter._set_run_status(run_id, "running", session_id="shared-transcript")


def room_headers(adapter, member, permission):
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import decode_room_grant, issue_room_grant

    token = issue_room_grant(
        adapter._room_grant_secret(), grant_id=f"grant-{member}-{permission}",
        room_id="room", home_install_id="home", authority_gateway_id="authority", authority_epoch=1,
        member_id=member, target_install_id=hosted_rooms.local_authority_gateway_id(),
        target_profile="default", permissions=(permission,))
    claims = decode_room_grant(adapter._room_grant_secret(), token, permission=permission)
    hosted_rooms.reserve_peer_room(hosted_rooms.default_db_path(), claims=claims,
                                   expires_at=claims["status_expires_at"])
    return {"Authorization": f"HermesRoom {token}"}


@pytest.mark.asyncio
@pytest.mark.parametrize("choices,multi", [(None, False), (["one", "two"], False), (["one", "two"], True)])
async def test_clarify_controls_are_exact_pending_owned_run_bindings(adapter, choices, multi):
    claim(adapter, "run-a")
    claim(adapter, "run-b")  # Newer run, even on the same transcript, must not receive A's prompt.
    entry = clarify.register("question-a", "run-a", "Pick?", choices, multi_select=multi)
    other = clarify.register("question-b", "run-b", "Different?", choices)
    try:
        result = await adapter.send_clarify("shared-transcript", entry.question, choices,
                                          entry.clarify_id, entry.session_key)
        assert result.success
        event = adapter._run_streams["run-a"].get_nowait()
        assert event["event"] == "clarify.request"
        assert {k: event[k] for k in ("run_id", "clarify_id", "question", "choices", "multi_select")} == {
            "run_id": "run-a", "clarify_id": entry.clarify_id, "question": entry.question,
            "choices": choices, "multi_select": multi,
        }
        text = adapter._run_streams["run-a"].get_nowait()
        assert text["event"] == "message.delta" and entry.question in text["delta"]
        if choices:
            assert "1. one" in text["delta"] and "2. two" in text["delta"]
            assert ("Multiple selections allowed" in text["delta"]) == multi
        assert entry.awaiting_text  # The base fallback must still arm typed responses.
        assert adapter._run_streams["run-b"].empty()

        async with InProcessClient(app_for(adapter)) as client:
            async def post(run="run-a", body=None, key="test-key"):
                return await client.post(f"/v1/runs/{run}/clarify", json=body,
                                         headers={"Authorization": f"Bearer {key}"})

            body = {"clarify_id": entry.clarify_id, "response": "one"}
            assert (await post(body=body, key="wrong")).status == 401
            assert (await post("run-b", body)).status == 409
            adapter._run_owners["run-a"] = "different-owner"
            assert (await post(body=body)).status == 404
            claim_scope = adapter._run_idempotency_scope(SimpleNamespace(headers={}))
            adapter._run_owners["run-a"] = claim_scope
            for invalid in ([], {"clarify_id": entry.clarify_id}, {"clarify_id": 3, "response": "x"},
                            {"clarify_id": entry.clarify_id, "response": []},
                            {"clarify_id": entry.clarify_id, "response": "x" * 65537},
                            {"clarify_id": " " * 2, "response": None},
                            {"clarify_id": "x" * 257, "response": None}):
                assert (await post(body=invalid)).status == 400
            assert (await client.post("/v1/runs/run-a/clarify", data="{",
                                      headers={"Authorization": "Bearer test-key"})).status == 400
            assert (await post(body={"clarify_id": "unknown", "response": "x"})).status == 409
            assert not entry.event.is_set() and not other.event.is_set()

            response = await post(body={"clarify_id": entry.clarify_id, "response": None})
            assert response.status == 200
            assert json.loads(response.text)["awaiting_text"] is True
            assert not entry.event.is_set()
            assert clarify.resolve_text_response_for_session("run-a", "1 2" if multi else "my own answer")
            expected = json.dumps(choices) if multi else "my own answer"
            assert entry.response == expected
            assert (await post(body=body)).status == 409
            assert (await post(body={**body, "response": None})).status == 409
            assert clarify.wait_for_response(entry.clarify_id, 1) == expected

            await adapter.retire_clarify_card(entry.clarify_id, "Answered")
            retired = adapter._run_streams["run-a"].get_nowait()
            assert retired["event"] == "clarify.retired"
            assert retired["run_id"] == "run-a" and retired["clarify_id"] == entry.clarify_id
            assert retired["notice"] == "Answered"
            assert adapter._run_streams["run-a"].get_nowait()["delta"] == "Answered"
            await adapter.retire_clarify_card(entry.clarify_id, "Repeated")
            assert adapter._run_streams["run-a"].empty()
            assert (await post(body=body)).status == 409

            await adapter.send_clarify("shared-transcript", other.question, choices,
                                       other.clarify_id, other.session_key)
            for status in ("completed", "failed", "cancelled", "interrupted", "stopping"):
                adapter._set_run_status("run-b", status)
                assert (await post("run-b", {"clarify_id": other.clarify_id, "response": "x"})).status == 409
            assert not other.event.is_set()

            # A stale id must fail while its run is still live, including after
            # a new registry entry reuses the id: object identity is part of the binding.
            adapter._set_run_status("run-b", "running")
            clarify.clear_session("run-b")
            replacement = clarify.register(other.clarify_id, "run-b", "Replacement?", choices)
            assert (await post("run-b", {"clarify_id": other.clarify_id, "response": "x"})).status == 409
            assert not replacement.event.is_set()
            await adapter.send_clarify("shared-transcript", replacement.question, choices,
                                       replacement.clarify_id, replacement.session_key)

            # Real signed grants and ownership checks; approve alone suffices,
            # dispatch alone does not, and a different member cannot answer.
            owner_headers = room_headers(adapter, "owner", "approve")
            wrong_headers = room_headers(adapter, "outsider", "approve")
            dispatch_headers = room_headers(adapter, "owner", "dispatch")
            path = "/v1/runs/run-b/clarify"
            request = make_mocked_request("POST", path, headers=owner_headers)
            adapter._run_owners["run-b"] = adapter._run_idempotency_scope(request)
            room_body = {"clarify_id": replacement.clarify_id, "response": None}
            assert (await client.post(path, json=room_body, headers=wrong_headers)).status == 404
            assert (await client.post(path, json=room_body, headers=dispatch_headers)).status == 401
            assert (await client.post(path, json=room_body, headers=owner_headers)).status == 200
            assert not replacement.event.is_set()
            # Expiry removes the registry entry independently of its rendered card.
            assert await asyncio.to_thread(clarify.wait_for_response, replacement.clarify_id, 0.01) is None
            assert (await client.post(path, json=room_body, headers=owner_headers)).status == 409
    finally:
        clarify.clear_session("run-a")
        clarify.clear_session("run-b")


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["answer", "timeout", "stop", "shutdown", "cancel"])
async def test_run_clarify_registry_endpoint_round_trip(adapter, monkeypatch, outcome):
    agent = MagicMock()
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    agent.session_id = None
    answer = {}
    finished = threading.Event()

    def conversation(**kwargs):
        from tools.clarify_tool import clarify_tool
        try:
            answer.update(json.loads(clarify_tool("Which?", ["one", "two"], multi_select=True,
                                                 callback=agent.clarify_callback)))
            return {"final_response": "done"}
        finally:
            finished.set()

    agent.run_conversation.side_effect = conversation
    def create_agent(**kwargs):
        assert kwargs['interactive_run'] is True
        return agent
    monkeypatch.setattr(adapter, "_create_agent", create_agent)
    timeout = {"timeout": 2, "answer": 10}.get(outcome, 0)
    monkeypatch.setattr(clarify, "get_clarify_timeout", lambda: timeout)
    headers = {"Authorization": "Bearer test-key"}
    run_id = None
    task = None
    try:
        async with InProcessClient(app_for(adapter)) as client:
            response = await client.post("/v1/runs", json={"input": "ask me"}, headers=headers)
            assert response.status == 202
            run_id = json.loads(response.text)["run_id"]
            task = adapter._active_run_tasks[run_id]
            queue = adapter._run_streams[run_id]
            event = await asyncio.wait_for(queue.get(), 10)
            assert event["event"] == "clarify.request"
            cid = event["clarify_id"]
            assert event["multi_select"] is True
            with clarify._lock:
                entry = clarify._entries[cid]
                assert entry.session_key == run_id
            if outcome == "answer":
                response = await client.post(f"/v1/runs/{run_id}/clarify", headers=headers,
                                             json={"clarify_id": cid, "response": '["one", "two"]'})
                assert response.status == 200
                assert json.loads(response.text)["resolved"] is True
            elif outcome == "stop":
                response = await client.post(f"/v1/runs/{run_id}/stop", headers=headers)
                assert response.status == 200
            elif outcome == "shutdown":
                adapter.interrupt_active_runs("test shutdown")
            elif outcome == "cancel":
                task.cancel()
            if outcome == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await asyncio.wait_for(asyncio.shield(task), 10)
            assert await asyncio.to_thread(finished.wait, 10)
            events = []
            while not queue.empty():
                events.append(queue.get_nowait())
            assert any(e and e["event"] == "clarify.retired" and e["clarify_id"] == cid for e in events)
            assert cid not in clarify._entries
            response = await client.post(f"/v1/runs/{run_id}/clarify", headers=headers,
                                         json={"clarify_id": cid, "response": "late"})
            assert response.status == 409
            if outcome == "answer":
                assert answer["user_response"] == ["one", "two"]
            elif outcome != "cancel":
                assert "user did not respond" in str(answer["user_response"])
    finally:
        if run_id:
            clarify.clear_session(run_id)
        if task and not task.cancelled():
            await asyncio.wait_for(asyncio.shield(task), 10)
