"""Run continuation must not turn a transport cancellation into another writer.

This exercises the real HTTP admission, executor and cancellation paths. Only
agent construction is replaced: a controlled blocking writer stands in for a
provider/tool operation which has not returned. This is an executor-boundary
regression, not proof of tool-effect recovery or a production restart canary.
"""

import asyncio
import json
import threading
from contextlib import suppress
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server_runs
from gateway.platforms.api_server import APIServerAdapter


@pytest.mark.asyncio
async def test_cancelled_transport_does_not_authorize_continuation(tmp_path):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "fixture-key"}))
    active_tasks = getattr(adapter, "_active_run_tasks")
    app = web.Application()
    # Use the production route table, not a test-only continuation route.
    for method, path, handler in api_server_runs._http_routes(adapter):
        app.router.add_route(method, path, handler)

    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    exited = asyncio.Event()
    release = threading.Event()
    effect_path = tmp_path / "unfinished-operation.txt"
    agent = MagicMock()
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0

    def blocking_operation(**kwargs):
        effect_path.write_text("effect occurred, operation has not settled", encoding="utf-8")
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(10):
            raise TimeoutError("test cleanup did not release the controlled operation")
        return {"final_response": "interrupted", "interrupted": True}

    agent.run_conversation.side_effect = blocking_operation
    original_worker = api_server_runs._run_agent_sync

    def observe_worker_exit(*args, **kwargs):
        try:
            return original_worker(*args, **kwargs)
        finally:
            loop.call_soon_threadsafe(exited.set)

    headers = {"Authorization": "Bearer fixture-key", "Idempotency-Key": "initial-operation"}
    with (
        patch.object(adapter, "_create_agent", return_value=agent) as create_agent,
        patch.object(api_server_runs, "_run_agent_sync", side_effect=observe_worker_exit),
    ):
        async with TestClient(TestServer(app)) as client:
            try:
                accepted = await client.post("/v1/runs", json={"input": "perform bounded work"}, headers=headers)
                assert accepted.status == 202
                run_id = (await accepted.json())["run_id"]
                await asyncio.wait_for(entered.wait(), timeout=5)
                assert effect_path.read_text(encoding="utf-8") == "effect occurred, operation has not settled"

                # Cancelling the coroutine does not stop its executor thread.
                task = active_tasks[run_id]
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                status = await client.get(f"/v1/runs/{run_id}", headers=headers)
                assert (await status.json())["status"] == "cancelled"
                assert not exited.is_set()
                assert run_id not in active_tasks

                response = await client.post(
                    f"/v1/runs/{run_id}/continue",
                    json={"input": "continue the original work"},
                    headers={**headers, "Idempotency-Key": "continue-operation"},
                )
                assert response.status == 409
                assert (await response.json())["error"]["code"] == "run_not_recoverable"
                assert create_agent.call_count == 1
                assert not exited.is_set()
            finally:
                release.set()
                if entered.is_set():
                    await asyncio.wait_for(exited.wait(), timeout=5)
                for task in list(active_tasks.values()):
                    with suppress(asyncio.CancelledError):
                        await task
                api_server_runs._close_run_state(adapter)


def _response(*, path=None, call_id=None):
    calls = None if path is None else [SimpleNamespace(
        id=call_id, type="function", function=SimpleNamespace(
            name="write_file", arguments=json.dumps({"path": str(path), "content": "settled\n"}))) ]
    message = SimpleNamespace(content="done" if calls is None else "", tool_calls=calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop" if calls is None else "tool_calls")],
        model="test/model", usage=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "resume", "file_drift", "lost_receipt", "request_drift", "runtime_drift",
    "postclaim_drift", "copy_drift", "postclaim_identity", "copy_identity",
])
async def test_settled_file_step_continues_once_after_adapter_restart(tmp_path, monkeypatch, case):
    """Real agent loop + native write handler + durable DB; only the model is scripted."""
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools import file_tools

    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *args, **kwargs: None)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "transcript.db")
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    created_agents = []
    native_writes = []
    original_write = file_tools.write_file_tool

    def observe_write(*args, **kwargs):
        result = original_write(*args, **kwargs)
        native_writes.append((args, kwargs, json.loads(result)))
        if case == "lost_receipt":
            raise TimeoutError("simulated loss after real file effect, before receipt delivery")
        return result

    def make_agent(**kwargs):
        with (
            patch("model_tools.get_tool_definitions", return_value=[{
                "type": "function", "function": {"name": "write_file", "description": "Write a file",
                    "parameters": {"type": "object", "properties": {
                        "path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"]}}}]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
            patch("agent.model_metadata.fetch_model_metadata", return_value={}),
        ):
            agent: Any = AIAgent(
                api_key="test-key", base_url="https://example.invalid/v1", model="test/model",
                quiet_mode=True, skip_context_files=True, skip_memory=True,
                session_id=kwargs["session_id"], session_db=db,
            )
        if created_agents and case == "runtime_drift":
            agent.model = "unexpected/model"
        agent.client = MagicMock()
        agent._cached_system_prompt = "Complete the requested file steps in order; do not repeat completed steps."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False
        first_turn = not created_agents
        callback = kwargs["tool_progress_callback"]

        def progress(event, *args, **event_kwargs):
            callback(event, *args, **event_kwargs)
            if first_turn and event == "tool.completed":
                agent.interrupt("declared between-step test interruption")

        agent.tool_progress_callback = progress
        agent.client.chat.completions.create.side_effect = (
            [_response(path=first, call_id="first-step")] if first_turn else
            [_response(path=second, call_id="second-step"), _response()])
        created_agents.append(agent)
        return agent

    def make_adapter(key="fixture-key"):
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key}))
        monkeypatch.setattr(adapter, "_create_agent", make_agent)
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: db)
        async def get_db():
            return db
        monkeypatch.setattr(adapter, "_ensure_session_db_async", get_db)
        app = web.Application()
        for method, path, handler in api_server_runs._http_routes(adapter):
            app.router.add_route(method, path, handler)
        return adapter, app

    headers = {"Authorization": "Bearer fixture-key", "Idempotency-Key": "initial-file-steps"}
    body = {"input": "Write first.txt, then second.txt.", "recovery_policy": "verified_file_steps"}
    with patch.object(file_tools, "write_file_tool", side_effect=observe_write):
        adapter, app = make_adapter()
        try:
            async with TestClient(TestServer(app)) as client:
                response = await client.post("/v1/runs", json=body, headers=headers)
                assert response.status == 202
                source = (await response.json())["run_id"]
                await asyncio.wait_for(getattr(adapter, "_active_run_tasks")[source], timeout=30)
                assert first.read_text(encoding="utf-8") == "settled\n"
                assert not second.exists()
                assert len(native_writes) == 1
                assert native_writes[0][2].get("verified") is True
                status = await (await client.get(f"/v1/runs/{source}", headers=headers)).json()
                assert status["status"] == "cancelled"
                if case == "lost_receipt":
                    assert status.get("recovery", {}).get("disposition") == "blocked"
                    refused = await client.post(f"/v1/runs/{source}/continue", json=body,
                        headers={**headers, "Idempotency-Key": "refuse-lost-receipt"})
                    assert refused.status == 409
                    assert len(created_agents) == 1
                    db.close()
                    return
                assert status.get("recovery", {}).get("disposition") == "safe_to_continue"
        finally:
            api_server_runs._close_run_state(adapter)

        # New adapter/store connection: no in-memory launch or agent is carried over.
        if case == "resume":
            other, other_app = make_adapter("other-principal-key")
            try:
                async with TestClient(TestServer(other_app)) as client:
                    response = await client.post(f"/v1/runs/{source}/continue", json=body,
                        headers={"Authorization": "Bearer other-principal-key"})
                    assert response.status == 404
                    assert len(created_agents) == 1
            finally:
                api_server_runs._close_run_state(other)
        adapter, app = make_adapter()
        try:
            async with TestClient(TestServer(app)) as client:
                payload = {**body, "input": "Continue the remaining original step.",
                           "checkpoint_id": status["recovery"]["checkpoint_id"]}
                def change_checkpoint_file():
                    if case.endswith("identity"):
                        replacement = tmp_path / "replacement.txt"
                        original_identity = first.stat().st_ino
                        replacement.write_bytes(first.read_bytes())
                        replacement.replace(first)
                        assert first.stat().st_ino != original_identity
                    else:
                        first.write_text("changed after verification", encoding="utf-8")

                if case in {"postclaim_drift", "postclaim_identity"}:
                    store = getattr(adapter, "_run_idempotency_store")
                    original_reserve = store.reserve

                    def reserve_then_change(*args, **kwargs):
                        outcome = original_reserve(*args, **kwargs)
                        if kwargs.get("source_run_id") == source and outcome[0] == "created":
                            change_checkpoint_file()
                        return outcome

                    monkeypatch.setattr(store, "reserve", reserve_then_change)
                if case in {"copy_drift", "copy_identity"}:
                    original_append = db.append_messages_batch

                    def append_then_change(session_id, *args, **kwargs):
                        result = original_append(session_id, *args, **kwargs)
                        if session_id != source:
                            change_checkpoint_file()
                        return result

                    monkeypatch.setattr(db, "append_messages_batch", append_then_change)
                if case == "resume":
                    unauthorized = await client.post(f"/v1/runs/{source}/continue", json=payload)
                    assert unauthorized.status == 401
                    assert len(created_agents) == 1
                if case == "file_drift":
                    first.write_text("changed externally", encoding="utf-8")
                if case == "request_drift":
                    payload["model"] = "unexpected/model"
                continued = await client.post(f"/v1/runs/{source}/continue", json=payload,
                    headers={**headers, "Idempotency-Key": "continue-file-steps"})
                if case in {"file_drift", "request_drift"}:
                    assert continued.status == 409
                    assert len(created_agents) == 1
                    assert len(native_writes) == 1
                    assert not second.exists()
                    return
                assert continued.status == 202, await continued.text()
                successor = (await continued.json())["run_id"]
                assert successor != source
                await asyncio.wait_for(getattr(adapter, "_active_run_tasks")[successor], timeout=30)
                final = await (await client.get(f"/v1/runs/{successor}", headers=headers)).json()
                if case in {"runtime_drift", "postclaim_drift", "copy_drift", "postclaim_identity", "copy_identity"}:
                    assert final["status"] == "failed"
                    assert not second.exists()
                    assert len(native_writes) == 1
                    assert created_agents[1].client.chat.completions.create.call_count == 0
                    refused = await client.post(f"/v1/runs/{source}/continue", json=payload,
                        headers={**headers, "Idempotency-Key": "retry-after-claimed-failure"})
                    assert refused.status == 409
                    assert len(created_agents) == 2
                    return
                assert final["status"] == "completed"
                assert final["continued_from_run_id"] == source
                assert second.read_text(encoding="utf-8") == "settled\n"
                assert len(native_writes) == 2
                # The provider receives the actual previous call AND receipt, not a restarted task.
                call_messages = created_agents[1].client.chat.completions.create.call_args_list[0].kwargs["messages"]
                assert any(m.get("tool_call_id") == "first-step" for m in call_messages)
                durable_successor = db.get_messages_as_conversation(successor)
                assert any(m.get("tool_call_id") == "first-step" for m in durable_successor)
                assert any(m.get("tool_call_id") == "second-step" for m in durable_successor)
                replay = await client.post(f"/v1/runs/{source}/continue", json=payload,
                    headers={**headers, "Idempotency-Key": "continue-file-steps"})
                assert replay.status == 202
                assert (await replay.json())["run_id"] == successor
                duplicate = await client.post(f"/v1/runs/{source}/continue", json=payload,
                    headers={**headers, "Idempotency-Key": "different-key"})
                assert duplicate.status == 409
                assert len(created_agents) == 2
                assert len(native_writes) == 2
        finally:
            api_server_runs._close_run_state(adapter)
            db.close()
