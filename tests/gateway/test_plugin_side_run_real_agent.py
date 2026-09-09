"""Real AIAgent/SDK/resolver/SQLite path with an in-memory HTTP transport."""
import asyncio
from copy import deepcopy
import json
import threading
from types import SimpleNamespace

import httpx
import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_real_parent_and_child_keep_separate_prompts_routes_and_transcripts(tmp_path, monkeypatch):
    from run_agent import AIAgent
    from gateway.side_runs import SideRunService
    from gateway.session_context import isolated_session_context
    from hermes_cli.plugin_side_runs import SideRunConfig
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom:fixture", "default": "forbidden-default"},
        "providers": {"fixture": {"base_url": "http://route.invalid/v1", "api_key": "fixture-only"}},
        "compression": {"enabled": False},
    }))
    db = SessionDB(tmp_path / "state.db")
    parent_entered = threading.Event()
    parent_release = threading.Event()
    requests = []

    def capture(request):
        assert request.url.host == "route.invalid"
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["model"] in {"parent-model", "child-model"}
        if payload["model"] == "parent-model":
            parent_entered.set()
            assert parent_release.wait(20)
        if payload.get("stream"):
            chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": 1,
                     "model": payload["model"], "choices": [{"index": 0, "finish_reason": "stop",
                     "delta": {"role": "assistant", "content": payload["model"] + " answer"}}],
                     "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json={"id": "fixture-completion", "object": "chat.completion", "created": 1,
            "model": payload["model"], "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": payload["model"] + " answer"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}})

    monkeypatch.setattr(AIAgent, "_build_keepalive_http_client",
                        staticmethod(lambda *args, **kwargs: httpx.Client(transport=httpx.MockTransport(capture))))
    runtime = resolve_runtime_provider(requested="custom:fixture", target_model="parent-model", strict=True)
    source = SessionSource(Platform.TELEGRAM, "chat", user_id="owner", thread_id="topic")
    parent = AIAgent(model="parent-model", provider=runtime["provider"], api_key=runtime["api_key"],
                     base_url=runtime["base_url"], api_mode=runtime["api_mode"],
                     requested_provider="custom:fixture", session_id="parent", session_db=db,
                     platform="telegram", enabled_toolsets=["todo"], fallback_model=[], max_iterations=2,
                     skip_memory=True, skip_context_files=True, skip_background_review=True, quiet_mode=True)
    runner = object.__new__(GatewayRunner)
    runner._draining = False
    runner._session_db = SimpleNamespace(_db=db)
    entry = SimpleNamespace(session_id="parent")
    runner.session_store = SimpleNamespace(_entries={"parent-key": entry})
    runner._session_key_for_source = lambda source: "parent-key"
    delivered = []
    async def send(chat_id, content, **kwargs):
        delivered.append((chat_id, content, kwargs))
        return SendResult(success=True)
    adapter = SimpleNamespace(send=send)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._adapter_for_source = lambda source: adapter
    service = SideRunService(runner)
    def run_parent():
        tokens = runner._set_session_env(SessionContext(source, [], {}, session_key="parent-key", session_id="parent"))
        try:
            with isolated_session_context("parent"):
                return parent.run_conversation(user_message="parent prompt", conversation_history=None, task_id="parent")
        finally:
            runner._clear_session_env(tokens)
    parent_task = asyncio.create_task(asyncio.to_thread(run_parent))
    try:
        assert await asyncio.to_thread(parent_entered.wait, 10)
        parent_prompt = parent._cached_system_prompt
        parent_tools = deepcopy(parent.tools)
        assert parent_tools
        parent_row = db.get_session("parent")
        parent_messages = db.get_messages("parent")
        child_id = service.start(MessageEvent(text="/route child prompt", source=source, message_id="trigger"),
                                 "fixture", "child prompt", SideRunConfig.from_mapping({
                                     "provider": "custom:fixture", "model": "child-model", "tools": [],
                                     "max_iterations": 2, "max_tokens": 80, "run_budget_seconds": 15}))
        await asyncio.wait_for(service.wait(), 15)
        expected_metadata = {**runner._thread_metadata_for_source(source, "trigger"), "_interim_send": True}
        assert delivered == [("chat", "child-model answer", {"reply_to": "trigger", "metadata": expected_metadata})]
        assert parent._cached_system_prompt == parent_prompt
        assert parent.tools == parent_tools
        assert parent.model == "parent-model"
        assert db.get_session("parent") == parent_row
        assert db.get_messages("parent") == parent_messages
        assert runner.session_store._entries["parent-key"] is entry
        child = db.get_session(child_id)
        assert child["end_reason"] == "side_run_completed"
        assert json.loads(child["model_config"])["side_run"]["owner"]["user_id"] == source.user_id
        assert child["model"] == "child-model"
        assert child["system_prompt"] or child.get("system_prompt_hash")
        assert [m["content"] for m in db.get_messages(child_id) if m["role"] in {"user", "assistant"}] == ["child prompt", "child-model answer"]
        assert sorted(r["model"] for r in requests) == ["child-model", "parent-model"]
        child_payload = next(r for r in requests if r["model"] == "child-model")
        assert all("parent prompt" not in str(m) for m in child_payload["messages"])
        assert not child_payload.get("tools")
    finally:
        parent_release.set()
        service.shutdown()
        await asyncio.wait_for(service.wait(), 15)
        await asyncio.wait_for(parent_task, 15)
        parent.close()
        runner._shutdown_executor()
        db.close()


@pytest.mark.asyncio
async def test_side_delivery_does_not_seal_parent_native_stream():
    from gateway.side_runs import SideRunService
    from tests.gateway.relay.test_relay_live_cards import _connected_adapter
    adapter, _ = _connected_adapter(supported_ops=("send", "edit", "typing", "draft"))
    class Transport:
        def __init__(self):
            self.ops = []
        async def send_outbound(self, payload, platform=None):
            self.ops.append(dict(payload))
            return {"success": True, "message_id": "111.222"}
    transport = Transport()
    adapter._transport = transport
    runner = object.__new__(GatewayRunner)
    source = SessionSource(Platform.SLACK, "C1", user_id="owner", thread_id="1700.42", scope_id="workspace")
    event = MessageEvent(text="/side prompt", source=source, message_id="1700.43")
    metadata = runner._thread_metadata_for_source(source, event.message_id)
    await adapter.send_draft("C1", 5, "parent streaming", metadata=metadata)
    key = adapter._draft_key("C1", metadata)
    service = SideRunService(runner, max_concurrent=1)
    await service._send(SimpleNamespace(adapter=adapter, event=event), "child final")
    assert adapter._open_draft_by_chat[key] == 5
    sent = [op for op in transport.ops if op["op"] == "send"]
    assert len(sent) == 1
    assert sent[0]["metadata"]["scope_id"] == source.scope_id
    assert sent[0]["metadata"]["user_id"] == source.user_id
    assert not any(op.get("final") for op in transport.ops)
