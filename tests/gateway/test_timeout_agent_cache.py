"""A timed-out worker must not lend its interrupted agent to another turn."""

import asyncio
import json
import socket
import threading
import time

import httpx
from openai import OpenAI
import pytest
import yaml


@pytest.mark.asyncio
@pytest.mark.parametrize("timed_out", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_next_turn_after_slow_post_hook(monkeypatch, tmp_path, timed_out, streaming):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "100")
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "audit-model", "context_length": 100000},
        "agent": {"gateway_timeout": 1 if timed_out else 0},
        "display": {"tool_progress": "off", "thinking": "off", "busy_input_mode": "queue",
                    "busy_text_mode": "queue", "long_running_notifications": "off"},
        "security": {"tirith_enabled": False},
        "auxiliary": {"title_generation": {"enabled": False}},
        "compression": {"enabled": False},
    }))

    def deny_connect(*args, **kwargs):
        raise AssertionError("No real network in this test")

    monkeypatch.setattr(socket.socket, "connect", deny_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_connect)
    from gateway.config import GatewayConfig, Platform, PlatformConfig, StreamingConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from hermes_cli.plugins import PluginContext, get_plugin_manager
    from hermes_cli.plugins_manifest import PluginManifest
    from hermes_state import SessionDB
    import agent.usage_pricing
    import run_agent

    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "1" if timed_out else "0")
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT_WARNING", "0")
    monkeypatch.setattr(agent.usage_pricing, "fetch_endpoint_model_metadata", lambda *a, **k: {})
    monkeypatch.setattr(GatewayRunner, "_init_startup_checks", lambda self: None)
    monkeypatch.setattr(GatewayRunner, "_resolve_session_agent_runtime", lambda *a, **k: (
        "audit-model", {"api_key": "offline", "base_url": "https://audit.invalid/v1",
                        "provider": "custom", "api_mode": "chat_completions"}))
    hook_entered, hook_release, hook_done = threading.Event(), threading.Event(), threading.Event()
    lease_waiting = threading.Event()
    agents, requests, results, workers, releases = [], [], [], [], []
    acquire_lease = SessionDB.acquire_session_turn_lease

    def record_lease_wait(self, *args, on_wait=None, **kwargs):
        def record_wait(elapsed):
            lease_waiting.set()
            if on_wait is not None:
                on_wait(elapsed)

        return acquire_lease(self, *args, on_wait=record_wait, **kwargs)

    monkeypatch.setattr(SessionDB, "acquire_session_turn_lease", record_lease_wait)

    def slow_post_hook(user_message, **kwargs):
        if str(user_message).startswith("FIRST"):
            hook_entered.set()
            try:
                assert hook_release.wait(20), "Post-hook barrier expired"
            finally:
                hook_done.set()

    class RecordingAgent(run_agent.AIAgent):
        def __init__(self, **kwargs):
            kwargs.update(skip_memory=True, skip_context_files=True, load_soul_identity=False,
                          enabled_toolsets=[], disabled_toolsets=[], save_trajectories=False)
            super().__init__(**kwargs)
            self._disable_streaming = not streaming
            agents.append(self)

        def release_clients(self, *args, **kwargs):
            releases.append((self, hook_done.is_set()))
            return super().release_clients(*args, **kwargs)

        def _create_request_openai_client(self, **kwargs):
            def response(request):
                payload = json.loads(request.content)
                text = [m["content"] for m in payload["messages"] if m["role"] == "user"][-1]
                requests.append((self, text))
                if payload.get("stream"):
                    base = {"id": "offline", "object": "chat.completion.chunk", "created": 0,
                            "model": "audit-model"}
                    chunks = [
                        {**base, "choices": [{"index": 0, "finish_reason": None,
                                             "delta": {"role": "assistant", "content": "ACK " + str(text)}}]},
                        {**base, "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}]},
                    ]
                    content = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
                    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                          content=content + "data: [DONE]\n\n")
                return httpx.Response(200, json={
                    "id": "offline", "object": "chat.completion", "created": 0, "model": "audit-model",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "ACK " + str(text)}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                })
            return OpenAI(api_key="offline", base_url="https://audit.invalid/v1",
                          http_client=httpx.Client(transport=httpx.MockTransport(response)))

    monkeypatch.setattr(run_agent, "AIAgent", RecordingAgent)
    real_run, real_worker = GatewayRunner._run_agent, GatewayRunner._run_agent_start_turn_worker

    async def record_run(self, *args, **kwargs):
        result = await real_run(self, *args, **kwargs)
        results.append(result)
        return result

    def record_worker(self, ctx, run_sync):
        worker = real_worker(self, ctx, run_sync)
        workers.append(worker)
        return worker

    monkeypatch.setattr(GatewayRunner, "_run_agent", record_run)
    monkeypatch.setattr(GatewayRunner, "_run_agent_start_turn_worker", record_worker)

    class Adapter(BasePlatformAdapter):
        async def connect(self, **kwargs):
            raise AssertionError("Offline adapter must not connect")

        async def disconnect(self):
            pass

        async def get_chat_info(self, chat_id):
            return {"id": chat_id}

        async def send_typing(self, *args, **kwargs):
            pass

        async def stop_typing(self, *args, **kwargs):
            pass

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            sent.append(content)
            return SendResult(success=True, message_id=str(len(sent)))

    runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions", streaming=StreamingConfig(enabled=False)))
    sent = []
    adapter = Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    adapter.gateway_runner = runner
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._wire_adapter_handlers(adapter)
    context = PluginContext(PluginManifest(name="timeout-cache-test"), get_plugin_manager())
    registration = context.register_hook("post_llm_call", slow_post_hook)

    async def until(predicate):
        deadline = time.monotonic() + 15
        while not predicate():
            assert time.monotonic() < deadline, (sent, results)
            await asyncio.sleep(0.02)

    async def send(text):
        event = MessageEvent(text=text, message_id=text, source=adapter.build_source(
            chat_id="100", chat_type="dm", user_id="100"))
        await adapter.handle_message(event)

    async def settle():
        await until(lambda: not any(not t.done() for t in adapter._background_tasks))

    try:
        await send("FIRST")
        await until(hook_entered.is_set)
        if not timed_out:
            hook_release.set()
        await settle()
        old_agent = agents[0]
        if timed_out:
            assert results[0]["failed"]
            assert not workers[0].worker_done.is_set()
            assert old_agent._interrupt_requested
            assert not releases
        assert not lease_waiting.is_set()
        await send("NEXT")
        await until(lambda: lease_waiting.is_set() or len(results) > 1)
        if timed_out:
            assert lease_waiting.is_set(), results
            assert not any(text == "NEXT" for _, text in requests)
        hook_release.set()
        await settle()
        for worker in workers:
            await asyncio.wait_for(asyncio.shield(worker.executor_task), 15)
        assert "ACK NEXT" in sent, results
        next_agent = next(a for a, text in requests if text == "NEXT")
        assert (next_agent is old_agent) is not timed_out
        if timed_out:
            await until(lambda: any(a is old_agent for a, _ in releases))
            assert [(a is old_agent, done) for a, done in releases] == [(True, True)]
        else:
            assert not releases
            assert not lease_waiting.is_set()
    finally:
        hook_release.set()
        await settle()
        for worker in workers:
            await asyncio.wait_for(asyncio.shield(worker.executor_task), 15)
        registration.dispose()
        runner._shutdown_executor(drain_timeout=5)
        for agent in agents:
            agent.release_clients()
        runner.close_all_session_db_handles()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", [
    "running", "finished", "cancelled_waiter", "worker_error", "interrupt_error", "real_result_wins",
])
@pytest.mark.parametrize("replacement_before_timeout", [False, True])
async def test_retirement_belongs_to_the_exact_worker(monkeypatch, phase, replacement_before_timeout):
    from gateway.run import GatewayRunner
    from gateway.turn_context import TurnContext

    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    entered, finish, body_done, released = (threading.Event() for _ in range(4))
    loop_thread = threading.get_ident()

    class Agent:
        def __init__(self):
            self.release_count = 0
            self._session_messages = ["retained transcript"]
            self._db_flush_scan_prefix = ["retained transcript"]
            self.session_resource = object()

        def get_activity_summary(self):
            return {"seconds_since_activity": 2, "last_activity_desc": "test worker"}

        def hard_interrupt(self, reason):
            if phase == "interrupt_error":
                raise RuntimeError("interrupt failed")

        def release_clients(self):
            assert body_done.is_set()
            assert threading.get_ident() != loop_thread
            self.release_count += 1
            released.set()

    old, new = Agent(), Agent()
    resource = old.session_resource
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = {"test": (old, "signature")}
    scheduled_releases = []
    spawn_release = runner._spawn_release_thread

    def record_release(target, args, name, *, inline_fallback):
        scheduled_releases.append(args[0])
        return spawn_release(target, args, name, inline_fallback=inline_fallback)

    monkeypatch.setattr(runner, "_spawn_release_thread", record_release)
    state = runner._session_state("test")
    state.conversation.ephemeral_pin = ("old", "prefix")
    state.conversation.vc_last = "old voice"
    ctx = TurnContext(session_key="test", agent_holder=[old])

    def run_sync():
        entered.set()
        try:
            assert finish.wait(10), "Worker barrier expired"
            if phase == "worker_error":
                raise RuntimeError("worker failed")
            return {"final_response": "completed"}
        finally:
            body_done.set()

    async def until(predicate):
        deadline = time.monotonic() + 5
        while not predicate():
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)

    def replace():
        runner._agent_cache["test"] = (new, "new signature")
        state.conversation.ephemeral_pin = ("new", "prefix")
        state.conversation.vc_last = "new voice"

    worker = runner._run_agent_start_turn_worker(ctx, run_sync)
    worker.agent_timeout = 1

    def select_timeout():
        if phase == "interrupt_error":
            with pytest.raises(RuntimeError, match="interrupt failed"):
                runner._run_agent_timeout_result(worker, ctx)
        else:
            runner._run_agent_timeout_result(worker, ctx)

    try:
        await until(entered.is_set)
        if replacement_before_timeout:
            replace()
        if phase in {"finished", "real_result_wins"}:
            finish.set()
            await worker.executor_task
        if phase == "real_result_wins":
            worker.timeout_fired.set()
            result = await runner._run_agent_await_turn_worker(worker, ctx, asyncio.Event(), None)
            assert result["final_response"] == "completed"
            assert runner._agent_cache["test"][0] is (new if replacement_before_timeout else old)
            assert old.release_count == new.release_count == 0
            return

        select_timeout()
        if replacement_before_timeout:
            assert runner._agent_cache["test"][0] is new
            assert state.conversation.ephemeral_pin == ("new", "prefix")
        else:
            assert "test" not in runner._agent_cache
            assert state.conversation.ephemeral_pin is None
            assert state.conversation.vc_last is None
            replace()
        if phase != "finished":
            assert old.release_count == 0
            assert old._session_messages == ["retained transcript"]
            assert old._db_flush_scan_prefix == ["retained transcript"]
        if phase == "cancelled_waiter":
            worker.executor_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker.executor_task
            assert not worker.worker_done.is_set()
            assert old.release_count == 0
        finish.set()
        if phase == "worker_error":
            with pytest.raises(RuntimeError, match="worker failed"):
                await worker.executor_task
        elif phase != "cancelled_waiter":
            await worker.executor_task
        await until(lambda: released.is_set() and old._db_flush_scan_prefix is None)
        select_timeout()
        assert scheduled_releases == ([old] if phase == "finished" else [])
        assert old.release_count == 1
        assert old._session_messages == []
        assert old._db_flush_scan_prefix is None
        assert old.session_resource is resource
        assert runner._agent_cache["test"][0] is new
        assert state.conversation.ephemeral_pin == ("new", "prefix")
        assert state.conversation.vc_last == "new voice"
        assert new.release_count == 0
    finally:
        finish.set()
        if not worker.executor_task.cancelled():
            await asyncio.gather(worker.executor_task, return_exceptions=True)
        runner._shutdown_executor(drain_timeout=5)
