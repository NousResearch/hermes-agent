"""Behavior contracts for Feishu native message_cot progress."""

import asyncio
import json
import queue
from types import SimpleNamespace

import pytest
from gateway.session import SessionSource
from gateway.config import Platform
import httpx

from plugins.platforms.feishu.cot import FeishuCOTClient
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext


class FakeCOTClient:
    def __init__(self, *, fail_stage: str | None = None, flush_interval: float = 0.01):
        self.fail_stage = fail_stage
        self.flush_interval = flush_interval
        self.calls: list[tuple[str, str, dict | None]] = []

    async def request_json(self, method: str, path: str, body: dict | None = None):
        self.calls.append((method, path, body))
        stage = (
            "create"
            if method == "POST" and path.startswith("/open-apis/im/v1/message_cot?")
            else ("complete" if "/complete/" in path else "update")
        )
        if self.fail_stage == stage:
            raise RuntimeError(stage)
        if stage == "create":
            return {"code": 0, "data": {"cot_id": "cot-1", "message_id": "om-cot-1"}}
        return {"code": 0}


def _payload(event: dict) -> dict:
    return json.loads(event["content"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "off"),
        (False, "off"),
        (True, "brief"),
        ("off", "off"),
        ("simple", "brief"),
        ("on", "brief"),
        ("brief", "brief"),
        ("detailed", "detailed"),
    ],
)
def test_cot_mode_config_contract(raw, expected):
    from gateway.display_config import resolve_display_setting
    from plugins.platforms.feishu.cot import normalize_cot_mode

    assert normalize_cot_mode(raw) == expected
    assert (
        resolve_display_setting(
            {"display": {"platforms": {"feishu": {"cot_messages": raw}}}},
            "feishu",
            "cot_messages",
        )
        == expected
    )


@pytest.mark.asyncio
async def test_brief_cot_uses_origin_and_omits_args_and_output():
    transport = FakeCOTClient()
    client = FeishuCOTClient(
        "app", "secret", "feishu", request_json=transport.request_json
    )
    run = await client.start(
        "oc-chat", "om-origin", "brief", flush_interval=transport.flush_interval
    )
    assert run is not None

    run.step(1)
    run.step(2, ["terminal"])
    run.tool_started("call-1", "terminal", {"command": "echo ok"})
    run.tool_completed(
        "call-1", "terminal", {"command": "echo ok"}, '{"exit_code":0,"output":"ok"}'
    )
    await run.finish("done")

    assert all("/messages/om-cot-1" not in path for _, path, _ in transport.calls)
    assert all(method != "PATCH" for method, _, _ in transport.calls)

    create = transport.calls[0]
    assert create == (
        "POST",
        "/open-apis/im/v1/message_cot?receive_id_type=chat_id",
        {"receive_id": "oc-chat", "origin_message_id": "om-origin"},
    )
    events = [
        event
        for method, path, body in transport.calls
        if method == "PUT"
        for event in body["events"]
    ]
    kinds = [event["event_type"] for event in events]
    assert kinds[0] == "RUN_STARTED"
    step_names = [
        _payload(event)["stepName"]
        for event in events
        if event["event_type"] == "STEP_STARTED"
    ]
    assert step_names == ["Agent 正在执行", "理解用户问题", "分析工具结果"]
    assert "TOOL_CALL_START" in kinds
    assert "TOOL_CALL_END" in kinds
    assert "TOOL_CALL_RESULT" in kinds
    assert kinds[-1] == "RUN_FINISHED"
    assert "TOOL_CALL_ARGS" not in kinds
    run_started = _payload(events[0])
    assert set(run_started) == {"threadId", "runId", "input"}
    tool_start = _payload(
        next(event for event in events if event["event_type"] == "TOOL_CALL_START")
    )
    assert set(tool_start) == {"toolCallId", "icon", "title", "toolCallName"}
    assert tool_start["toolCallName"] == "terminal"
    tool_result = _payload(
        next(event for event in events if event["event_type"] == "TOOL_CALL_RESULT")
    )
    assert set(tool_result) == {"messageId", "toolCallId", "role", "content"}
    assert '"output":"ok"' not in tool_result["content"]
    assert transport.calls[-1][:2] == (
        "POST",
        "/open-apis/im/v1/message_cot/complete/cot-1?message_id=om-cot-1&reason=done",
    )


@pytest.mark.asyncio
async def test_detailed_cot_force_redacts_and_truncates_args_and_output():
    transport = FakeCOTClient()
    client = FeishuCOTClient(
        "app", "secret", "feishu", request_json=transport.request_json
    )
    run = await client.start(
        "oc-chat", None, "detailed", flush_interval=transport.flush_interval
    )
    assert run is not None

    secret = "sk-" + "x" * 60
    run.tool_started(
        "call-1",
        "terminal",
        {"command": f"curl -H 'Authorization: Bearer {secret}' " + "x" * 1400},
    )
    run.tool_completed(
        "call-1", "terminal", {}, f"Authorization: Bearer {secret}" + "y" * 1400
    )
    await run.finish("done")

    events = [
        event
        for method, _, body in transport.calls
        if method == "PUT"
        for event in body["events"]
    ]
    args = _payload(
        next(event for event in events if event["event_type"] == "TOOL_CALL_ARGS")
    )["delta"]
    output = _payload(
        next(event for event in events if event["event_type"] == "TOOL_CALL_RESULT")
    )["content"]
    assert secret not in args + output
    assert "***" in args + output or "redacted" in (args + output).lower()
    assert len(args) <= 1200
    assert len(output) <= 1200


@pytest.mark.asyncio
async def test_commentary_is_visible_but_hidden_reasoning_has_no_cot_api():
    transport = FakeCOTClient()
    client = FeishuCOTClient(
        "app", "secret", "feishu", request_json=transport.request_json
    )
    run = await client.start(
        "oc-chat", "om-origin", "brief", flush_interval=transport.flush_interval
    )
    run.commentary("I am checking the gateway path.")
    assert not hasattr(run, "reasoning")
    await run.finish("done")

    events = [
        event
        for method, _, body in transport.calls
        if method == "PUT"
        for event in body["events"]
    ]
    kinds = [event["event_type"] for event in events]
    assert kinds.count("TEXT_MESSAGE_START") == 1
    assert kinds.count("TEXT_MESSAGE_CONTENT") == 1
    assert kinds.count("TEXT_MESSAGE_END") == 1
    assert (
        "I am checking"
        in _payload(
            next(e for e in events if e["event_type"] == "TEXT_MESSAGE_CONTENT")
        )["delta"]
    )
    assert "hidden" not in json.dumps(events).lower()


@pytest.mark.asyncio
async def test_same_named_tools_keep_distinct_call_ids_and_batch_until_finish():
    transport = FakeCOTClient(flush_interval=60)
    client = FeishuCOTClient(
        "app", "secret", "feishu", request_json=transport.request_json
    )
    run = await client.start(
        "oc-chat", "om-origin", "brief", flush_interval=transport.flush_interval
    )
    run.tool_started("call-1", "read_file", {"path": "a"})
    run.tool_completed("call-1", "read_file", {}, "a")
    run.tool_started("call-2", "read_file", {"path": "b"})
    run.tool_completed("call-2", "read_file", {}, "b")
    await asyncio.sleep(0)
    assert not any(method == "PUT" for method, _, _ in transport.calls)

    await run.finish("done")
    updates = [body for method, _, body in transport.calls if method == "PUT"]
    assert len(updates) == 1
    starts = [
        _payload(e)
        for e in updates[0]["events"]
        if e["event_type"] == "TOOL_CALL_START"
    ]
    assert [item["toolCallId"] for item in starts] == ["call-1", "call-2"]
    assert len({event["timestamp"] for event in updates[0]["events"]}) == len(
        updates[0]["events"]
    )


@pytest.mark.asyncio
async def test_cot_flushes_one_batch_after_interval_then_flushes_before_complete():
    transport = FakeCOTClient(flush_interval=0.01)
    client = FeishuCOTClient(
        "app", "secret", "feishu", request_json=transport.request_json
    )
    run = await client.start(
        "oc-chat", "om-origin", "brief", flush_interval=transport.flush_interval
    )
    run.tool_started("call-1", "read_file", {"path": "a"})
    await asyncio.sleep(0.03)

    first_update = next(body for method, _, body in transport.calls if method == "PUT")
    assert [event["event_type"] for event in first_update["events"]] == [
        "RUN_STARTED",
        "STEP_STARTED",
        "TOOL_CALL_START",
        "TOOL_CALL_END",
    ]

    run.commentary("finishing")
    await run.finish("done")
    assert transport.calls[-2][0] == "PUT"
    assert transport.calls[-1][1].endswith("reason=done")


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_stage", ["create", "update", "complete"])
async def test_cot_failures_are_nonfatal(fail_stage):
    transport = FakeCOTClient(fail_stage=fail_stage)
    client = FeishuCOTClient(
        "app", "secret", "feishu", request_json=transport.request_json
    )
    run = await client.start(
        "oc-chat", "om-origin", "brief", flush_interval=transport.flush_interval
    )
    if fail_stage == "create":
        assert run is None
        return
    run.commentary("safe progress")
    await run.finish("error")
    assert transport.calls[-1][1].endswith("reason=error")


def test_gateway_native_cot_uses_real_ids_and_suppresses_ordinary_progress():
    cot = SimpleNamespace(
        tool_started=lambda *args: calls.append(("start", args)),
        tool_completed=lambda *args: calls.append(("complete", args)),
        commentary=lambda text: calls.append(("commentary", text)),
    )
    calls = []
    ctx = TurnContext(
        native_cot=cot,
        progress_queue=queue.Queue(),
        tool_progress_enabled=True,
        _run_still_current=lambda: True,
    )
    runner = TurnRunner(SimpleNamespace(adapters={}), ctx)

    runner.progress_callback(
        "tool.started", "terminal", "echo ok", {"command": "echo ok"}
    )
    runner.progress_callback("_thinking", "private hidden reasoning", None, None)
    runner.combined_tool_start_callback("call-1", "terminal", {"command": "echo ok"})
    runner.combined_tool_complete_callback("call-1", "terminal", {}, "ok")
    runner.native_cot_commentary("visible commentary")

    assert ctx.progress_queue.empty()
    assert calls[0][0:2] == ("start", ("call-1", "terminal", {"command": "echo ok"}))
    assert calls[1][0] == "complete"
    assert calls[2] == ("commentary", "visible commentary")


def test_gateway_without_cot_keeps_ordinary_tool_progress():
    ctx = TurnContext(
        progress_queue=queue.Queue(),
        tool_progress_enabled=True,
        progress_mode="all",
        _run_still_current=lambda: True,
    )
    runner = TurnRunner(SimpleNamespace(adapters={}), ctx)

    runner.progress_callback(
        "tool.started", "terminal", "echo ok", {"command": "echo ok"}
    )

    assert ctx.progress_queue.get_nowait() == "⚙️ Running echo ok"


@pytest.mark.asyncio
async def test_default_off_does_not_call_adapter():
    adapter = SimpleNamespace(
        start_native_cot=lambda *_: pytest.fail("COT must stay off")
    )
    ctx = TurnContext(native_cot_mode="off", source=SimpleNamespace(chat_id="oc"))
    runner = TurnRunner(SimpleNamespace(_adapter_for_source=lambda _: adapter), ctx)

    await runner.start_native_cot()
    assert ctx.native_cot is None


@pytest.mark.asyncio
async def test_create_failure_falls_back_to_existing_tool_progress():
    adapter = SimpleNamespace(
        start_native_cot=lambda *_: asyncio.sleep(0, result=None),
        format_tool_preview=lambda prepared: prepared.text,
    )
    ctx = TurnContext(
        native_cot_mode="brief",
        source=SimpleNamespace(chat_id="oc"),
        progress_queue=queue.Queue(),
        tool_progress_enabled=True,
        progress_mode="all",
        _run_still_current=lambda: True,
    )
    runner = TurnRunner(SimpleNamespace(_adapter_for_source=lambda _: adapter), ctx)

    await runner.start_native_cot()
    runner.progress_callback(
        "tool.started", "terminal", "echo ok", {"command": "echo ok"}
    )

    assert ctx.native_cot is None
    assert ctx.progress_queue.get_nowait() == "⚙️ Running echo ok"


@pytest.mark.asyncio
async def test_tenant_token_is_cached_without_exposing_credentials():
    calls = []
    client = FeishuCOTClient("app-id", "app-secret", "feishu")

    async def fake_http(method, path, body, token=""):
        calls.append((method, path, body, token))
        if path.endswith("tenant_access_token/internal"):
            return {"code": 0, "tenant_access_token": "tenant-token", "expire": 7200}
        return {"code": 0}

    client._http_json = fake_http
    await client.request_json("PUT", "/cot", {})
    await client.request_json("PUT", "/cot", {})

    token_calls = [
        call for call in calls if call[1].endswith("tenant_access_token/internal")
    ]
    assert len(token_calls) == 1
    assert [call[3] for call in calls if call[1] == "/cot"] == [
        "tenant-token",
        "tenant-token",
    ]


def test_native_cot_disables_normal_stream_preview_but_keeps_commentary_callback():
    seen = []
    ctx = TurnContext(
        native_cot=SimpleNamespace(commentary=seen.append),
        interim_assistant_messages_enabled=False,
        resolve_display_setting=lambda *_: True,
        user_config={},
        streaming_tts_consumer_holder=[None],
        _run_still_current=lambda: True,
    )
    runner = TurnRunner(
        SimpleNamespace(
            config=SimpleNamespace(
                streaming=SimpleNamespace(enabled=True, transport="auto")
            )
        ),
        ctx,
    )

    consumer, delta_cb, commentary_cb, wants_commentary = runner._setup_stream_consumer(
        "feishu"
    )
    commentary_cb("visible phase")

    assert consumer is None
    assert delta_cb is None
    assert wants_commentary is True
    assert seen == ["visible phase"]


@pytest.mark.asyncio
async def test_detailed_redacts_plaintext_sensitive_fields_recursively():
    transport = FakeCOTClient()
    client = FeishuCOTClient(
        "app", "secret", "feishu", request_json=transport.request_json
    )
    fields = [
        "app_secret",
        "api_key",
        "x-api-key",
        "private_key",
        "private-key",
        "access_key",
        "access-key",
        "x_private_key",
        "x-access-key",
        "password",
        "access_token",
        "secret",
        "Authorization",
        "Cookie",
        "credential",
        "credentials",
    ]
    sensitive = {key: f"plain-value-{i}" for i, key in enumerate(fields)}
    data = {"nested": [sensitive], "key": "retain-this-detail"}
    run = await client.start("oc", None, "detailed", input_preview=json.dumps(data))
    run.tool_started("call", "search", {"query": json.dumps(data), **data})
    run.tool_completed("call", "search", {}, json.dumps(data))
    await run.finish()
    bodies = json.dumps(transport.calls)
    payloads = json.dumps([
        _payload(e)
        for method, _, body in transport.calls
        if method == "PUT"
        for e in body["events"]
    ])
    for value in sensitive.values():
        assert value not in bodies
        assert value not in payloads
    assert "retain-this-detail" in payloads
    assert "TOOL_CALL_ARGS" in bodies


@pytest.mark.asyncio
async def test_finish_closes_all_callbacks_before_waiting_for_update():
    transport = FakeCOTClient()
    entered, release = asyncio.Event(), asyncio.Event()

    async def request(method, path, body=None):
        if method == "PUT":
            entered.set()
            await release.wait()
        return await transport.request_json(method, path, body)

    client = FeishuCOTClient("app", "secret", "feishu", request_json=request)
    run = await client.start("oc", None, "detailed", flush_interval=60)
    await asyncio.to_thread(run.commentary, "accepted-before-close")
    finish = run.finish()
    assert finish is run.finish("error")
    await asyncio.wait_for(entered.wait(), 2)

    def late_callbacks():
        run.commentary("late-commentary")
        run.step(999)
        run.tool_started("late-call", "terminal", {"command": "late-command"})
        run.tool_completed("late-call", "terminal", {}, "late-result")

    await asyncio.to_thread(late_callbacks)
    release.set()
    await finish
    await asyncio.to_thread(late_callbacks)
    await asyncio.sleep(0)
    await run.flush()
    bodies = json.dumps(transport.calls, ensure_ascii=False)
    assert "accepted-before-close" in bodies
    assert "late-" not in bodies
    assert "规划下一步" not in bodies
    assert run._flush_task is None
    assert transport.calls[-1][1].endswith("reason=done")


@pytest.mark.asyncio
async def test_bridge_request_budget_allows_slow_create_update_and_complete(monkeypatch):
    transport = FakeCOTClient()

    async def request(method, path, body=None):
        if method == "PUT" or "/complete/" in path:
            await asyncio.sleep(1.05)
        return await transport.request_json(method, path, body)

    client = FeishuCOTClient("app", "secret", "feishu", request_json=request)

    async def start(*args):
        await asyncio.sleep(0.7)
        return await client.start(*args)

    gateway = object.__new__(GatewayRunner)
    monkeypatch.setattr(
        gateway,
        "_adapter_for_source",
        lambda source: SimpleNamespace(start_native_cot=start),
    )
    ctx = TurnContext(
        native_cot_mode="brief",
        source=SimpleNamespace(chat_id="oc"),
        _run_still_current=lambda: True,
    )
    turn = TurnRunner(gateway, ctx)
    try:
        await turn.start_native_cot()
        assert ctx.native_cot is not None
        ctx.native_cot.commentary("slow but healthy")
        await turn.finish_native_cot({})
        await asyncio.wait_for(asyncio.gather(*gateway._background_tasks), 6)
        assert any(method == "PUT" for method, _, _ in transport.calls)
        assert "/complete/" in transport.calls[-1][1]
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["create", "update", "complete"])
async def test_http_transport_timeout_cancels_without_late_side_effect(stage, monkeypatch):
    import plugins.platforms.feishu.cot as cot_module

    monkeypatch.setattr(cot_module, "COT_REQUEST_TIMEOUT_SECONDS", 0.01, raising=False)
    cancelled, release = asyncio.Event(), asyncio.Event()
    effects = []
    transport = FakeCOTClient()

    async def send(self, request):
        path = request.url.raw_path.decode()
        current = "complete" if "/complete/" in path else (
            "update" if request.method == "PUT" else "create"
        )
        if current == stage:
            try:
                await release.wait()
                effects.append(current)
            except asyncio.CancelledError:
                cancelled.set()
                raise
        body = json.loads(request.content) if request.content else None
        return httpx.Response(200, json=await transport.request_json(request.method, path, body))

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", send)
    client = FeishuCOTClient("app", "secret", "feishu")
    client._base_url = "http://127.0.0.1:1"  # Never contact Feishu, including on the red base.
    client._token, client._token_expires_at = "test-token", float("inf")
    gateway = object.__new__(GatewayRunner)
    gateway._adapter_for_source = lambda _: SimpleNamespace(start_native_cot=client.start)
    ctx = TurnContext(native_cot_mode="brief", source=SimpleNamespace(chat_id="oc"),
                      _run_still_current=lambda: True)
    turn = TurnRunner(gateway, ctx)
    try:
        await asyncio.wait_for(turn.start_native_cot(), 2)
        if stage == "create":
            assert ctx.native_cot is None
        else:
            assert ctx.native_cot is not None
            await turn.finish_native_cot({})
            await asyncio.wait_for(asyncio.gather(*gateway._background_tasks), 4)
            if stage == "update":
                assert "/complete/" in transport.calls[-1][1]
        assert cancelled.is_set()
        release.set()
        await asyncio.sleep(0)
        assert effects == []
    finally:
        release.set()
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stalled", [None, "complete", "finalizer"])
async def test_client_close_drains_gateway_finalizer_before_closing_transport(stalled, monkeypatch):
    entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    transport = FakeCOTClient()
    connections = []

    async def send(self, request):
        connections.append(self)
        if "/complete/" in request.url.path:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        body = json.loads(request.content) if request.content else None
        return httpx.Response(200, json=await transport.request_json(
            request.method, request.url.raw_path.decode(), body))

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", send)
    client = FeishuCOTClient("app", "secret", "feishu")
    client._base_url = "http://127.0.0.1:1"
    client._token, client._token_expires_at = "test-token", float("inf")
    cot = await client.start("oc", None, "brief", flush_interval=60)
    assert cot is not None
    if stalled == "finalizer":
        await cot._flush_lock.acquire()
    original_close = client._http_client.aclose

    async def close_transport():
        if stalled == "finalizer":
            assert cot._finalizer_task.cancelled()
        elif stalled == "complete":
            assert cancelled.is_set()
        else:
            assert any("/complete/" in path for _, path, _ in transport.calls)
        assert not client._finalizer_tasks
        assert not client._active_runs
        await original_close()

    monkeypatch.setattr(client._http_client, "aclose", close_transport)
    gateway = object.__new__(GatewayRunner)
    turn = TurnRunner(gateway, TurnContext(native_cot=cot, _run_still_current=lambda: True))
    # close owns runs that have not yet reached Gateway finish.
    closing = asyncio.create_task(client.close())
    try:
        if stalled == "finalizer":
            await asyncio.sleep(0)  # Let close begin while the update lock is held.
        else:
            await asyncio.wait_for(entered.wait(), 2)
        assert not closing.done()
        assert await client.start("oc", None, "brief") is None
        task = cot.finish()
        assert task is cot.finish("error")
        await turn.finish_native_cot({})
        assert task in gateway._background_tasks
        if not stalled:
            release.set()
        await asyncio.wait_for(closing, 5)
        assert cancelled.is_set() == (stalled == "complete")
        assert task.done()
        assert cot.finish() is task
        assert not client._finalizer_tasks
        assert not client._active_runs
        assert all(connection is connections[0] for connection in connections)
        assert client._http_client.is_closed
    finally:
        release.set()
        await closing
        if stalled == "finalizer":
            cot._flush_lock.release()
        await asyncio.gather(*getattr(gateway, "_background_tasks", ()), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["done", "failed", "exception", "cancel", "stale"])
async def test_proxy_turn_uses_native_cot_lifecycle(outcome, monkeypatch):
    transport = FakeCOTClient()
    client = FeishuCOTClient(
        "app", "secret", "feishu", request_json=transport.request_json
    )
    adapter = SimpleNamespace(start_native_cot=client.start)
    gateway = object.__new__(GatewayRunner)
    gateway.adapters = {Platform.FEISHU: adapter}
    gateway._adapter_for_source = lambda _: adapter
    gateway._get_proxy_url = lambda: "https://proxy.invalid"
    gateway._resolve_turn_toolsets = lambda *_: ([], [])
    current = [True]
    gateway._run_still_current_fn = lambda *_: lambda: current[0]
    gateway._run_agent_bind_turn_wiring = lambda *_: None
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"display": {"platforms": {"feishu": {"cot_messages": "brief"}}}},
    )

    async def proxy(**kwargs):
        if outcome == "exception":
            raise RuntimeError("proxy failed")
        if outcome == "cancel":
            raise asyncio.CancelledError()
        if outcome == "stale":
            current[0] = False
        if outcome == "failed":
            return gateway._agent_error_result("proxy unavailable")
        return {"final_response": "ordinary final"}

    gateway._run_agent_via_proxy = proxy
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc")
    try:
        result = await gateway._run_agent_inner("query", "", [], source, "sid")
        assert result["final_response"] == (
            "proxy unavailable" if outcome == "failed" else "ordinary final"
        )
    except (RuntimeError, asyncio.CancelledError):
        assert outcome in {"exception", "cancel"}
    await asyncio.gather(*getattr(gateway, "_background_tasks", set()))
    events = [
        e["event_type"]
        for method, _, body in transport.calls
        if method == "PUT"
        for e in body["events"]
    ]
    assert "RUN_STARTED" in events
    assert events[-1] == ("RUN_FINISHED" if outcome == "done" else "RUN_ERROR")
    assert "/complete/" in transport.calls[-1][1]


@pytest.mark.asyncio
async def test_authentication_early_return_marks_cot_failed():
    transport = FakeCOTClient()
    client = FeishuCOTClient(
        "app", "secret", "feishu", request_json=transport.request_json
    )
    cot = await client.start("oc", None, "brief")
    gateway = object.__new__(GatewayRunner)
    gateway._get_system_prompt_for_channel = lambda *args, **kwargs: ""

    def resolve(**kwargs):
        raise RuntimeError("invalid credentials")

    gateway._resolve_session_agent_runtime = resolve
    ctx = TurnContext(
        native_cot=cot,
        source=SimpleNamespace(platform=Platform.FEISHU, chat_id="oc"),
        _run_still_current=lambda: True,
    )
    turn = TurnRunner(gateway, ctx)
    result = turn.run_sync()
    await turn.finish_native_cot(result)
    await asyncio.gather(*gateway._background_tasks)
    assert result.get("failed") is True
    assert result.get("completed") is False
    assert result.get("error")
    assert transport.calls[-1][1].endswith("reason=error")


@pytest.mark.asyncio
async def test_events_received_during_put_get_next_batch():
    entered, release, delivered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    transport = FakeCOTClient()

    async def request(method, path, body=None):
        if method == "PUT" and not entered.is_set():
            entered.set()
            await release.wait()
        elif method == "PUT":
            delivered.set()
        return await transport.request_json(method, path, body)

    client = FeishuCOTClient("app", "secret", "feishu", request_json=request)
    run = await client.start("oc", None, "brief", flush_interval=0.01)
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.to_thread(run.commentary, "arrived-during-put")
        release.set()
        await asyncio.wait_for(delivered.wait(), 2)
    finally:
        release.set()
        await run.finish()
    assert "arrived-during-put" in json.dumps(transport.calls)


@pytest.mark.asyncio
async def test_plaintext_query_and_tool_preview_redact_named_credentials():
    transport = FakeCOTClient()
    client = FeishuCOTClient(
        "app", "secret", "feishu", request_json=transport.request_json
    )
    query = 'app_secret="ordinary secret words" cookie=ordinary-cookie'
    from plugins.platforms.feishu.cot import _safe_text

    for field in ("x-api-key", "api_key", "private_key", "private-key",
                  "access_key", "access-key", "x-private-key", "x_access_key",
                  "app_secret", "credentials", "password", "token",
                  "authorization", "cookie"):
        assert "plain-value" not in _safe_text(f"{field}=plain-value", 1200)
    assert "visible" in _safe_text("key=visible", 1200)
    run = await client.start("oc", None, "detailed", input_preview=query)
    run.tool_started("call", "web_search", {"query": query})
    run.tool_completed("call", "web_search", {}, query)
    await run.finish()
    bodies = json.dumps(transport.calls)
    assert "ordinary secret words" not in bodies
    assert "secret words" not in bodies  # a truncated preview must not leak a suffix
    assert "ordinary-cookie" not in bodies


@pytest.mark.asyncio
async def test_cleanup_can_cancel_finalizer_before_it_starts():
    transport = FakeCOTClient()
    client = FeishuCOTClient(
        "app", "secret", "feishu", request_json=transport.request_json
    )
    cot = await client.start("oc", None, "brief", flush_interval=60)
    await asyncio.sleep(0)
    gateway = object.__new__(GatewayRunner)
    turn = TurnRunner(
        gateway, TurnContext(native_cot=cot, _run_still_current=lambda: True)
    )
    await turn.finish_native_cot({})
    tasks = list(gateway._background_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)
    assert cot._flush_task is None
    assert not gateway._background_tasks
    assert not client._finalizer_tasks
    assert not client._active_runs


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
async def test_local_turn_returns_final_or_queue_followup_while_complete_is_slow(
    queued, monkeypatch
):
    from unittest.mock import AsyncMock
    transport = FakeCOTClient()
    entered, release = asyncio.Event(), asyncio.Event()

    async def request(method, path, body=None):
        if "/complete/" in path:
            entered.set()
            await release.wait()
        return await transport.request_json(method, path, body)

    client = FeishuCOTClient("app", "secret", "feishu", request_json=request)
    adapter = SimpleNamespace(start_native_cot=client.start, send=AsyncMock())
    gateway = object.__new__(GatewayRunner)
    gateway.adapters = {Platform.FEISHU: adapter}
    gateway._adapter_for_source = lambda _: adapter
    gateway._get_proxy_url = lambda: None
    gateway._resolve_turn_toolsets = lambda *_: ([], [])
    gateway._run_still_current_fn = lambda *_: lambda: True
    gateway._run_agent_bind_turn_wiring = lambda *_: None
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "display": {
                "tool_progress": "off",
                "platforms": {"feishu": {"cot_messages": "brief"}},
            }
        },
    )
    gateway._run_agent_start_streaming_tts = lambda *_: None
    gateway._run_agent_start_turn_worker = lambda *_: SimpleNamespace(
        executor_task=None
    )
    gateway._run_agent_await_turn_worker = AsyncMock(
        return_value={"final_response": "ordinary final"}
    )
    gateway._run_agent_evict_on_fallback = lambda *_: None
    gateway._run_agent_finalize_streaming_tts = AsyncMock()
    gateway._run_agent_drain_pending = AsyncMock(
        return_value=(None, "queued" if queued else None)
    )
    gateway._run_agent_queued_followup = AsyncMock(
        return_value={"final_response": "queue result"}
    )
    gateway._run_agent_schedule_bubble_cleanup = lambda *_: None
    gateway._run_agent_mark_streamed_delivery = AsyncMock()
    gateway._draining = False
    for name in (
        "_run_agent_stream_consumer_task",
        "_run_agent_track_agent",
        "_run_agent_monitor_for_interrupt",
        "_run_agent_notify_long_running",
    ):
        setattr(gateway, name, AsyncMock())

    source = SessionSource(platform=Platform.FEISHU, chat_id="oc")
    result = await asyncio.wait_for(
        gateway._run_agent_inner("query", "", [], source, "sid"), 2
    )
    await adapter.send(source.chat_id, result["final_response"])
    adapter.send.assert_awaited_once_with(
        "oc", "queue result" if queued else "ordinary final"
    )
    assert gateway._run_agent_queued_followup.await_count == int(queued)
    await asyncio.wait_for(entered.wait(), 2)
    assert not any("/complete/" in path for _, path, _ in transport.calls)
    release.set()
    await asyncio.gather(*gateway._background_tasks)


@pytest.mark.asyncio
async def test_client_close_reclaims_inflight_flush_and_prevents_later_requests():
    entered, cancelled = asyncio.Event(), asyncio.Event()
    transport = FakeCOTClient()

    async def request(method, path, body=None):
        if method == "PUT":
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return await transport.request_json(method, path, body)

    client = FeishuCOTClient("app", "secret", "feishu", request_json=request)
    cot = await client.start("oc", None, "brief", flush_interval=0.01)
    await asyncio.wait_for(entered.wait(), 2)
    await client.close()
    assert cancelled.is_set()
    calls_before = list(transport.calls)
    await cot.finish()
    assert await client.start("oc", None, "brief") is None
    assert transport.calls == calls_before
    assert cot._flush_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "json", "shape", "api", "timeout"])
async def test_http_errors_are_sanitized(failure, monkeypatch, caplog):
    from plugins.platforms.feishu.cot import FeishuCOTClient, FeishuCOTError

    secret = "ordinary-private-credential"

    async def send(self, request):
        if failure == "timeout":
            raise httpx.ReadTimeout(secret)
        return httpx.Response(
            403 if failure == "http" else 200,
            content=secret if failure in {"http", "json"} else json.dumps(
                [] if failure == "shape" else {"code": secret, "msg": secret}
            ),
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", send)
    client = FeishuCOTClient("app", secret, "feishu")
    client._base_url = "http://127.0.0.1:1"
    try:
        with pytest.raises(FeishuCOTError) as error:
            await client.request_json("PUT", "/cot", {})
        client._log_failure("test", error.value)
        assert secret not in str(error.value) + caplog.text
        assert error.value.__cause__ is None
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_http_cancellation_propagates_to_real_transport(timeout, monkeypatch):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def send(self, request):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", send)
    client = FeishuCOTClient("app", "secret", "feishu")
    client._base_url = "http://127.0.0.1:1"
    task = asyncio.create_task(client._http_json("POST", "/cot", {}))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if timeout:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(task, 0.01)
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert task.cancelled()
        assert cancelled.is_set()
    finally:
        await client.close()
