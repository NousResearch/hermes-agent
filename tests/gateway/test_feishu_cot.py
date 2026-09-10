"""Behavior contracts for Feishu native message_cot progress."""

import asyncio
import json
import queue
from types import SimpleNamespace

import pytest


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
    from plugins.platforms.feishu.cot import FeishuCOTClient

    transport = FakeCOTClient()
    client = FeishuCOTClient(
        "app", "secret", "feishu", request_json=transport.request_json
    )
    run = await client.start(
        "oc-chat", "om-origin", "brief", flush_interval=transport.flush_interval
    )
    assert run is not None

    run.tool_started("call-1", "terminal", {"command": "echo ok"})
    run.tool_completed(
        "call-1", "terminal", {"command": "echo ok"}, '{"exit_code":0,"output":"ok"}'
    )
    await run.finish("done")

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
    from plugins.platforms.feishu.cot import FeishuCOTClient

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
    from plugins.platforms.feishu.cot import FeishuCOTClient

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
    from plugins.platforms.feishu.cot import FeishuCOTClient

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
    from plugins.platforms.feishu.cot import FeishuCOTClient

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
    from plugins.platforms.feishu.cot import FeishuCOTClient

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
    if fail_stage == "update":
        assert transport.calls[-1][1] == "/open-apis/im/v1/message_cot"
    else:
        assert transport.calls[-1][1].endswith("reason=error")


def test_gateway_native_cot_uses_real_ids_and_suppresses_ordinary_progress():
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

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
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

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
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    adapter = SimpleNamespace(
        start_native_cot=lambda *_: pytest.fail("COT must stay off")
    )
    ctx = TurnContext(native_cot_mode="off", source=SimpleNamespace(chat_id="oc"))
    runner = TurnRunner(SimpleNamespace(_adapter_for_source=lambda _: adapter), ctx)

    await runner.start_native_cot()
    assert ctx.native_cot is None


@pytest.mark.asyncio
async def test_create_failure_falls_back_to_existing_tool_progress():
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

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
    runner.progress_callback("tool.started", "terminal", "echo ok", {"command": "echo ok"})

    assert ctx.native_cot is None
    assert ctx.progress_queue.get_nowait() == "⚙️ Running echo ok"


def test_native_cot_flush_interval_is_approximately_600ms():
    from plugins.platforms.feishu.cot import COT_FLUSH_INTERVAL_SECONDS

    assert COT_FLUSH_INTERVAL_SECONDS == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_tenant_token_is_cached_without_exposing_credentials():
    from plugins.platforms.feishu.cot import FeishuCOTClient

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
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

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
async def test_final_message_delivery_does_not_edit_cot_message_id():
    """The COT lifecycle has no access to the adapter's normal send/edit path."""
    from plugins.platforms.feishu.cot import FeishuCOTClient

    transport = FakeCOTClient()
    client = FeishuCOTClient(
        "app", "secret", "feishu", request_json=transport.request_json
    )
    run = await client.start(
        "oc-chat", "om-origin", "brief", flush_interval=transport.flush_interval
    )
    await run.finish("done")

    assert all("/messages/om-cot-1" not in path for _, path, _ in transport.calls)
    assert all(method != "PATCH" for method, _, _ in transport.calls)
