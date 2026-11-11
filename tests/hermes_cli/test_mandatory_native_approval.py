"""Mandatory plugin consent through real SDK cards and authorized callback routers.

Only outbound delivery and human input are simulated. The plugin, dispatcher,
observers, middleware, gateway queue and native renderers are real.
"""
import asyncio
import html
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from hermes_cli import plugins
from tests.waiting import async_wait_until
from tools import approval, approval_context, approval_gateway_wait


class NativeConsent:
    def __init__(self, platform, loop):
        self.platform, self.loop = platform, loop
        self.session = "agent:main:consent:dm:123"
        self.cards, self.texts, self.executed, self.observed, self.middleware = [], [], [], [], []
        self.failed = False
        if platform == "telegram":
            from plugins.platforms.telegram.adapter import TelegramAdapter
            self.adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
        else:
            from plugins.platforms.discord.adapter import DiscordAdapter
            self.adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
            self.adapter._allowed_user_ids = {"123"}
        self.adapter.pause_typing_for_chat = Mock()
        self.adapter.resume_typing_for_chat = Mock()
        self.adapter.edit_message = AsyncMock(return_value=SendResult(success=True))
        self.adapter._edit_md_quiet = AsyncMock()
        self.adapter._send_prompt = self.deliver
        self.adapter.send = self.send
        ctx = NS(_status_adapter=self.adapter, _status_chat_id="123", session_key=self.session,
                 _status_thread_metadata={}, _run_still_current=lambda: True)
        self.runner = NS(_ctx=ctx, _schedule=self.schedule, _close_native_stream_boundary=lambda _: None)

    def schedule(self, coro, label):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def notify(self, data):
        TurnRunner._render_approval_sync(self.runner, data)

    async def send(self, chat_id, content, **kwargs):
        self.texts.append(content)
        return SendResult(success=True, message_id="text")

    async def deliver(self, *args, **kwargs):
        if self.failed:
            return SendResult(success=False, error="simulated native delivery failure")
        build = args[3] if self.platform == "telegram" else args[2]
        if self.platform == "telegram":
            text, keyboard, on_sent = build()
            on_sent(NS(message_id=len(self.cards) + 1))
            buttons = keyboard.to_dict()["inline_keyboard"]
            self.cards.append(NS(text=html.unescape(text), buttons=buttons))
        else:
            payload, view = build(None)
            message = NS(id=len(self.cards) + 1, embeds=[payload["embed"]], edit=AsyncMock())
            view._message = message
            self.cards.append(NS(text=payload["content"], view=view, message=message))
            assert len(view.to_components()[0]["components"]) == 2
        return SendResult(success=True, message_id=str(len(self.cards)))

    async def tap(self, index, choice="once", actor=123, missing_id=False):
        card = self.cards[index]
        if self.platform == "telegram":
            button = next(b for row in card.buttons for b in row if b["callback_data"].startswith("ea:" + choice + ":"))
            if missing_id:
                key = int(button["callback_data"].rsplit(":", 1)[1])
                self.adapter._approval_request_ids.pop(key, None)
            query = NS(data=button["callback_data"], from_user=NS(id=actor, first_name="Operator"),
                       message=NS(chat_id=123, chat=NS(type="private"), message_thread_id=None), answer=AsyncMock())
            await self.adapter._handle_callback_query(NS(callback_query=query), None)
        else:
            if missing_id:
                card.view.request_id = None
            interaction = NS(user=NS(id=actor, display_name="Operator", roles=[]), message=card.message,
                             response=NS(edit_message=AsyncMock(), send_message=AsyncMock(), is_done=lambda: False))
            button = card.view.allow_once if choice == "once" else card.view.deny
            await button.callback(interaction)

    def dispatch(self, label):
        import model_tools
        token = approval_context.set_current_session_key(self.session)
        try:
            return json.loads(model_tools.handle_function_call(
                "consent_dummy", {"label": label}, session_id="session", task_id="task",
                turn_id="turn", tool_call_id="call-" + label))
        finally:
            approval_context.reset_current_session_key(token)

    async def start(self, label, count):
        task = asyncio.create_task(asyncio.to_thread(self.dispatch, label))
        await async_wait_until(lambda: len(self.cards) == count or task.done(), what="native card")
        assert len(self.cards) == count, task.result() if task.done() else "no card"
        return task


@pytest_asyncio.fixture(params=["telegram", "discord"])
async def native(request, monkeypatch, tmp_path):
    platform = request.param
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", platform)
    monkeypatch.setenv("HERMES_SESSION_ACTOR_ID", "123")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "dm")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "123")
    (tmp_path / "config.yaml").write_text("approvals:\n  mode: 'off'\n  approvers: []\n")
    manager = plugins.PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    n = NativeConsent(platform, asyncio.get_running_loop())
    ctx = plugins.PluginContext(plugins.PluginManifest(name="consent-policy", source="test"), manager)
    ctx.register_hook("pre_tool_call", lambda **kw: {
        "action": "approve", "approval_policy": "mandatory", "rule_key": "dummy:write",
        "message": "Confirm exact dummy operation " + kw["args"]["label"],
    })
    ctx.register_hook("pre_approval_request", lambda **kw: n.observed.append(("pre", kw)))
    ctx.register_hook("post_approval_response", lambda **kw: n.observed.append(("post", kw)))

    def middleware(args, next_call):
        n.middleware.append(args["label"])
        return next_call(args)

    def handler(args, **kwargs):
        n.executed.append(args["label"])
        return json.dumps({"ok": args["label"]})

    ctx.register_middleware("tool_execution", middleware)
    ctx.register_tool("consent_dummy", "consent_test", {
        "name": "consent_dummy", "description": "Dummy operation", "parameters": {"type": "object"},
    }, handler)
    approval.register_gateway_notify(n.session, n.notify)
    try:
        yield n
    finally:
        approval.unregister_gateway_notify(n.session)
        manager.unload()
        for card in n.cards:
            if platform == "discord":
                card.view.stop()
        await asyncio.sleep(0)
        if platform == "discord" and n.adapter._background_tasks:
            await asyncio.gather(*n.adapter._background_tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("labels", [("A", "B"), ("same", "same")])
async def test_native_second_card_executes_only_second(native, labels):
    n = native
    first, second = labels
    a = await n.start(first, 1)
    b = await n.start(second, 2)
    try:
        assert json.dumps({"label": second}) in n.cards[1].text
        await n.tap(1)
        assert (await asyncio.wait_for(b, 10))["ok"] == second
        assert not a.done()
        assert n.executed == n.middleware == [second]
        assert [kw["choice"] for event, kw in n.observed if event == "post"] == ["once"]
    finally:
        approval.unregister_gateway_notify(n.session)
        await asyncio.gather(a, b)


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", ["expired", "missing", "wrong_actor"])
async def test_invalid_native_answer_never_selects_pending_request(native, monkeypatch, stale):
    n = native
    with monkeypatch.context() as m:
        m.setattr(approval_gateway_wait, "_poll_event", lambda *a, **kw: "timeout")
        assert "error" in await asyncio.to_thread(n.dispatch, "A")
    b = await n.start("B", 2)
    try:
        await n.tap(1 if stale == "wrong_actor" else 0, actor=999 if stale == "wrong_actor" else 123,
                    missing_id=stale == "missing")
        assert not b.done()
        assert n.executed == n.middleware == []
        await n.tap(1)
        assert (await asyncio.wait_for(b, 10))["ok"] == "B"
    finally:
        approval.unregister_gateway_notify(n.session)
        await asyncio.gather(b)


@pytest.mark.asyncio
@pytest.mark.parametrize("resolve_all", [False, True])
async def test_generic_session_answer_cannot_grant_mandatory(native, resolve_all):
    n = native
    a = await n.start("A", 1)
    try:
        assert approval.resolve_gateway_approval(n.session, "once", resolve_all=resolve_all) == 0
        assert not a.done()
        assert not n.executed
        await n.tap(0)
        assert (await asyncio.wait_for(a, 10))["ok"] == "A"
    finally:
        approval.unregister_gateway_notify(n.session)
        await asyncio.gather(a)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "no_native"])
async def test_failed_native_delivery_does_not_arm_typed_fallback(native, failure):
    n = native
    n.failed = True
    if failure == "no_native":
        n.runner._ctx._status_adapter = NS(pause_typing_for_chat=Mock(), send=n.send)
    task = asyncio.create_task(asyncio.to_thread(n.dispatch, "A"))
    await async_wait_until(lambda: task.done() or n.texts, what="delivery result")
    try:
        assert not n.texts
        assert "error" in await asyncio.wait_for(task, 10)
        assert approval.list_gateway_approvals(n.session) == []
        assert n.executed == n.middleware == []
    finally:
        approval.unregister_gateway_notify(n.session)
        await asyncio.gather(task)
