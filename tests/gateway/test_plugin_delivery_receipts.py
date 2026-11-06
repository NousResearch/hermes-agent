"""Stable producer identities and transport acknowledgements for plugin-owned initiatives."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tests.gateway.test_plugin_message_injection import _entry, _runner, _RoutingAdapter


@pytest.mark.asyncio
async def test_context_reference_reaches_durable_admission(tmp_path, monkeypatch):
    import gateway.session_authorities as authorities

    entry = _entry()
    adapter = _RoutingAdapter()
    runner = _runner(entry, adapter)
    runner._gateway_loop = asyncio.get_running_loop()
    authority = SimpleNamespace(admit_automation=AsyncMock())
    monkeypatch.setattr(authorities, "active_authority", lambda _: authority)
    manager = PluginManager()
    manager.set_gateway_message_injector(runner, runner._schedule_plugin_message_injection)
    ctx = PluginContext(PluginManifest(name="initiatives", key="initiatives", source="user"), manager)
    monkeypatch.setattr(ctx, "_gateway_injection_allowed", lambda: True)
    for _ in range(2):
        assert ctx.inject_message("Check status", session_key=entry.session_key, message_ref="occurrence-7")
        await asyncio.gather(*runner._background_tasks)
    calls = authority.admit_automation.await_args_list
    first, second = (call.args[2] for call in calls)
    assert first == second
    assert json.loads(first) == ["plugin", "initiatives", entry.session_key, "occurrence-7"]
    assert calls[0].args[1].message_id == first
    assert ctx.inject_message("Check status", session_key=entry.session_key)
    await asyncio.gather(*runner._background_tasks)
    assert authority.admit_automation.await_args.args[2] != first


@pytest.mark.asyncio
@pytest.mark.parametrize("success,error", [(True, None), (False, "permission denied")])
async def test_final_text_receipt_reports_transport_result(monkeypatch, success, error):
    from gateway.platforms.event import MessageEvent
    from hermes_cli import lifecycle

    entry = _entry()
    adapter = _RoutingAdapter()
    reference = json.dumps(["plugin", "initiatives", entry.session_key, "occurrence-7"])
    event = MessageEvent(text="Check status", source=entry.origin, message_id=reference)
    result = SendResult(success=success, message_id="platform-42" if success else None, error=error)
    adapter._record_delivery_obligation = AsyncMock(return_value=None)
    adapter._send_with_retry = AsyncMock(return_value=result)
    observed = AsyncMock()
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "post_message_delivered")
    monkeypatch.setattr(lifecycle, "ainvoke_hook", observed)
    returned, owner = await adapter.send_final_ledgered(event, entry.session_key, "The answer", {}, reply_to=None)
    assert returned is result and owner is adapter
    observed.assert_awaited_once_with(
        "post_message_delivered", session_key=entry.session_key, message_ref="occurrence-7",
        platform="telegram", platform_message_id=result.message_id, success=success, error=error)
    adapter._send_with_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_reference_retries_deduplicate_real_fifo_and_reject_changed_content(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session_ingress_context import native_callback, register_transport_home
    from hermes_state_runtime import list_session_admissions
    from tests.gateway.test_automation_notification_category import _authority

    runner, authority = await _authority(tmp_path, monkeypatch)
    adapter = _RoutingAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    human = MessageEvent(text="human", source=source, message_id="human")
    try:
        with native_callback(runner, human, tmp_path):
            await authority.admit_native(human)
        route = runner.session_store._generate_session_key(source)
        sid = runner.session_store.peek_session_id(route)
        async def inject(text, plugin="initiatives"):
            return await runner._admit_plugin_injection(
                adapter, source, session_key=route, session_id=sid, content=text,
                plugin_id=plugin, message_ref="occurrence-7")
        assert await inject("Check status")
        assert await inject("Check status")
        assert not await inject("Changed payload")
        assert await inject("Check status", plugin="another-plugin")
        rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
        assert [row["payload"]["text"] for row in rows] == ["human", "Check status", "Check status"]
    finally:
        authority.db.close()
