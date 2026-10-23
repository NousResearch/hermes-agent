"""``inject_message(origin=...)`` under the session authority answers ``False`` up front for a chat
the FIFO cannot admit it into (no human message admitted there yet), instead of ``True`` for a
message the dispatch then drops (PLG-6). Real GatewayRunner, DiscordAdapter and SessionAuthority.
"""
import asyncio

import pytest
import hermes_yaml as yaml

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tests.waiting import async_wait_until

PLUGIN = "desk"


def _context(home) -> tuple[PluginContext, PluginManager]:
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"entries": {PLUGIN: {"allow_gateway_injection": True}}},
    }), encoding="utf-8")
    manager = PluginManager(scope_key=str(home))
    return PluginContext(PluginManifest(name=PLUGIN, key=PLUGIN, source="user"), manager), manager


def _publish(runner, manager) -> None:
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    manager.set_gateway_message_injector(runner, runner._schedule_plugin_message_injection)


def _plugin_rows(authority, session_id, text):
    from hermes_state_runtime import list_session_admissions
    return [row for row in list_session_admissions(authority.db, session_id=session_id, pending_only=False)
            if row["payload"].get("text") == text]


@pytest.mark.asyncio
async def test_origin_injection_is_refused_up_front_until_the_chat_holds_a_human_turn(monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from hermes_constants import get_hermes_home
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")
    monkeypatch.setattr(adapter, "send", send)
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner")
    runner._wire_adapter_handlers(adapter)
    executed = []

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        captured = execution_result.get()
        if captured is not None:
            captured["result"] = {"final_response": "reply", "messages": []}
        return "reply"
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)

    ctx, manager = _context(get_hermes_home())
    _publish(runner, manager)
    origin = {"platform": "discord", "chat_id": "42", "chat_type": "dm", "user_id": "42"}
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    key = runner.session_store._generate_session_key(source)

    # No session in that chat yet: the FIFO cannot admit a producer turn there.
    assert await asyncio.to_thread(ctx.inject_message, "reminder", origin=origin) is False
    assert runner.session_store.lookup_by_session_key(key) is None

    # A session row with no committed human turn has no connector provenance to borrow either.
    runner.session_store.get_or_create_session(source)
    assert await asyncio.to_thread(ctx.inject_message, "reminder", origin=origin) is False

    # Once a human message was admitted there, the plugin turn joins the chat's durable FIFO.
    await adapter.handle_message(MessageEvent(text="A", source=source, message_id="A"))
    await async_wait_until(lambda: "A" in executed, timeout=10, what="the human turn")
    assert await asyncio.to_thread(ctx.inject_message, "reminder", origin=origin) is True
    owner = authority.logical_owner(runner.session_store.lookup_by_session_key(key).session_id)
    await async_wait_until(lambda: _plugin_rows(authority, owner, "reminder"), timeout=10,
                           what="the plugin admission")
