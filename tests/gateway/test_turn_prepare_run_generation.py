"""Turn preparation must carry logical Stop identity into registered plugin tools."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource
from gateway.session_context import clear_session_vars, get_session_run_generation
from tools.registry import ToolRegistry


@pytest.mark.asyncio
@pytest.mark.parametrize("generation", [3, 9])
async def test_prepared_turn_injects_its_logical_generation_into_plugin_tool(generation):
    runner = object.__new__(GatewayRunner)
    runner.config, runner.adapters = GatewayConfig(), {}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm",
                           user_id="u1", profile="default")
    key = runner._session_key_for_source(source)
    now = datetime.now(timezone.utc)
    entry = SessionEntry(key, "prepared-session", now, now, origin=source)
    event = MessageEvent(text="work", message_type=MessageType.TEXT, source=source)
    runner._hmwa_open_session = AsyncMock(return_value=(False, False))
    runner._pinned_session_context_prompt = MagicMock(return_value="context")
    runner._hmwa_acquire_turn_lease = AsyncMock()
    runner._mark_durable_active_turn = AsyncMock()
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, load_transcript=AsyncMock(return_value=[]))
    runner._hmwa_run_session_hygiene = AsyncMock(return_value=[])
    runner._hmwa_first_contact_notes = AsyncMock()
    runner._voice_channel_sidecar_note = MagicMock(return_value=None)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="work")
    runner._hmwa_apply_message_timestamp = lambda event, text: (text, text, None)
    runner._bind_adapter_run_generation = MagicMock()
    runner._delivery_adapter_for = lambda source: None
    seen = []

    def plugin(args, *, invocation_context):
        seen.append(invocation_context)
        return "{}"

    registry = ToolRegistry()
    registry.register(name="stop_origin_probe", toolset="core", handler=plugin,
                      schema={"name": "stop_origin_probe", "parameters": {"type": "object"}},
                      inject_invocation_context=True)
    assert get_session_run_generation() is None
    prepared, tokens = await runner._hmwa_prepare_turn(event, source, entry, key, key, generation)
    try:
        assert prepared.message_text == "work"
        registry.dispatch("stop_origin_probe", {"run_generation": generation + 100})
        assert len(seen) == 1
        assert (seen[0].session_key, seen[0].run_generation) == (key, generation)
        assert seen[0].profile == "default"
        runner._bind_adapter_run_generation.assert_called_once_with(None, key, generation)
    finally:
        clear_session_vars(tokens)
    assert get_session_run_generation() is None
