"""Runtime routing keeps raw values while diagnostic copies are scoped."""

import logging
from collections import OrderedDict

import pytest

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


@pytest.mark.parametrize(
    "platform", [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD, Platform.TELEGRAM]
)
def test_failed_source_cache_keeps_routing_inputs_and_scopes_traceback(
    platform, monkeypatch, caplog
):
    from gateway.run import GatewayRunner

    source = SessionSource(
        platform=platform, chat_type="dm", chat_id="15551234567", user_id="15551234567"
    )
    key = build_session_key(source)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._session_sources = OrderedDict()
    seen = []

    def fail_replace(value):
        seen.append(value)
        raise ValueError("private provider payload")

    monkeypatch.setattr("gateway.run.dataclasses.replace", fail_replace)
    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        runner._cache_session_source(key, source)

    assert seen == [source]
    assert not runner._session_sources
    record = next(
        r for r in caplog.records if "Failed to cache live session source" in r.message
    )
    if platform in {Platform.WHATSAPP, Platform.WHATSAPP_CLOUD}:
        assert "15551234567" not in record.message
        assert "private provider payload" not in caplog.text
        assert record.exc_info is None
    else:
        assert key in record.message
        assert "private provider payload" in caplog.text
        assert record.exc_info is not None


@pytest.mark.parametrize(
    "platform", [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD, Platform.TELEGRAM]
)
def test_startup_restore_preserves_original_event_and_scopes_chat_log(platform, caplog):
    from gateway.run import GatewayRunner
    from gateway.platforms.base import MessageEvent

    source = SessionSource(
        platform=platform, chat_type="dm", chat_id="15551234567", user_id="15551234567"
    )
    event = MessageEvent(text="private inbound text", source=source)
    runner = GatewayRunner.__new__(GatewayRunner)
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        runner._queue_startup_restore_event(event)
    assert runner._startup_restore_queue == [event]
    assert runner._startup_restore_queue[0] is event
    assert event.text == "private inbound text"
    assert source.chat_id == "15551234567"
    if platform in {Platform.WHATSAPP, Platform.WHATSAPP_CLOUD}:
        assert "15551234567" not in caplog.text
    else:
        assert "15551234567" in caplog.text
    assert "private inbound text" not in caplog.text


@pytest.mark.parametrize(
    "platform", [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD, Platform.TELEGRAM]
)
def test_profile_matcher_receives_raw_source_and_scopes_failure_traceback(
    platform, monkeypatch, caplog
):
    from types import SimpleNamespace
    from gateway.run import GatewayRunner

    source = SessionSource(
        platform=platform, chat_type="dm", chat_id="15551234567", user_id="15551234567"
    )
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=True, profile_routes=[object()])
    calls = []

    def match(*args, **kwargs):
        calls.append((args, kwargs))
        raise ValueError("private route failure payload")

    monkeypatch.setattr("gateway.profile_routing.match_profile_route", match)
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        result = runner._profile_name_for_source(source)
    assert result is None
    assert calls[0][1]["chat_id"] == source.chat_id
    assert calls[0][1]["platform"] == platform.value
    if platform in {Platform.WHATSAPP, Platform.WHATSAPP_CLOUD}:
        assert "15551234567" not in caplog.text
        assert "private route failure payload" not in caplog.text
    else:
        assert "15551234567" in caplog.text
        assert "private route failure payload" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD, Platform.TELEGRAM])
@pytest.mark.parametrize("operation", ["steer", "redirect"])
async def test_priority_failure_keeps_raw_instruction_and_fallback(platform, operation, caplog):
    from unittest.mock import Mock
    from gateway.run import GatewayRunner
    from gateway.platforms.base import MessageEvent

    source = SessionSource(platform=platform, chat_id="15551234567", user_id="15551234567")
    key = build_session_key(source)
    event = MessageEvent(text="exact private steering instruction", source=source)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._queue_or_replace_pending_event = Mock()
    agent = Mock()
    agent._supports_active_turn_redirect = True
    method = getattr(agent, operation)
    method.side_effect = ValueError("private route failure payload")
    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        if operation == "steer":
            runner._hm_busy_steer(event, agent, key)
            runner._queue_or_replace_pending_event.assert_called_once_with(key, event)
        else:
            await runner._hm_busy_interrupt(event, source, agent, key)
            agent.interrupt.assert_called_once_with(event.text)
    method.assert_called_once()
    # Current main prefixes priority instructions with trusted origin metadata; the original
    # instruction must still reach the agent unchanged at the end of that context.
    assert method.call_args.args[0].endswith(event.text)
    if platform == Platform.TELEGRAM:
        assert key in caplog.text
        assert "private route failure payload" in caplog.text
    else:
        assert "15551234567" not in caplog.text
        assert "private route failure payload" not in caplog.text
