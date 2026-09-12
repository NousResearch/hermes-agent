"""Independent source-port reproductions; synthetic transport and state only."""

import logging
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


PLATFORMS = [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD, Platform.TELEGRAM]
PHONE = "15551234567"
PRIVATE = "private provider response body"


def context(platform):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    source = SessionSource(platform=platform, chat_type="dm", chat_id=PHONE, user_id=PHONE)
    key = build_session_key(source)
    adapter = NS(platform=platform)
    return runner, source, key, adapter


def scoped_logs(caplog, platform, *, traceback=False, identity=True):
    if platform in {Platform.WHATSAPP, Platform.WHATSAPP_CLOUD}:
        assert PRIVATE not in caplog.text
        if identity:
            assert PHONE not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)
    else:
        assert PRIVATE in caplog.text
        if identity:
            assert PHONE in caplog.text
        if traceback:
            assert any(record.exc_info for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", PLATFORMS)
async def test_shutdown_notice_failed_result(platform, caplog):
    runner, source, key, adapter = context(platform)
    adapter.send = AsyncMock(return_value=NS(success=False, error=PRIVATE))
    metadata = {"thread_id": "raw-thread"}
    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        result = await runner._send_shutdown_notice(
            adapter, PHONE, "raw shutdown message", "active chat", platform.value, metadata=metadata
        )
    assert result is False
    adapter.send.assert_awaited_once_with(PHONE, "raw shutdown message", metadata=metadata)
    scoped_logs(caplog, platform)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", PLATFORMS)
async def test_home_channel_exception(platform, caplog):
    runner, source, key, adapter = context(platform)
    adapter.send = AsyncMock(side_effect=ValueError(PRIVATE))
    runner._thread_metadata_for_target = lambda *a, **k: None
    home = NS(chat_id=PHONE, thread_id=None)
    transport = NS(adapter=adapter, is_relay=False)
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        result = await runner._send_home_channel_message(
            platform, home, transport, "raw home message", "Home-channel startup notification failed for %s:%s: %s"
        )
    assert result is False
    adapter.send.assert_awaited_once_with(PHONE, "raw home message")
    scoped_logs(caplog, platform)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", PLATFORMS)
async def test_private_notice_fallback_exception(platform, caplog):
    runner, source, key, adapter = context(platform)
    runner.config = NS(get_notice_delivery=lambda p: "private")
    runner._adapter_for_source = lambda s: adapter
    metadata = {"thread_id": "raw-thread"}
    runner._thread_metadata_for_source = lambda s: metadata
    adapter.send_private_notice = AsyncMock(side_effect=ValueError(PRIVATE))
    adapter.send = AsyncMock(return_value=NS(success=True))
    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        await runner._deliver_platform_notice(source, "raw platform notice")
    adapter.send_private_notice.assert_awaited_once_with(PHONE, PHONE, "raw platform notice", metadata=metadata)
    adapter.send.assert_awaited_once_with(PHONE, "raw platform notice", metadata=metadata)
    scoped_logs(caplog, platform, traceback=True, identity=False)




@pytest.mark.asyncio
@pytest.mark.parametrize("platform", PLATFORMS)
async def test_deferred_goal_status_exception(platform, caplog):
    runner, source, key, adapter = context(platform)
    runner._adapter_for_source = lambda s: adapter
    runner._session_key_for_source = lambda s: key
    runner._thread_metadata_for_source = lambda s: None
    adapter.send = AsyncMock(side_effect=ValueError(PRIVATE))
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await runner._defer_goal_status_notice_after_delivery(source, "raw goal state")
    adapter.send.assert_awaited_once_with(PHONE, "raw goal state", metadata=None)
    scoped_logs(caplog, platform, traceback=True, identity=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["result", "exception"])
@pytest.mark.parametrize("platform", PLATFORMS)
async def test_reconcile_failed_edit(platform, failure, caplog):
    runner, source, key, adapter = context(platform)
    adapter.edit_message = AsyncMock(return_value=NS(success=False, error=PRIVATE))
    if failure == "exception":
        adapter.edit_message.side_effect = ValueError(PRIVATE)
    consumer = NS(adapter=adapter, message_id="raw-message-id")
    response = {"final_response": "raw final response"}
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await runner._run_agent_edit_streamed_message(
            consumer, source, response, "raw final response", _sk=key,
            ok=("Reconciled session %s", key),
            fail_result="Stale-finalize reconciliation edit failed for session %s (%s)",
            fail_exc="Stale-finalize reconciliation edit failed for session %s: %s",
        )
    assert response == {"final_response": "raw final response"}
    adapter.edit_message.assert_awaited_once_with(
        chat_id=PHONE, message_id="raw-message-id", content="raw final response", finalize=True
    )
    scoped_logs(caplog, platform)
