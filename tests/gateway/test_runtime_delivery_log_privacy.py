"""Delivery outcomes and exact destinations survive scoped diagnostic copies."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD, Platform.TELEGRAM])
@pytest.mark.parametrize("outcome", ["sent", "failed", "raised"])
async def test_shutdown_send_logs_keep_raw_transport_and_outcome(platform, outcome, caplog):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(platform=platform)
    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=outcome == "sent", error="private response body"),
        side_effect=RuntimeError("private response body") if outcome == "raised" else None,
    )
    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        delivered = await runner._send_shutdown_notice(
            adapter, "15551234567", "exact notice", "active", platform.value,
            metadata={"thread_id": "raw-thread"},
        )
    assert delivered == (outcome == "sent")
    adapter.send.assert_awaited_once_with("15551234567", "exact notice", metadata={"thread_id": "raw-thread"})
    assert ("15551234567" in caplog.text) == (platform == Platform.TELEGRAM)
    if outcome != "sent":
        assert ("private response body" in caplog.text) == (platform == Platform.TELEGRAM)


@pytest.mark.parametrize("platform", [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD, Platform.TELEGRAM])
@pytest.mark.parametrize("with_traceback", [False, True])
@pytest.mark.parametrize("selector", ["platform", "session_key"])
def test_suppressed_failure_preserves_branch_and_scopes_error(platform, with_traceback, selector, caplog):
    from gateway.run_shutdown import _log_suppressed
    from gateway.session import SessionSource, build_session_key

    key = build_session_key(SessionSource(platform=platform, chat_id="15551234567"))
    options = {selector: platform if selector == "platform" else key}
    fmt = "cleanup failed" if with_traceback else "cleanup failed: %s"
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        with _log_suppressed(logging.WARNING, fmt, exc_info=with_traceback, **options):
            raise ValueError("private response body")
    if platform == Platform.TELEGRAM:
        assert "private response body" in caplog.text
        assert bool(caplog.records[-1].exc_info) == with_traceback
    else:
        assert "private response body" not in caplog.text
        assert caplog.records[-1].exc_info is None
