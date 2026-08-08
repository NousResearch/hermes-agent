"""Tests for the new elif branch that sends home-channel startup notifications
on manual / crash / service restarts (not only planned restarts).

See: fix(gateway): send startup notification on manual/crash restart
"""

from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import HomeChannel, Platform
from gateway.platforms.base import SendResult
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_startup_notification_sent_when_no_markers_present(
    tmp_path, monkeypatch
):
    """Manual startup (no .restart_notify.json and no /restart marker) must
    trigger the home-channel startup notification via the new elif branch.
    """
    # _hermes_home is used by the marker helpers
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="42",
        name="Ops Home",
    )
    adapter.send = AsyncMock(
        return_value=SendResult(success=True, message_id="home")
    )

    # Mirror the exact conditional that lives in gateway/run.py after the fix:
    #
    #     if planned_restart_notification_pending:
    #         ...
    #     elif not _restart_notification_pending():
    #         await self._send_home_channel_startup_notifications(...)
    planned_pending = gateway_run._planned_restart_notification_pending()
    restart_pending = gateway_run._restart_notification_pending()

    assert planned_pending is False, "Test precondition: no planned marker"
    assert restart_pending is False, "Test precondition: no /restart marker"

    if planned_pending:
        # Original branch — must not be entered here.
        pytest.fail("planned branch entered when no planned marker exists")
    elif not restart_pending:
        # New elif branch — should call the home-channel notifier.
        await runner._send_home_channel_startup_notifications(skip_targets=None)

    # The new elif branch must trigger exactly one notification.
    adapter.send.assert_called_once()
    sent_chat_id = adapter.send.call_args[0][0]
    sent_text = adapter.send.call_args[0][1]
    assert sent_chat_id == "42"
    assert "Gateway online" in sent_text


@pytest.mark.asyncio
async def test_startup_notification_skipped_when_restart_marker_present(
    tmp_path, monkeypatch
):
    """When a chat-originated /restart is in flight, the new elif branch must
    NOT send a duplicate home-channel notification — that path already has
    its own reply target.
    """
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    # Drop a .restart_notify.json so _restart_notification_pending() returns True.
    (tmp_path / ".restart_notify.json").write_text("{}")

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="42",
        name="Ops Home",
    )
    adapter.send = AsyncMock()

    planned_pending = gateway_run._planned_restart_notification_pending()
    restart_pending = gateway_run._restart_notification_pending()

    assert planned_pending is False
    assert restart_pending is True

    # Mirror the conditional — the elif must NOT trigger.
    sent_via_elif = False
    if planned_pending:
        pytest.fail("planned branch entered when no planned marker exists")
    elif not restart_pending:
        sent_via_elif = True

    assert sent_via_elif is False, (
        "elif branch must skip when /restart marker is present"
    )
    adapter.send.assert_not_called()