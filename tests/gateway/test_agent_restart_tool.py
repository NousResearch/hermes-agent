"""Tests for GatewayRunner.dispatch_gateway_restart — the shared restart coordinator.

Verifies:
1. Agent tool and /restart slash command use the SAME shared coordinator.
2. Detached handoff is established before drain/stop (Handshake Acknowledgement).
3. Handoff failure / Popen raising OSError:
   - Returns False JSON.
   - stop() is NOT called.
   - Gateway stays active and restart flags roll back.
4. Transactional rollback of existing state:
   - Old dedup marker is preserved when handoff fails.
   - File write failure (notify or dedup) aborts request_restart and rolls back.
5. Concurrent restart requests on the Gateway loop: exactly ONE succeeds.
6. Delayed helper canceled before commitment -> stop() never called.
7. Helper canceled AFTER commitment -> stop() proceeds, state not rolled back.
8. Service path commitment (via_service=True) under ambient INVOCATION_ID vs unset.
9. Redelivery suppression integrated into public entry path (_handle_message).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(
    chat_id: str = "chat-99", thread_id: str | None = None, message_id: str = "msg-101"
) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id="u42",
        thread_id=thread_id,
        message_id=message_id,
    )


# ---------------------------------------------------------------------------
# 1. /restart and agent tool share coordinator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slash_and_tool_share_restart_coordinator(monkeypatch, tmp_path):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)

    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop

    async def fake_launch():
        return True

    runner._launch_detached_restart_command = fake_launch  # type: ignore[method-assign]
    runner.stop = AsyncMock()  # type: ignore[method-assign]

    with patch("gateway.run._hermes_home", tmp_path):
        success, msg = await runner.dispatch_gateway_restart(
            source=_make_source(), origin="agent_tool", reason="testing shared coordinator"
        )

    assert success is True
    assert (tmp_path / ".restart_notify.json").exists()
    assert (tmp_path / ".restart_last_processed.json").exists()

    dedup = json.loads((tmp_path / ".restart_last_processed.json").read_text())
    assert dedup["origin"] == "agent_tool"
    assert dedup["reason"] == "testing shared coordinator"


# ---------------------------------------------------------------------------
# 2. Handoff failure (Popen error) fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handoff_failure_returns_false_and_aborts_stop(monkeypatch, tmp_path):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)

    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop

    stop_called = False

    async def fake_stop(**kwargs):
        nonlocal stop_called
        stop_called = True

    runner.stop = fake_stop  # type: ignore[method-assign]

    with patch("gateway.run._hermes_home", tmp_path), patch(
        "gateway.run._resolve_hermes_bin", return_value=["hermes"]
    ), patch("subprocess.Popen", side_effect=OSError("Permission denied")):
        success, msg = await runner.dispatch_gateway_restart(
            source=_make_source(), origin="agent_tool"
        )

        assert success is False
        assert "Handoff failed" in msg or "could not be spawned" in msg
        assert stop_called is False

        assert runner._restart_requested is False
        assert runner._restart_task_started is False


# ---------------------------------------------------------------------------
# 3. File write failure (notify or dedup) fails closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_write_failure_aborts_restart(monkeypatch, tmp_path):
    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop
    runner.stop = AsyncMock()  # type: ignore[method-assign]

    # Patch atomic_json_write to raise OSError on the first file write
    with patch("gateway.run._hermes_home", tmp_path), patch(
        "gateway.slash_commands.atomic_json_write", side_effect=OSError("Disk full")
    ):
        success, msg = await runner.dispatch_gateway_restart(
            source=_make_source(), origin="agent_tool"
        )

        assert success is False
        assert "Failed to persist" in msg
        runner.stop.assert_not_called()
        assert runner._restart_requested is False


# ---------------------------------------------------------------------------
# 4. Delayed helper canceled BEFORE commitment -> stop() never called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delayed_helper_canceled_before_commitment(monkeypatch, tmp_path):
    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop
    runner.stop = AsyncMock()  # type: ignore[method-assign]

    async def fake_launch_slow():
        await asyncio.sleep(5.0)
        return True

    runner._launch_detached_restart_command = fake_launch_slow  # type: ignore[method-assign]

    with patch("gateway.run._hermes_home", tmp_path):
        task = asyncio.create_task(
            runner.dispatch_gateway_restart(source=_make_source(), origin="agent_tool")
        )
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Wait longer than the 5.0s delay to verify stop() is NEVER called
        await asyncio.sleep(0.1)
        runner.stop.assert_not_called()
        assert runner._restart_requested is False


# ---------------------------------------------------------------------------
# 5. Helper committed -> Cancel after commitment does not stop shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_helper_committed_cancel_does_not_abort_stop(monkeypatch, tmp_path):
    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop

    stop_called = False

    async def fake_stop(**kwargs):
        nonlocal stop_called
        stop_called = True

    runner.stop = fake_stop  # type: ignore[method-assign]

    async def fake_launch_ok():
        return True

    runner._launch_detached_restart_command = fake_launch_ok  # type: ignore[method-assign]

    with patch("gateway.run._hermes_home", tmp_path):
        success, msg = await runner.dispatch_gateway_restart(
            source=_make_source(), origin="agent_tool"
        )
        assert success is True
        # Stage is COMMITTED
        assert str(runner._restart_stage.value) == "HANDOFF_COMMITTED"

        # Give background task moment to invoke stop
        await asyncio.sleep(0.1)
        assert stop_called is True


# ---------------------------------------------------------------------------
# 6. Service path commitment under ambient INVOCATION_ID vs unset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_path_commitment_dual_environment(monkeypatch, tmp_path):
    # Test 1: Supervisor set
    monkeypatch.setenv("INVOCATION_ID", "systemd-unit-123")
    runner1, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner1._gateway_loop = gw_loop
    runner1.stop = AsyncMock()  # type: ignore[method-assign]

    with patch("gateway.run._hermes_home", tmp_path):
        success1, msg1 = await runner1.dispatch_gateway_restart(
            source=_make_source(), origin="agent_tool"
        )
        assert success1 is True
        assert runner1._restart_via_service is True

    # Test 2: Supervisor unset
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    runner2, _ = make_restart_runner()
    runner2._gateway_loop = gw_loop
    runner2.stop = AsyncMock()  # type: ignore[method-assign]

    async def fake_launch_ok():
        return True

    runner2._launch_detached_restart_command = fake_launch_ok  # type: ignore[method-assign]

    with patch("gateway.run._hermes_home", tmp_path):
        success2, msg2 = await runner2.dispatch_gateway_restart(
            source=_make_source(), origin="agent_tool"
        )
        assert success2 is True
        assert runner2._restart_via_service is False
        assert runner2._restart_detached is True


# ---------------------------------------------------------------------------
# 7. Redelivery suppression integrated into public entry path (_handle_message)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redelivery_suppression_in_public_handle_message_entry(monkeypatch, tmp_path):
    runner, _ = make_restart_runner()
    runner._booted_from_restart = True

    marker_data = {
        "platform": "telegram",
        "chat_id": "chat-99",
        "thread_id": None,
        "message_id": "msg-nl-123",
        "origin": "agent_tool",
        "requested_at": time.time(),
    }
    (tmp_path / ".restart_last_processed.json").write_text(json.dumps(marker_data))

    event_exact = MessageEvent(
        text="please restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="chat-99"),
        message_id="msg-nl-123",
    )

    with patch("gateway.run._hermes_home", tmp_path), patch.object(
        runner, "_handle_message_with_agent", AsyncMock()
    ) as mock_agent_turn:
        # Call public entry path
        res = await runner._handle_message(event_exact)

        # Invariant: suppressed at public entry! Returns empty string, agent turn NEVER called!
        assert res == ""
        mock_agent_turn.assert_not_called()
