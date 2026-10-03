"""Reconnect attention must describe a failed attempt, not an old queue (#126825)."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from tests.gateway.test_platform_reconnect import StubAdapter, _make_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "waiting", "fatal", "missing"])
async def test_old_queue_without_retryable_failure_does_not_escalate(outcome):
    runner = _make_runner()
    now = time.monotonic()
    info = {"config": runner.config.platforms[Platform.TELEGRAM], "attempts": 0,
            "queued_at": now - 10000, "next_retry": now + 60 if outcome == "waiting" else 0}
    runner._failed_platforms[Platform.TELEGRAM] = info
    adapter = StubAdapter(fatal_error="invalid credential" if outcome == "fatal" else None,
                          fatal_retryable=False)
    runner._create_adapter = MagicMock(return_value=None if outcome == "missing" else adapter)
    runner._wire_adapter_handlers = MagicMock()
    runner._install_reconnected_adapter = AsyncMock()
    runner._update_platform_runtime_status = MagicMock()
    with patch("hermes_cli.config.load_config", return_value={"agent": {"reconnect_attention_after": 1}}):
        await runner._reconnect_failed_platform(Platform.TELEGRAM, now)
    assert not info.get("attention_flagged"), "queue age alone must not flag attention"
    assert not any(call.kwargs.get("needs_attention") is True
                   for call in runner._update_platform_runtime_status.call_args_list)
    if outcome == "success":
        runner._install_reconnected_adapter.assert_awaited_once_with(Platform.TELEGRAM, adapter)
    if outcome == "waiting":
        runner._create_adapter.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_retryable_failure_escalates_after_recording_attempt(raises):
    runner = _make_runner()
    now = time.monotonic()
    info = {"config": runner.config.platforms[Platform.TELEGRAM], "attempts": 0,
            "queued_at": now - 10000, "next_retry": 0}
    runner._failed_platforms[Platform.TELEGRAM] = info
    adapter = StubAdapter(succeed=False)
    if raises:
        adapter.connect = AsyncMock(side_effect=RuntimeError("temporary transport failure"))
    runner._create_adapter = MagicMock(return_value=adapter)
    runner._wire_adapter_handlers = MagicMock()
    attempts_at_attention = []
    def status(_platform, **fields):
        if fields.get("needs_attention") is True:
            attempts_at_attention.append(info["attempts"])
    runner._update_platform_runtime_status = status
    with patch("hermes_cli.config.load_config", return_value={"agent": {"reconnect_attention_after": 1}}):
        await runner._reconnect_failed_platform(Platform.TELEGRAM, now)
        await runner._reconnect_failed_platform(Platform.TELEGRAM, now)
    assert attempts_at_attention == [1], "escalate once, only after the failed attempt is recorded"
    assert info["next_retry"] > now
    assert runner._failed_platforms[Platform.TELEGRAM] is info
