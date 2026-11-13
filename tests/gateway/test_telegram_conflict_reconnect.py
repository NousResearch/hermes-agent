"""A Telegram adapter that exhausted its own 409 polling-conflict ladder is reconnected by the runner on a bounded
back-off, then parked as ``token_in_use_elsewhere`` once another poller keeps the token (#146909). Both the launch
profile's watcher queue and a multiplexed secondary's reconnect slot go through the real runner methods."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import hermes_agent.gateway.run_adapters_recovery as recovery
from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig
from hermes_agent.gateway.run import GatewayRunner
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_multiplex_adapter_registry import _secondary_recovery_runner

TG = Platform.TELEGRAM
_CONFLICT = type("Conflict", (Exception,), {})("Conflict: terminated by other getUpdates request")


def _exhausted_conflict_adapter(handler):
    """A real adapter one 409 away from the end of its own ladder (the 200 s budget is already spent)."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:abc"))
    adapter._app = SimpleNamespace(updater=SimpleNamespace(stop=AsyncMock(), running=False))
    adapter._polling_conflict_count = 5
    adapter.set_fatal_error_handler(handler)
    return adapter


def _record_status(monkeypatch, runner):
    writes = []
    monkeypatch.setattr(runner, "_update_platform_runtime_status", lambda key, **kw: writes.append((key, kw)))
    return writes


@pytest.mark.asyncio
async def test_secondary_polling_conflict_reconnects_then_parks_a_foreign_holder(monkeypatch):
    monkeypatch.setattr(recovery, "_TOKEN_CONTENTION_RECONNECT_DELAYS", (0.01, 0.01, 0.01), raising=False)
    runner = _secondary_recovery_runner()
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: Path("/profiles") / name)
    monkeypatch.setattr("hermes_agent.gateway.run_adapters.record_platform_disconnect", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_safe_adapter_disconnect", AsyncMock())
    writes = _record_status(monkeypatch, runner)
    reconnected = asyncio.Queue()

    async def reconnect(profile_name, platform, predecessor=None):
        runner._profile_failed_platforms[profile_name].pop(platform)
        reconnected.put_nowait(predecessor)

    monkeypatch.setattr(runner, "_run_secondary_profile_reconnect", reconnect)
    handler = runner._make_profile_fatal_error_handler("a", TG)

    for _ in recovery._TOKEN_CONTENTION_RECONNECT_DELAYS:
        adapter = _exhausted_conflict_adapter(handler)
        runner._profile_adapters["a"] = {TG: adapter}
        await adapter._handle_polling_conflict(_CONFLICT)
        assert await asyncio.wait_for(reconnected.get(), timeout=5) is adapter  # rescheduled with its predecessor

    # A holder still there after every reconnect is another live poller: park, flag, stop retrying.
    adapter = _exhausted_conflict_adapter(handler)
    runner._profile_adapters["a"] = {TG: adapter}
    await adapter._handle_polling_conflict(_CONFLICT)
    assert TG not in runner._profile_failed_platforms.get("a", {})
    assert (adapter.fatal_error_code, adapter.fatal_error_retryable) == ("token_in_use_elsewhere", False)
    key, last = writes[-1]
    assert key == "a:telegram"
    assert (last["platform_state"], last["error_code"], last["needs_attention"]) == ("fatal", "token_in_use_elsewhere", True)


@pytest.mark.asyncio
async def test_primary_polling_conflict_is_queued_on_the_back_off_then_parked(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery, "_TOKEN_CONTENTION_RECONNECT_DELAYS", (30.0, 60.0, 90.0), raising=False)
    runner = GatewayRunner(GatewayConfig(
        platforms={TG: PlatformConfig(enabled=True, token="123:abc")}, sessions_dir=tmp_path / "sessions"))
    runner.stop = AsyncMock()
    monkeypatch.setattr(runner, "_safe_adapter_disconnect", AsyncMock())
    monkeypatch.setattr(runner, "_ensure_reconnect_watcher_running", lambda: None)
    writes = _record_status(monkeypatch, runner)

    for delay in recovery._TOKEN_CONTENTION_RECONNECT_DELAYS:
        adapter = _exhausted_conflict_adapter(runner._handle_adapter_fatal_error)
        runner.adapters = {TG: adapter}
        runner.delivery_router.adapters = runner.adapters
        await adapter._handle_polling_conflict(_CONFLICT)
        entry = runner._failed_platforms.pop(TG)  # the watcher's reconnect would take it from here
        assert entry["next_retry"] - entry["queued_at"] == pytest.approx(delay)
        runner.stop.assert_not_awaited()

    adapter = _exhausted_conflict_adapter(runner._handle_adapter_fatal_error)
    runner.adapters = {TG: adapter}
    runner.delivery_router.adapters = runner.adapters
    await adapter._handle_polling_conflict(_CONFLICT)
    assert TG not in runner._failed_platforms
    assert any(kw.get("error_code") == "token_in_use_elsewhere" and kw.get("needs_attention") for _, kw in writes)
    # The last platform parks degraded (cron and local clients keep running) instead of a service-restart exit.
    runner.stop.assert_not_awaited()
    assert runner._exit_with_failure is False
