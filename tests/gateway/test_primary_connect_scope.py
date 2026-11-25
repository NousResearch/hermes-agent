"""Primary adapter connect() runs under the default root's secret scope (multiplex), so background
tasks it starts inherit the scope and can call get_secret later."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import secret_scope
from agent.secret_scope import UnscopedSecretError, get_secret
from gateway.config import Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.run import _profile_runtime_scope
from tests.gateway.test_default_profile_adapter_scope import NAME, VAR, _home, rig


def _install_task_adapter(results):
    """Re-register the fake platform with an adapter whose connect() spawns a task reading the secret later."""

    class _A:
        gateway_runner = None
        has_fatal_error = False
        fatal_error_retryable = True
        _platform_lock_takeover_allowed = False

        def __init__(self):
            self.task = None

        async def connect(self, *, is_reconnect=False):
            async def _poll():
                await asyncio.sleep(0.05)  # well after connect() returned and the scope was reset
                try:
                    results.append(get_secret(VAR))
                except UnscopedSecretError:
                    results.append("UNSCOPED")
            self.task = asyncio.create_task(_poll())
            return True

    platform_registry.unregister(NAME)
    platform_registry.register(PlatformEntry(
        name=NAME, label="Fake", adapter_factory=lambda cfg: _A(), check_fn=lambda: True,
        validate_config=lambda cfg: True, source="plugin"))
    return _A


@pytest.mark.asyncio
async def test_startup_primary_connect_task_inherits_default_root_scope(rig):
    runner, _, _ = rig
    results = []
    _install_task_adapter(results)
    runner._startup_should_abort = lambda: False
    runner._abort_startup_if_shutdown_requested = AsyncMock(return_value=False)
    runner._update_platform_runtime_status = MagicMock()
    runner._platform_lock_takeover_on_start = False
    platform = Platform(NAME)
    with runner._default_root_adapter_scope():
        adapter = runner._create_adapter(platform, PlatformConfig(enabled=True))
    raw = await runner._start_connect_pending([(platform, PlatformConfig(enabled=True), adapter)])
    assert raw[0][3] == "ok"
    await adapter.task
    assert results == ["default-root-value"]
    with pytest.raises(UnscopedSecretError):  # caller's context is not left scoped
        get_secret(VAR)


@pytest.mark.asyncio
async def test_reconnect_primary_connect_task_inherits_default_root_scope(rig):
    runner, _, _ = rig
    results = []
    _install_task_adapter(results)
    runner._running = True
    runner._update_platform_runtime_status = MagicMock()
    holder = {}
    runner._wire_adapter_handlers = MagicMock()

    async def _install(platform, adapter):
        holder["a"] = adapter
    runner._install_reconnected_adapter = _install
    runner._drop_from_reconnect_queue = MagicMock()
    platform = Platform(NAME)
    runner._failed_platforms[platform] = {
        "config": PlatformConfig(enabled=True, token="t"), "attempts": 0, "next_retry": 0,
    }
    await runner._reconnect_failed_platform(platform, time.monotonic() + 1)
    await holder["a"].task
    assert results == ["default-root-value"]
    with pytest.raises(UnscopedSecretError):
        get_secret(VAR)


@pytest.mark.asyncio
async def test_secondary_connect_keeps_its_own_scope(rig):
    runner, _, tmp_path = rig
    results = []
    _A = _install_task_adapter(results)
    secondary = _home(tmp_path, "secondary", "secondary-value")
    adapter = _A()
    with _profile_runtime_scope(secondary):
        await runner._connect_initial_adapter_with_timeout(adapter, Platform(NAME))
    await adapter.task
    assert results == ["secondary-value"]


@pytest.mark.asyncio
async def test_default_root_env_is_read_off_the_event_loop(rig, monkeypatch):
    import threading
    import gateway.run as gr
    runner, _, _ = rig
    seen = []
    real = gr._load_profile_secret_scope

    def _spy(home):
        seen.append(threading.current_thread() is threading.main_thread())
        return real(home)
    monkeypatch.setattr(gr, "_load_profile_secret_scope", _spy)
    async with runner._async_default_root_adapter_scope():
        assert get_secret(VAR) == "default-root-value"
    assert seen == [False]


@pytest.mark.asyncio
async def test_not_multiplexed_async_scope_binds_nothing(rig):
    runner, _, _ = rig
    runner.config.multiplex_profiles = False
    async with runner._async_default_root_adapter_scope():
        assert secret_scope.current_secret_scope() is None
