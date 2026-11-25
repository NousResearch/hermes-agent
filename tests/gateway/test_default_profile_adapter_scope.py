"""Default-profile plugin adapters are created under the DEFAULT root's secret scope (multiplex)."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import hermes_constants
from agent import secret_scope
from agent.secret_scope import UnscopedSecretError, get_secret
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.run import GatewayRunner, _profile_runtime_scope

VAR = "FAKEPLUGIN_APP_PASSWORD"
NAME = "fakeplugin_scope"


def _home(tmp_path: Path, name: str, value: str) -> Path:
    home = tmp_path / name
    home.mkdir()
    (home / ".env").write_text(f"{VAR}={value}\n")
    return home


@pytest.fixture
def rig(tmp_path, monkeypatch):
    root = _home(tmp_path, "root", "default-root-value")
    launch = _home(tmp_path, "launch", "launch-home-value")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv(VAR, "os-environ-value")
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda *a, **k: root)
    secret_scope.set_multiplex_active(True)
    seen = {"validate": [], "factory": []}

    class _Adapter:
        gateway_runner = None
        has_fatal_error = False

        async def connect(self, *, is_reconnect=False):
            return True

    def _validate(cfg):
        seen["validate"].append(get_secret(VAR))
        return True

    def _factory(cfg):
        seen["factory"].append(get_secret(VAR))
        return _Adapter()

    platform_registry.register(PlatformEntry(
        name=NAME, label="Fake", adapter_factory=_factory, check_fn=lambda: bool(get_secret(VAR) or True),
        validate_config=_validate, source="plugin"))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True, platforms={})
    runner._failed_platforms = {}
    runner.adapters = {}
    yield runner, seen, tmp_path
    platform_registry.unregister(NAME)
    secret_scope.set_multiplex_active(False)


def test_default_profile_creation_reads_default_root_value(rig):
    runner, seen, _ = rig
    with runner._default_root_adapter_scope():
        adapter = runner._create_adapter(Platform(NAME), PlatformConfig(enabled=True))
    assert adapter is not None
    assert seen["validate"] == ["default-root-value"]
    assert seen["factory"] == ["default-root-value"]


def test_creation_without_scope_fails_closed(rig):
    runner, seen, _ = rig
    assert runner._create_adapter(Platform(NAME), PlatformConfig(enabled=True)) is None
    assert seen["factory"] == []


def test_unscoped_get_secret_outside_creation_still_raises(rig):
    runner = rig[0]
    with runner._default_root_adapter_scope():
        assert get_secret(VAR) == "default-root-value"
    with pytest.raises(UnscopedSecretError):  # scope is released again
        get_secret(VAR)


def test_secondary_still_gets_its_own_value(rig):
    runner, seen, tmp_path = rig
    secondary = _home(tmp_path, "secondary", "secondary-value")
    with _profile_runtime_scope(secondary):
        runner._create_adapter(Platform(NAME), PlatformConfig(enabled=True))
    assert seen["factory"] == ["secondary-value"]


def test_not_multiplexed_binds_nothing(rig):
    runner, _, _ = rig
    runner.config.multiplex_profiles = False
    with runner._default_root_adapter_scope():
        assert secret_scope.current_secret_scope() is None


@pytest.mark.asyncio
async def test_reconnect_path_reads_default_root_value(rig, monkeypatch):
    runner, seen, _ = rig
    runner._running = True
    runner._update_platform_runtime_status = MagicMock()
    runner._install_reconnected_adapter = AsyncMock()
    runner._drop_from_reconnect_queue = MagicMock()
    platform = Platform(NAME)
    runner._failed_platforms[platform] = {
        "config": PlatformConfig(enabled=True, token="t"), "attempts": 0, "next_retry": 0,
    }
    await runner._reconnect_failed_platform(platform, time.monotonic() + 1)
    assert seen["factory"] == ["default-root-value"]
    assert seen["validate"] == ["default-root-value"]
