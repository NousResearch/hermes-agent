"""Tests for live auto-decompose settings resolution (issue #49638).

The gateway dispatcher used to capture ``kanban.auto_decompose`` once at boot,
so a user who flipped it to ``false`` to STOP runaway auto-decompose (which had
created and launched tasks they didn't intend) found the flag had no effect
without a full gateway restart. ``_resolve_auto_decompose_settings`` is now
called every tick, reading the current config.
"""

from __future__ import annotations

import pytest

from gateway.kanban_watchers import (
    _resolve_auto_decompose_settings,
    _resolve_dispatch_in_gateway_enabled,
)


def test_enabled_by_default_when_key_absent():
    enabled, per_tick = _resolve_auto_decompose_settings(lambda: {"kanban": {}})
    assert enabled is True
    assert per_tick == 3


def test_disabled_when_flag_false():
    enabled, per_tick = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose": False}}
    )
    assert enabled is False


def test_dispatch_in_gateway_enabled_by_default():
    assert _resolve_dispatch_in_gateway_enabled(lambda: {"kanban": {}}) is True


def test_dispatch_in_gateway_disabled_from_config():
    assert (
        _resolve_dispatch_in_gateway_enabled(
            lambda: {"kanban": {"dispatch_in_gateway": False}}
        )
        is False
    )


def test_dispatch_in_gateway_env_override_disables_without_config_load(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "false")

    def _boom():
        raise AssertionError("config should not be read")

    assert _resolve_dispatch_in_gateway_enabled(_boom) is False


def test_dispatch_in_gateway_config_read_error_fails_safe_disabled():
    def _boom():
        raise RuntimeError("config read failed")

    assert _resolve_dispatch_in_gateway_enabled(_boom) is False


def test_running_dispatcher_rechecks_dispatch_gate_each_tick(monkeypatch, tmp_path):
    import asyncio

    from gateway.run import GatewayRunner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    configs = iter(
        [
            {"kanban": {"dispatch_in_gateway": True, "dispatch_interval_seconds": 1}},
            {"kanban": {"dispatch_in_gateway": False, "dispatch_interval_seconds": 1}},
        ]
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: next(configs, {"kanban": {"dispatch_in_gateway": False}}),
    )
    monkeypatch.setattr(
        "gateway.kanban_watchers._acquire_singleton_lock",
        lambda _path: (None, "unavailable"),
    )

    async def _to_thread(*_args, **_kwargs):
        raise AssertionError("disabled dispatch gate must skip worker dispatch")

    async def _sleep(_delay):
        runner._running = False

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", _sleep)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True

    asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=3.0))

