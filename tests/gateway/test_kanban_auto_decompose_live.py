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


def test_per_tick_respected_and_clamped():
    enabled, per_tick = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose": True, "auto_decompose_per_tick": 7}}
    )
    assert (enabled, per_tick) == (True, 7)

    # 0 is treated as "unset" by the `or 3` fallback → default 3 (a 0 per-tick
    # cap would disable progress, so falling back to the default is the safe read).
    _, per_tick_zero = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose_per_tick": 0}}
    )
    assert per_tick_zero == 3

    # A genuine negative value clamps up to 1.
    _, per_tick_neg = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose_per_tick": -5}}
    )
    assert per_tick_neg == 1


def test_malformed_per_tick_falls_back_to_default():
    _, per_tick = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose_per_tick": "lots"}}
    )
    assert per_tick == 3


def test_config_read_error_fails_safe_disabled():
    """A transient config read failure must DISABLE auto-decompose, never
    silently fall back to the default-on behaviour the user turned off."""

    def _boom():
        raise RuntimeError("config read failed")

    enabled, per_tick = _resolve_auto_decompose_settings(_boom)
    assert enabled is False
    assert per_tick == 3


def test_non_dict_config_fails_safe():
    enabled, _ = _resolve_auto_decompose_settings(lambda: None)
    assert enabled is True  # no kanban key → default-on (not an error path)
    enabled2, _ = _resolve_auto_decompose_settings(lambda: ["not", "a", "dict"])
    assert enabled2 is True


def test_live_toggle_takes_effect_between_calls():
    """Simulate a user flipping the flag while the dispatcher runs: a later
    resolution reflects the new value without any restart."""
    state = {"kanban": {"auto_decompose": True}}
    assert _resolve_auto_decompose_settings(lambda: state)[0] is True
    # User edits config.yaml mid-run.
    state["kanban"]["auto_decompose"] = False
    assert _resolve_auto_decompose_settings(lambda: state)[0] is False


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
