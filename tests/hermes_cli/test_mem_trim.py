"""Tests for the long-lived gateway heap-trim helper."""

from unittest.mock import Mock

import pytest

import hermes_cli.mem_trim as mem_trim


@pytest.fixture(autouse=True)
def _reset_trim_state(monkeypatch):
    monkeypatch.setattr(mem_trim, "_last_trim_monotonic", 0.0)
    monkeypatch.setattr(mem_trim, "_probe_done", True)
    monkeypatch.setattr(mem_trim, "_malloc_trim", None)


def test_unsupported_allocator_is_noop_without_gc(monkeypatch):
    collect = Mock()
    monkeypatch.setattr(mem_trim.gc, "collect", collect)

    assert mem_trim.trim_memory(force=True, reason="test") is False
    collect.assert_not_called()


def test_config_kill_switch_overrides_force(monkeypatch):
    trim = Mock(return_value=1)
    monkeypatch.setattr(mem_trim, "_malloc_trim", trim)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"context": {"memory_trim": {"enabled": False}}},
    )

    assert mem_trim.trim_memory(force=True) is False
    trim.assert_not_called()


def test_default_config_declares_memory_trim_controls():
    from hermes_cli.config import DEFAULT_CONFIG

    context = DEFAULT_CONFIG["context"]
    assert isinstance(context, dict)
    assert context["memory_trim"] == {
        "enabled": True,
        "cooldown_seconds": 60.0,
    }


def test_success_collects_then_trims(monkeypatch):
    calls = []
    monkeypatch.setattr(mem_trim.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(
        mem_trim, "_malloc_trim", lambda pad: calls.append(("trim", pad)) or 1
    )
    monkeypatch.setattr(mem_trim.time, "monotonic", lambda: 100.0)

    assert mem_trim.trim_memory(reason="turn", cooldown_seconds=60) is True
    assert calls == ["gc", ("trim", 0)]
    assert mem_trim._last_trim_monotonic == 100.0


def test_cooldown_suppresses_repeated_collection(monkeypatch):
    collect = Mock()
    trim = Mock(return_value=1)
    monkeypatch.setattr(mem_trim.gc, "collect", collect)
    monkeypatch.setattr(mem_trim, "_malloc_trim", trim)
    monkeypatch.setattr(mem_trim, "_last_trim_monotonic", 95.0)
    monkeypatch.setattr(mem_trim.time, "monotonic", lambda: 100.0)

    assert mem_trim.trim_memory(cooldown_seconds=60) is False
    collect.assert_not_called()
    trim.assert_not_called()
    assert mem_trim.trim_memory(force=True, cooldown_seconds=60) is True


def test_config_cooldown_controls_rate_limit(monkeypatch):
    trim = Mock(return_value=1)
    monkeypatch.setattr(mem_trim, "_malloc_trim", trim)
    monkeypatch.setattr(mem_trim, "_last_trim_monotonic", 1.0)
    monkeypatch.setattr(mem_trim.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "context": {
                "memory_trim": {"enabled": True, "cooldown_seconds": 120.0}
            }
        },
    )

    assert mem_trim.trim_memory() is False
    trim.assert_not_called()


def test_legacy_environment_switch_does_not_control_behavior(monkeypatch):
    trim = Mock(return_value=1)
    monkeypatch.setattr(mem_trim, "_malloc_trim", trim)
    monkeypatch.setenv("HERMES_DISABLE_MEMORY_TRIM", "1")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"context": {"memory_trim": {"enabled": True}}},
    )

    assert mem_trim.trim_memory(force=True) is True
    trim.assert_called_once_with(0)


def test_libc_failure_is_fail_open_and_rate_limited(monkeypatch):
    trim = Mock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(mem_trim, "_malloc_trim", trim)
    monkeypatch.setattr(mem_trim.time, "monotonic", lambda: 100.0)

    assert mem_trim.trim_memory(reason="test", cooldown_seconds=60) is False
    assert mem_trim._last_trim_monotonic == 100.0
    assert mem_trim.trim_memory(cooldown_seconds=60) is False
    assert trim.call_count == 1
