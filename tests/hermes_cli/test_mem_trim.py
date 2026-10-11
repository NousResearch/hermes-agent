"""Tests for the long-lived gateway heap-trim helper."""

import logging
from unittest.mock import Mock

import pytest

from hermes_cli import mem_trim


@pytest.fixture(autouse=True)
def _reset_trim_state(monkeypatch):
    monkeypatch.setattr(mem_trim, "_last_trim_monotonic", 0.0)
    monkeypatch.setattr(mem_trim, "_probe_done", True)
    monkeypatch.setattr(mem_trim, "_malloc_trim", None)
    # raising=False keeps the fixture valid against the pre-#131766 module, where
    # the darwin probe globals did not exist yet.
    monkeypatch.setattr(mem_trim, "_darwin_probe_done", True, raising=False)
    monkeypatch.setattr(mem_trim, "_darwin_relief", None, raising=False)
    monkeypatch.setattr(mem_trim, "_trim_call_count", 0)


def test_gc_collect_runs_even_without_platform_primitive(monkeypatch):
    # Reverses the pre-#131766 contract: with NO page-release primitive resolved
    # (unsupported platform or allocator), trim_memory must still run its
    # rate-limited gc.collect() pass — cycle collection was never glibc-specific.
    # Only the reported success flag stays False when there is no primitive.
    collect = Mock()
    monkeypatch.setattr(mem_trim.gc, "collect", collect)

    assert mem_trim.trim_memory(force=True, reason="test") is False
    collect.assert_called_once()


def test_config_kill_switch_overrides_force_from_config_file(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "context:\n  memory_trim:\n    enabled: false\n",
        encoding="utf-8",
    )
    trim = Mock(return_value=1)
    monkeypatch.setattr(mem_trim, "_malloc_trim", trim)
    token = set_hermes_home_override(hermes_home)

    try:
        assert mem_trim.trim_memory(force=True) is False
        trim.assert_not_called()
    finally:
        reset_hermes_home_override(token)




def test_collect_memory_snapshot_parses_linux_proc_status(monkeypatch):
    # No ``sys.platform`` pin: the only platform check lives inside
    # ``_read_proc_status``, which is replaced below — the subject here is
    # the /proc/self/status parser, which is host-independent.
    monkeypatch.setattr(
        mem_trim,
        "_read_proc_status",
        lambda: "Name:\tpython\nVmRSS:\t1234 kB\nRssAnon:\t567 kB\n",
    )
    monkeypatch.setattr(mem_trim.threading, "active_count", lambda: 9)

    assert mem_trim.collect_memory_snapshot(history_bytes=42) == {
        "rss_kib": 1234,
        "rss_anon_kib": 567,
        "thread_count": 9,
        "history_bytes": 42,
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


def test_darwin_pressure_relief_runs_after_gc_and_reports_success(monkeypatch):
    # macOS page release: libSystem malloc_zone_pressure_relief(zone=None, len=0,
    # flags=0) relieves every allocator zone. It must run AFTER the cycle
    # collection and report success (True) when it executes without raising —
    # the void API has no return code, so "executed" is the success contract.
    monkeypatch.setattr(mem_trim.sys, "platform", "darwin")
    calls = []
    monkeypatch.setattr(mem_trim.gc, "collect", lambda: calls.append("gc"))
    relief = Mock(
        side_effect=lambda zone, length, flags: calls.append(("relief", zone, length, flags)))
    monkeypatch.setattr(mem_trim, "_darwin_probe_done", True)
    monkeypatch.setattr(mem_trim, "_darwin_relief", relief)
    monkeypatch.setattr(mem_trim.time, "monotonic", lambda: 100.0)

    assert mem_trim.trim_memory(force=True, reason="turn") is True
    assert calls == ["gc", ("relief", None, 0, 0)]
    assert mem_trim._trim_call_count == 1
    assert mem_trim._last_trim_monotonic == 100.0


def test_darwin_relief_failure_is_fail_open_and_rate_limited(monkeypatch):
    # A raising pressure-relief call behaves like a failing malloc_trim: warn,
    # return False, and keep the cooldown timestamp so bursts do not stack retries.
    monkeypatch.setattr(mem_trim.sys, "platform", "darwin")
    relief = Mock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(mem_trim, "_darwin_probe_done", True)
    monkeypatch.setattr(mem_trim, "_darwin_relief", relief)
    monkeypatch.setattr(mem_trim.gc, "collect", lambda: None)
    monkeypatch.setattr(mem_trim.time, "monotonic", lambda: 100.0)

    assert mem_trim.trim_memory(reason="test", cooldown_seconds=60) is False
    assert mem_trim._last_trim_monotonic == 100.0
    assert mem_trim.trim_memory(cooldown_seconds=60) is False
    assert relief.call_count == 1


def test_darwin_success_log_still_honors_log_every_n_throttle(monkeypatch, caplog):
    # The darwin path must not bypass the _should_log_trim throttle: with
    # log_every_n=3 only the third successful release logs, exactly like the
    # glibc malloc_trim path.
    monkeypatch.setattr(mem_trim.sys, "platform", "darwin")
    monkeypatch.setattr(mem_trim, "_config_settings", lambda: (True, 0.0, 3, 0.0))
    relief = Mock()
    monkeypatch.setattr(mem_trim, "_darwin_probe_done", True)
    monkeypatch.setattr(mem_trim, "_darwin_relief", relief)
    monkeypatch.setattr(mem_trim.gc, "collect", lambda: None)
    ticks = iter((100.0, 101.0, 102.0))
    monkeypatch.setattr(mem_trim.time, "monotonic", lambda: next(ticks))

    with caplog.at_level(logging.INFO, logger=mem_trim.logger.name):
        for _ in range(3):
            assert mem_trim.trim_memory(reason="turn", cooldown_seconds=0) is True

    trim_logs = [r for r in caplog.records if "memory trim" in r.getMessage()]
    assert len(trim_logs) == 1


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
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "context": {
                "memory_trim": {"enabled": True, "cooldown_seconds": 120.0}
            }
        },
    )

    assert mem_trim.trim_memory() is False
    trim.assert_not_called()




def test_libc_failure_is_fail_open_and_rate_limited(monkeypatch):
    trim = Mock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(mem_trim, "_malloc_trim", trim)
    monkeypatch.setattr(mem_trim.time, "monotonic", lambda: 100.0)

    assert mem_trim.trim_memory(reason="test", cooldown_seconds=60) is False
    assert mem_trim._last_trim_monotonic == 100.0
    assert mem_trim.trim_memory(cooldown_seconds=60) is False
    assert trim.call_count == 1


def test_force_floor_coalesces_burst_closes(monkeypatch):
    """A delegate batch closes N child agents back-to-back, each forcing a
    trim — the short force floor must coalesce the burst instead of stacking
    N uncooled full gc.collect() passes in the same process."""
    collect = Mock()
    trim = Mock(return_value=1)
    monkeypatch.setattr(mem_trim.gc, "collect", collect)
    monkeypatch.setattr(mem_trim, "_malloc_trim", trim)
    monkeypatch.setattr(mem_trim, "_config_settings", lambda: (True, 0.0, 1, 0.0))
    monkeypatch.setattr(
        mem_trim,
        "collect_memory_snapshot",
        lambda: {"rss_kib": 4096, "rss_anon_kib": 3072, "thread_count": 3},
    )
    monkeypatch.setattr(mem_trim, "_last_trim_monotonic", 0.0)

    # t=100: first forced close runs.
    monkeypatch.setattr(mem_trim.time, "monotonic", lambda: 100.0)
    assert mem_trim.trim_memory(force=True, reason="agent close") is True
    assert trim.call_count == 1

    # t=101..103: three more child closes inside the floor — all coalesced.
    for t in (101.0, 102.0, 103.0):
        monkeypatch.setattr(mem_trim.time, "monotonic", lambda t=t: t)
        assert mem_trim.trim_memory(force=True, reason="agent close") is False
    assert trim.call_count == 1, "burst closes must not stack forced trims"

    # t=106: past the floor — the parent's final close-trim still fires.
    monkeypatch.setattr(mem_trim.time, "monotonic", lambda: 106.0)
    assert mem_trim.trim_memory(force=True, reason="agent close") is True
    assert trim.call_count == 2
