"""``--replace`` must wait the old gateway's full graceful-stop budget before SIGKILL.

A fixed 10s takeover grace force-killed a draining gateway mid-SQLite-write on every busy
restart (``restart_drain_timeout`` can be minutes); ``state.db`` later reported ``database disk
image is malformed``. The wait now derives from the same budget systemd's ``TimeoutStopSec``
uses (``resolve_systemd_timeout_stop_sec``).
"""
from __future__ import annotations

import pytest

from gateway.restart import (
    CRON_DRAIN_CLEANUP_RESERVE_S,
    REPLACE_TAKEOVER_GRACE_FLOOR_S,
    REPLACE_TAKEOVER_GRACE_HEADROOM_S,
    resolve_replace_takeover_grace_s,
    resolve_systemd_timeout_stop_sec,
)


class TestResolveReplaceTakeoverGrace:
    def test_idle_default_exceeds_the_old_ten_second_cap(self):
        got = resolve_replace_takeover_grace_s(0, 0)
        assert got == max(REPLACE_TAKEOVER_GRACE_FLOOR_S, REPLACE_TAKEOVER_GRACE_HEADROOM_S)
        assert got > 10.0

    def test_drain_timeout_extends_the_grace(self):
        assert resolve_replace_takeover_grace_s(180, 0) == 180 + REPLACE_TAKEOVER_GRACE_HEADROOM_S

    def test_cron_budget_wins_when_larger(self):
        got = resolve_replace_takeover_grace_s(30, 600)
        assert got == 600 + CRON_DRAIN_CLEANUP_RESERVE_S + REPLACE_TAKEOVER_GRACE_HEADROOM_S

    def test_agrees_with_systemd_model(self):
        for drain, cron in ((0, 0), (180, 0), (30, 600), (5, 5)):
            assert resolve_replace_takeover_grace_s(
                drain, cron, headroom_s=30.0, floor_s=60.0
            ) == float(resolve_systemd_timeout_stop_sec(drain, cron, headroom_s=30.0, floor_s=60.0))

    def test_garbage_degrades_like_idle(self):
        got = resolve_replace_takeover_grace_s("nope", None)  # type: ignore[arg-type]
        assert got == resolve_replace_takeover_grace_s(0, 0)
        assert got >= REPLACE_TAKEOVER_GRACE_FLOOR_S


@pytest.mark.asyncio
async def test_replace_waits_past_ten_seconds_before_sigkill(monkeypatch, tmp_path):
    """RED on the old code: SIGKILL after 20 x 0.5s while the old gateway is still inside a
    40s drain. GREEN with the fix: the wait honors the drain budget and the old gateway exits
    on its own."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "40")
    monkeypatch.setenv("HERMES_CRON_DRAIN_TIMEOUT", "0")

    import gateway.run as run_mod

    events = []
    # Virtual clock: each asyncio.sleep(d) advances d; the old gateway exits 25s after SIGTERM
    # (inside a 40s drain, outside the old 10s cap).
    clock = {"t": 0.0}
    old_alive = {"v": True}

    async def _fake_sleep(d):
        clock["t"] += d
        if clock["t"] >= 25.0:
            old_alive["v"] = False

    monkeypatch.setattr(run_mod.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: old_alive["v"])
    monkeypatch.setattr("gateway.status._get_process_start_time", lambda pid: 0)
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: 0, raising=False)
    monkeypatch.setattr("gateway.status.write_takeover_marker", lambda pid: None)
    monkeypatch.setattr("gateway.status._snapshot_gateway_children", lambda pid: [])
    monkeypatch.setattr("gateway.status.reap_gateway_children",
                        lambda children, *, parent_pid, timeout=5.0: 0)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.release_all_scoped_locks", lambda **kwargs: 0)
    monkeypatch.setattr(
        "gateway.status._read_pid_record",
        lambda path=None: {
            "pid": 42, "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway", "run"],
            "start_time": 0, "hermes_home": str(tmp_path),
        },
    )

    def _terminate(pid, force=False, **kwargs):
        events.append(("terminate", pid, force, clock["t"]))
        if force:
            old_alive["v"] = False

    monkeypatch.setattr("gateway.status.terminate_pid", _terminate)
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)

    ok = await run_mod._start_gateway_replace_existing_instance(42, replace=True)
    assert ok is True
    forced = [e for e in events if e[2] is True]
    assert not forced, (
        f"SIGKILL fired at t={forced[0][3]}s while old gateway was still inside its 40s drain"
    )
    assert events and events[0][:3] == ("terminate", 42, False)
    assert old_alive["v"] is False


# --- Version-skew: the lease belongs to the OLD incarnation, not to current config -------------
# Both regressions requested by @andrexibiza on #113355. The old GatewayRunner freezes
# ``_restart_drain_timeout`` / ``_cron_drain_timeout`` at ITS construction and ``stop()`` drains
# from those retained values, so sizing the destructive deadline from config re-read in the
# REPLACEMENT process answers for a different generation.


def _pid_record_with_budget(tmp_path, budget_s):
    record = {
        "pid": 42, "kind": "hermes-gateway",
        "argv": ["python", "-m", "hermes_cli.main", "gateway", "run"],
        "start_time": 0, "hermes_home": str(tmp_path),
    }
    if budget_s is not None:
        record["stop_budget_s"] = budget_s
    return record


def _install_replace_stubs(monkeypatch, tmp_path, *, pid_record, old_exits_at_s):
    """Virtual-clock stubs for ``_start_gateway_replace_existing_instance``; returns
    ``(events, clock, old_alive)``."""
    import gateway.run as run_mod

    events = []
    clock = {"t": 0.0}
    old_alive = {"v": True}

    async def _fake_sleep(d):
        clock["t"] += d
        if clock["t"] >= old_exits_at_s:
            old_alive["v"] = False

    monkeypatch.setattr(run_mod.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: old_alive["v"])
    monkeypatch.setattr("gateway.status._get_process_start_time", lambda pid: 0)
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: 0, raising=False)
    monkeypatch.setattr("gateway.status.write_takeover_marker", lambda pid: None)
    monkeypatch.setattr("gateway.status._snapshot_gateway_children", lambda pid: [])
    monkeypatch.setattr("gateway.status.reap_gateway_children",
                        lambda children, *, parent_pid, timeout=5.0: 0)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.release_all_scoped_locks", lambda **kwargs: 0)
    monkeypatch.setattr("gateway.status._read_pid_record", lambda path=None: pid_record)

    def _terminate(pid, force=False, **kwargs):
        events.append(("terminate", pid, force, clock["t"]))
        if force:
            old_alive["v"] = False

    monkeypatch.setattr("gateway.status.terminate_pid", _terminate)
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)
    return events, clock, old_alive


def test_runner_publishes_its_frozen_stop_budget_into_the_pid_record(monkeypatch, tmp_path):
    """The lease the successor reads must come from the OLD runner's own snapshot."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "180")
    monkeypatch.setenv("HERMES_CRON_DRAIN_TIMEOUT", "0")

    import gateway.status as status_mod
    from gateway.run import GatewayRunner

    monkeypatch.setattr(status_mod, "_PUBLISHED_STOP_BUDGET_S", None, raising=False)
    runner = object.__new__(GatewayRunner)
    GatewayRunner._init_runtime_settings(runner)

    assert runner._restart_drain_timeout == 180.0
    assert status_mod._build_pid_record()["stop_budget_s"] == 180.0

    # Config lowered AFTER that snapshot must not change the published lease.
    monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "0")
    assert status_mod._build_pid_record()["stop_budget_s"] == 180.0
    assert status_mod.read_published_stop_budget_s(
        42, _pid_record_with_budget(tmp_path, 180.0)) == 180.0
    # Unknown lease must read as unknown, never as zero.
    assert status_mod.read_published_stop_budget_s(
        42, _pid_record_with_budget(tmp_path, None)) is None


@pytest.mark.asyncio
async def test_lowered_config_cannot_shorten_the_old_generations_lease(monkeypatch, tmp_path):
    """RED on the PR head: old gateway started with ``restart_drain_timeout=180`` and is still
    inside that frozen drain; config was lowered to 0 before ``gateway run --replace``, so the
    successor derived ~30s and SIGKILLed it mid-drain. GREEN: the lease is read from the old
    incarnation's published budget, so no force-kill happens before it elapses."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "0")  # lowered since the old gateway started
    monkeypatch.setenv("HERMES_CRON_DRAIN_TIMEOUT", "0")

    import gateway.run as run_mod

    events, _clock, old_alive = _install_replace_stubs(
        monkeypatch, tmp_path,
        pid_record=_pid_record_with_budget(tmp_path, 180.0),
        old_exits_at_s=150.0,  # legitimately inside its frozen 180s drain
    )

    ok = await run_mod._start_gateway_replace_existing_instance(42, replace=True)
    assert ok is True
    forced = [e for e in events if e[2] is True]
    assert not forced, (
        f"SIGKILL fired at t={forced[0][3]}s while the old gateway was still inside the 180s "
        "drain it snapshotted at its own start"
    )
    assert old_alive["v"] is False
    assert run_mod._replace_takeover_grace_s(42) >= 180.0


@pytest.mark.asyncio
async def test_config_loader_failure_cannot_collapse_the_kill_deadline(monkeypatch, tmp_path):
    """RED on the PR head: ``except Exception -> (0, 0)`` minted the SHORTEST destructive lease
    (~30s) from an unreadable config. GREEN: an unknown lease fails closed on the conservative
    floor, so a loader error cannot make the replacement force-kill earlier."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import gateway.run as run_mod

    def _boom(*_a, **_kw):
        raise RuntimeError("config.yaml is unreadable")

    monkeypatch.setattr(run_mod._REAL_GATEWAY_RUNNER_CLASS, "_load_restart_drain_timeout",
                        classmethod(_boom))
    monkeypatch.setattr(run_mod._REAL_GATEWAY_RUNNER_CLASS, "_load_cron_drain_timeout",
                        classmethod(_boom))

    events, _clock, old_alive = _install_replace_stubs(
        monkeypatch, tmp_path,
        pid_record=_pid_record_with_budget(tmp_path, None),
        old_exits_at_s=45.0,  # inside the conservative floor, outside the old ~30s lease
    )
    ok = await run_mod._start_gateway_replace_existing_instance(42, replace=True)
    assert ok is True
    forced = [e for e in events if e[2] is True]
    assert not forced, (
        f"SIGKILL fired at t={forced[0][3]}s — an unreadable config minted the shortest lease"
    )
    assert old_alive["v"] is False

    # Unreadable config, and the old gateway published no lease: the longest bound, not the shortest.
    from gateway.restart import REPLACE_TAKEOVER_GRACE_UNKNOWN_LEASE_FLOOR_S
    monkeypatch.setattr("gateway.status._read_pid_record",
                        lambda path=None: _pid_record_with_budget(tmp_path, None))
    assert run_mod._replace_takeover_grace_s(42) >= REPLACE_TAKEOVER_GRACE_UNKNOWN_LEASE_FLOOR_S

    # Unreadable config while the old gateway DID publish a long lease: honour the old lease.
    monkeypatch.setattr("gateway.status._read_pid_record",
                        lambda path=None: _pid_record_with_budget(tmp_path, 300.0))
    assert run_mod._replace_takeover_grace_s(42) >= 300.0
