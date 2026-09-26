from datetime import datetime, timedelta, timezone

from hermes_cli.gateway import _runtime_health_lines


def _iso_age(seconds_ago: float) -> str:
    """ISO-8601 UTC timestamp ``seconds_ago`` in the past (drives _marker_is_stale)."""
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


_STALE_LINE_PREFIX = "⚠ Stale gateway_state.json:"


def _stale_lines(lines):
    return [ln for ln in lines if ln.startswith(_STALE_LINE_PREFIX)]


def test_runtime_health_lines_flags_stale_running_with_dead_pid(monkeypatch):
    """Stale updated_at + dead PID + 'running' -> contradiction line, no draining line."""
    from gateway import status as status_mod

    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "running",
            "pid": 4242,
            "start_time": 111,
            "updated_at": _iso_age(600),  # well past the 120s TTL -> stale
            "active_agents": 0,
        },
    )
    # Recorded PID is gone (ungraceful kill); no real process is touched.
    monkeypatch.setattr(status_mod, "_pid_exists", lambda pid: False)
    monkeypatch.setattr(status_mod._process_identity, "get_process_start_time", lambda pid: None)

    lines = _runtime_health_lines()

    stale = _stale_lines(lines)
    assert len(stale) == 1, lines
    assert "recorded state 'running'" in stale[0]
    assert "recorded process is gone" in stale[0]
    # The misleading live-state summary must be suppressed.
    assert not any("draining" in ln.lower() for ln in lines), lines


def test_runtime_health_lines_include_fatal_platform_and_startup_reason(monkeypatch):
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "startup_failed",
            "exit_reason": "telegram conflict",
            "platforms": {
                "telegram": {
                    "state": "fatal",
                    "error_message": "another poller is active",
                }
            },
        },
    )

    lines = _runtime_health_lines()

    assert "⚠ telegram: another poller is active" in lines
    assert "⚠ Last startup issue: telegram conflict" in lines


def test_runtime_health_lines_flag_stale_heartbeat_with_live_pid(monkeypatch):
    """'running' + updated_at past the TTL + PID ALIVE is the reporter's 'not a crash' case
    (#113372): housekeeping stopped stamping the heartbeat while the file still says running.
    Render it as a heartbeat warning naming the age and the live PID; a fresh stamp stays silent."""
    from gateway import status as status_mod

    record = {"gateway_state": "running", "pid": 4242, "start_time": 111,
              "updated_at": _iso_age(900), "active_agents": 0, "platforms": {}}
    monkeypatch.setattr("gateway.status.read_runtime_status", lambda: record)
    monkeypatch.setattr(status_mod, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(status_mod._process_identity, "get_process_start_time", lambda pid: 111)

    stale = [ln for ln in _runtime_health_lines() if ln.startswith("⚠ Gateway heartbeat stale:")]
    assert len(stale) == 1, _runtime_health_lines()
    assert 900 <= int(stale[0].split(" for ")[1].split(" s")[0]) <= 930
    assert "pid 4242 alive" in stale[0]
    assert not _stale_lines(_runtime_health_lines())  # not the dead-PID contradiction line

    record["updated_at"] = _iso_age(5)
    assert not [ln for ln in _runtime_health_lines() if "heartbeat" in ln]


def test_runtime_health_lines_render_watchdog_degraded_exit(monkeypatch):
    """A watchdog-stamped ``degraded`` + exit_reason renders as a health line; the startup-time
    ``degraded`` (retryable platforms queued, no exit_reason) stays silent (#113372)."""
    record = {"gateway_state": "degraded", "exit_reason": "loop_liveness_watchdog",
              "pid": 4242, "updated_at": _iso_age(30), "platforms": {}}
    monkeypatch.setattr("gateway.status.read_runtime_status", lambda: record)

    degraded = [ln for ln in _runtime_health_lines() if ln.startswith("⚠ Gateway exited degraded:")]
    assert len(degraded) == 1
    assert "event loop stopped dispatching" in degraded[0]

    record["exit_reason"] = None
    assert not [ln for ln in _runtime_health_lines() if "degraded" in ln]


def test_runtime_status_running_pid_validates_live_gateway_record(monkeypatch):
    from gateway import status as status_mod

    runtime = {
        "pid": 12345,
        "kind": "hermes-gateway",
        "argv": ["/opt/hermes/hermes_cli/main.py", "gateway", "run", "--replace"],
        "start_time": None,
        "gateway_state": "running",
    }
    monkeypatch.setattr(status_mod, "_pid_exists", lambda pid: pid == 12345)
    monkeypatch.setattr(status_mod._process_identity, "get_process_start_time", lambda pid: None)
    monkeypatch.setattr(status_mod, "_looks_like_gateway_process", lambda pid: False)

    assert status_mod.get_runtime_status_running_pid(runtime) == 12345
