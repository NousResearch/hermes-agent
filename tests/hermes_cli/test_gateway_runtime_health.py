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
    monkeypatch.setattr(status_mod, "_get_process_start_time", lambda pid: None)

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
    monkeypatch.setattr(status_mod, "_get_process_start_time", lambda pid: 111)

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


def test_runtime_health_lines_show_a_platform_holding_undelivered_inbound(monkeypatch):
    """A DM held by the platform must be visible in `hermes gateway status`.

    Kanban t_1915c9a3 (C3): when the getUpdates consumer is wedged, Telegram holds the
    operator's messages server-side. They are not lost, but nothing local says so — the
    status file carried only a fatal-platform error line, so a platform that was UP but
    holding an unanswered DM rendered as silently healthy.

    The platform is flagged ``needs_attention`` while connected; the health lines must
    name it, so the operator can tell "held, not lost" from "gone".
    """
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "running",
            "pid": 4242,
            "start_time": 111,
            "updated_at": _iso_age(5),
            "platforms": {
                "telegram": {
                    # The platform's own state stays "connected": the send path is
                    # healthy and the platform is up. It is the INBOUND queue that is
                    # not draining, and that is what needs_attention is for.
                    "state": "connected",
                    "needs_attention": True,
                    "error_code": "inbound_updates_outstanding",
                    "error_message": "3 inbound update(s) held by Telegram for 900s — the "
                                      "getUpdates consumer is not draining them. Messages are "
                                      "not lost, but they are not being answered yet.",
                }
            },
        },
    )

    lines = _runtime_health_lines()

    held = [ln for ln in lines if ln.startswith("⚠ telegram:")]
    assert len(held) == 1, lines
    assert "3 inbound update(s) held by Telegram" in held[0]
    assert "not lost" in held[0]


def test_runtime_health_lines_stay_silent_for_a_healthy_connected_platform(monkeypatch):
    """A connected platform with no attention flag must not produce a warning line."""
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "running",
            "pid": 4242,
            "start_time": 111,
            "updated_at": _iso_age(5),
            "platforms": {"telegram": {"state": "connected"}},
        },
    )

    assert not [ln for ln in _runtime_health_lines() if ln.startswith("⚠ telegram:")]


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
    monkeypatch.setattr(status_mod, "_get_process_start_time", lambda pid: None)
    monkeypatch.setattr(status_mod, "_looks_like_gateway_process", lambda pid: False)

    assert status_mod.get_runtime_status_running_pid(runtime) == 12345


