"""Active-hours / quiet-hours window for session heartbeats (#93029).

``heartbeat.active_hours`` in config.yaml restricts heartbeat turns to a
timezone-aware window: start inclusive, end exclusive, overnight windows
wrap midnight, out-of-window due ticks are skipped (not recorded) until
the first in-window tick.
"""

import time
from datetime import datetime, timezone as dt_timezone

import pytest

from hermes_cli import heartbeat
from hermes_cli.heartbeat import HeartbeatManager, parse_hhmm


def _configure(monkeypatch, start="", end="", tz=""):
    """Pin load_config() so window resolution is deterministic.

    Default tz is empty → host-local, matching the naive timestamps that
    ``_ts`` builds by default.
    """
    cfg = {
        "heartbeat": {
            "active_hours": {"start": start, "end": end, "timezone": tz},
        },
    }

    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: cfg
    )


def _ts(hour, minute=0, zone=None):
    """Timestamp for HH:MM TODAY in the given zone (or host-local).

    Relative to today so these epochs stay comparable against heartbeat
    state anchored to the real clock.
    """
    if zone is None:
        base = datetime.now().astimezone()
    else:
        base = datetime.now(zone)
    return base.replace(
        hour=hour, minute=minute, second=0, microsecond=0
    ).timestamp()


def _ready_manager(session_id="sess-active-hours"):
    manager = HeartbeatManager(session_id)
    state = manager.set("check things", heartbeat.MIN_INTERVAL_SECONDS)
    # Anchor two days back so ANY wall-clock hour today is overdue — the
    # window gate, not interval math, decides whether a tick fires.
    state.created_at = time.time() - 2 * 86400
    state.last_fired_at = 0.0
    return manager


UTC = dt_timezone.utc


@pytest.fixture(autouse=True)
def _reset_warn_once():
    """Each test sees its own warn-once state."""
    heartbeat._warned_active_hours = False
    yield
    heartbeat._warned_active_hours = False


@pytest.mark.parametrize("value,expected", [
    ("08:00", (8, 0)),
    ("22:05", (22, 5)),
    ("8:00", (8, 0)),          # single-digit hour tolerated
    ("06:59", (6, 59)),
])
def test_parse_hhmm_valid(value, expected):
    parsed = parse_hhmm(value)
    assert parsed is not None
    assert (parsed.hour, parsed.minute) == expected


@pytest.mark.parametrize("value", [
    "", None, "24:00", "12:60", "8am", "0800", "aa:bb", "12:3",
])
def test_parse_hhmm_rejects_garbage(value):
    assert parse_hhmm(value) is None


# ---------------------------------------------------------------------------
# window predicate
# ---------------------------------------------------------------------------


def test_no_window_means_always_active(monkeypatch):
    _configure(monkeypatch, start="", end="")
    for hour in (0, 3, 12, 23):
        assert heartbeat.in_active_hours(_ts(hour)) is True


def test_day_window_boundaries_inclusive_start_exclusive_end(monkeypatch):
    _configure(monkeypatch, start="08:00", end="22:00")
    assert heartbeat.in_active_hours(_ts(7, 59)) is False
    assert heartbeat.in_active_hours(_ts(8, 0)) is True      # start inclusive
    assert heartbeat.in_active_hours(_ts(21, 59)) is True
    assert heartbeat.in_active_hours(_ts(22, 0)) is False    # end exclusive


def test_overnight_window_wraps_midnight(monkeypatch):
    _configure(monkeypatch, start="22:00", end="06:00")
    assert heartbeat.in_active_hours(_ts(21, 59)) is False
    assert heartbeat.in_active_hours(_ts(22, 0)) is True
    assert heartbeat.in_active_hours(_ts(2, 30)) is True
    assert heartbeat.in_active_hours(_ts(5, 59)) is True
    assert heartbeat.in_active_hours(_ts(6, 0)) is False     # end exclusive
    assert heartbeat.in_active_hours(_ts(12, 0)) is False


def test_timezone_aware_window_uses_declared_zone(monkeypatch):
    """The same wall clock reads differently through the declared zone."""
    _configure(monkeypatch, start="09:00", end="17:00", tz="America/Los_Angeles")
    # 16:30 UTC == 09:30 PDT -> inside; 23:30 UTC == 16:30 PDT -> inside;
    # 07:30 UTC == 00:30 PDT -> outside.
    assert heartbeat.in_active_hours(_ts(16, 30, UTC)) is True
    assert heartbeat.in_active_hours(_ts(23, 30, UTC)) is True
    assert heartbeat.in_active_hours(_ts(7, 30, UTC)) is False


def test_unknown_timezone_falls_back_to_host_local(monkeypatch, caplog):
    _configure(monkeypatch, start="00:00", end="23:59", tz="Not/AZone")
    # Must not raise; a whole-day-except-one-minute window contains most
    # host-local times either way — just prove it evaluates.
    with caplog.at_level("WARNING"):
        result = heartbeat.in_active_hours(_ts(12, 0))
    assert isinstance(result, bool)
    assert any("unknown" in r.message for r in caplog.records)


def test_equal_start_and_end_is_rejected_not_always_on(monkeypatch, caplog):
    """Equal bounds are ambiguous; ignore the window rather than guess.

    Critically this must NOT degrade to "always active" silently — a
    warning tells the operator their window was rejected (#93029 asks for
    unambiguous handling).
    """
    _configure(monkeypatch, start="08:00", end="08:00")
    with caplog.at_level("WARNING"):
        assert heartbeat.in_active_hours(_ts(3, 0)) is True
    assert any("distinct" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# due_prompt integration
# ---------------------------------------------------------------------------


def test_due_tick_outside_window_skips_without_recording(monkeypatch):
    _configure(monkeypatch, start="08:00", end="22:00")
    manager = _ready_manager()
    outside = _ts(3, 0)
    assert manager.due_prompt(now=outside) is None
    state = manager.state
    assert state.fire_count == 0          # skipped, NOT recorded
    assert state.last_fired_at == 0.0     # anchor untouched


def test_first_in_window_poll_fires_immediately(monkeypatch):
    _configure(monkeypatch, start="08:00", end="22:00")
    manager = _ready_manager()
    inside = _ts(8, 0)
    prompt = manager.due_prompt(now=inside)
    assert prompt is not None
    assert "check things" in prompt
    assert manager.state.fire_count == 1


def test_out_of_window_ticks_never_stack_a_backlog(monkeypatch):
    """Many suppressed polls must behave like one skipped moment."""
    _configure(monkeypatch, start="08:00", end="22:00")
    manager = _ready_manager()
    for minute in range(0, 60):
        assert manager.due_prompt(now=_ts(7, minute)) is None
    assert manager.state.fire_count == 0
    assert manager.due_prompt(now=_ts(8, 0)) is not None   # exactly one fire
    assert manager.state.fire_count == 1


def test_status_line_surfaces_the_window(monkeypatch):
    _configure(monkeypatch, start="08:00", end="22:00")
    manager = HeartbeatManager("sess-status-window")
    manager.set("check things", heartbeat.MIN_INTERVAL_SECONDS)
    line = manager.status_line()
    assert "active 08:00" in line and "22:00" in line
