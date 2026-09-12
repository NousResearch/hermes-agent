"""Scoped locks must not be held forever by a wedged-but-running gateway.

A gateway process whose asyncio loop has died but whose OS process lingers
passes every pre-existing staleness oracle in ``acquire_scoped_lock()``:

* the PID exists,
* ``start_time`` still matches (it is the same process, not a recycled PID),
* the command line still looks like a gateway,
* and the SIGTSTP probe reads ``/proc/<pid>/status``, which does not exist on
  macOS, so it can never fire there.

The zombie therefore keeps e.g. the Feishu ``app_id`` lock indefinitely while a
supervisor (launchd ``KeepAlive``, systemd ``Restart=``) respawns a replacement
that immediately exits with a non-retryable lock conflict — an unbounded
crash loop that only a manual ``kill -9`` breaks.

``_lock_owner_loop_is_dead()`` closes that gap by reading the event-loop
heartbeat that ``gateway/shutdown_watchdog.py`` writes every 30s. It is
deliberately conservative: anything it cannot positively confirm leaves the
lock intact.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from gateway import status


@pytest.fixture()
def lock_env(tmp_path, monkeypatch):
    """Isolate the lock dir and a fake HERMES_HOME with a heartbeat file."""
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    home = tmp_path / "home"
    (home / "state").mkdir(parents=True)
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    return {"lock_dir": lock_dir, "home": home}


def _write_heartbeat(home, pid, age_seconds):
    """Write <home>/state/gateway.heartbeat aged ``age_seconds`` in the past."""
    stamped = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    path = home / "state" / "gateway.heartbeat"
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "updated_at": stamped.isoformat(),
                "monotonic": 1234.5,
                "start_time": 1789185711.88,
                "loop_tick_socket": True,
                "loop_tick_tcp_port": None,
            }
        ),
        encoding="utf-8",
    )
    return path


def _lock_record(home, pid):
    return {
        "pid": pid,
        "kind": "hermes-gateway",
        "argv": ["hermes_cli/main.py", "gateway", "run"],
        "start_time": 178841426019,
        "hermes_home": str(home),
        "scope": "feishu-app-id",
        "metadata": {"platform": "feishu"},
    }


def test_cold_heartbeat_marks_owner_dead(lock_env):
    """A heartbeat far past the threshold proves the owner's loop is gone."""
    home = lock_env["home"]
    pid = os.getpid()  # a genuinely live PID — only the heartbeat says otherwise
    _write_heartbeat(home, pid, status._LOCK_OWNER_HEARTBEAT_STALE_AFTER_S + 600)

    assert status._lock_owner_loop_is_dead(_lock_record(home, pid), pid) is True


def test_fresh_heartbeat_keeps_lock(lock_env):
    """A ticking loop must never lose its lock."""
    home = lock_env["home"]
    pid = os.getpid()
    _write_heartbeat(home, pid, 5)

    assert status._lock_owner_loop_is_dead(_lock_record(home, pid), pid) is False


def test_heartbeat_just_inside_threshold_keeps_lock(lock_env):
    """Boundary: below the cutoff is still alive (a briefly-blocked loop)."""
    home = lock_env["home"]
    pid = os.getpid()
    _write_heartbeat(home, pid, status._LOCK_OWNER_HEARTBEAT_STALE_AFTER_S - 60)

    assert status._lock_owner_loop_is_dead(_lock_record(home, pid), pid) is False


def test_missing_heartbeat_keeps_lock(lock_env):
    """No heartbeat file is not evidence of death — never evict on absence."""
    home = lock_env["home"]
    pid = os.getpid()

    assert status._lock_owner_loop_is_dead(_lock_record(home, pid), pid) is False


def test_heartbeat_owned_by_other_pid_keeps_lock(lock_env):
    """A heartbeat from a different process proves nothing about this owner."""
    home = lock_env["home"]
    pid = os.getpid()
    # Stale heartbeat, but written by some *other* gateway.
    _write_heartbeat(home, pid + 1, status._LOCK_OWNER_HEARTBEAT_STALE_AFTER_S + 600)

    assert status._lock_owner_loop_is_dead(_lock_record(home, pid), pid) is False


def test_future_dated_heartbeat_keeps_lock(lock_env):
    """Clock skew must not be read as death."""
    home = lock_env["home"]
    pid = os.getpid()
    _write_heartbeat(home, pid, -3600)  # an hour in the future

    assert status._lock_owner_loop_is_dead(_lock_record(home, pid), pid) is False


def test_corrupt_heartbeat_keeps_lock(lock_env):
    """Unparseable heartbeat contents leave the lock alone."""
    home = lock_env["home"]
    pid = os.getpid()
    (home / "state" / "gateway.heartbeat").write_text("{not json", encoding="utf-8")

    assert status._lock_owner_loop_is_dead(_lock_record(home, pid), pid) is False


def test_unparseable_timestamp_keeps_lock(lock_env):
    """A heartbeat with a garbage updated_at is not proof of death."""
    home = lock_env["home"]
    pid = os.getpid()
    (home / "state" / "gateway.heartbeat").write_text(
        json.dumps({"pid": pid, "updated_at": "not-a-timestamp"}), encoding="utf-8"
    )

    assert status._lock_owner_loop_is_dead(_lock_record(home, pid), pid) is False


def test_record_without_hermes_home_keeps_lock(lock_env):
    """Legacy lock records with no hermes_home cannot be checked — keep them."""
    pid = os.getpid()
    record = {"pid": pid, "kind": "hermes-gateway", "start_time": 1}

    assert status._lock_owner_loop_is_dead(record, pid) is False


def test_acquire_evicts_lock_whose_owner_loop_died(lock_env, monkeypatch):
    """End-to-end: a wedged live owner loses the lock to a fresh starter.

    This is the regression that matters — every other oracle reports the
    owner as healthy, so without the heartbeat check the new gateway would
    bounce with a non-retryable conflict forever.
    """
    home = lock_env["home"]
    owner_pid = os.getpid() + 1

    # Every pre-existing oracle says "the owner is a healthy live gateway".
    monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 178841426019)
    monkeypatch.setattr(status, "_looks_like_gateway_process", lambda pid: True)

    lock_path = status._get_scope_lock_path("feishu-app-id", "app-abc")
    lock_path.write_text(json.dumps(_lock_record(home, owner_pid)), encoding="utf-8")

    # ...but its loop stopped ticking a day ago.
    _write_heartbeat(home, owner_pid, 86400)

    acquired, existing = status.acquire_scoped_lock("feishu-app-id", "app-abc")

    assert acquired is True
    record = json.loads(lock_path.read_text(encoding="utf-8"))
    assert record["pid"] == os.getpid()


def test_acquire_respects_lock_whose_owner_is_ticking(lock_env, monkeypatch):
    """A healthy owner with a fresh heartbeat keeps the lock."""
    home = lock_env["home"]
    owner_pid = os.getpid() + 1

    monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 178841426019)
    monkeypatch.setattr(status, "_looks_like_gateway_process", lambda pid: True)

    lock_path = status._get_scope_lock_path("feishu-app-id", "app-abc")
    lock_path.write_text(json.dumps(_lock_record(home, owner_pid)), encoding="utf-8")
    _write_heartbeat(home, owner_pid, 10)

    acquired, existing = status.acquire_scoped_lock("feishu-app-id", "app-abc")

    assert acquired is False
    assert existing is not None
    assert existing["pid"] == owner_pid
