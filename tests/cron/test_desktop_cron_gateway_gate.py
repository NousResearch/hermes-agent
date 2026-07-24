"""#66629 — the desktop cron ticker must defer to a live gateway.

When ``hermes gateway run`` and ``hermes serve`` (desktop) share a HERMES_HOME,
the desktop ticker's standalone delivery path has no live platform adapters, so
Feishu interactive cards silently degrade to plain text. The fix gates the
desktop ticker on the gateway runtime lock: it dispatches only when no live
gateway owns this HERMES_HOME, and resumes on its own if the gateway stops.

The headline test is behavioral, through the public ticker entry point
(``hermes_cli.web_server._start_desktop_cron_ticker``): it fails before the fix
(the ungated ticker fires regardless of the gateway lock) and passes after.
The probe unit tests pin its owner-aware classification (held / free / unknown)
and, critically, the by-construction guarantee that the probe never touches the
OS lock — so a probe collision can never make gateway startup lose its own
lock (#66629 amend: observer must not kill owner, enforced by construction not
by timing).
"""
import json
import os
import threading
import time
from unittest.mock import patch

import pytest


def _wait_until(predicate, timeout=10.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _make_live_gateway_record(hermes_home):
    """Build a record for the current process framed as a running gateway."""
    import gateway.status as status
    pid = os.getpid()
    return {
        "pid": pid,
        "kind": status._GATEWAY_KIND,
        "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(pid),
        "hermes_home": str(hermes_home),
    }


def _write_record(lock_path, record):
    lock_path.write_text(json.dumps(record), encoding="utf-8")


def test_desktop_ticker_defers_to_a_live_gateway_then_resumes(monkeypatch, tmp_path):
    """RED before the gate, GREEN after. While a live gateway owns the runtime
    lock on this HERMES_HOME the desktop ticker must not drive
    ``cron.scheduler.tick`` (its standalone path degrades interactive cards).
    It resumes once the record disappears."""
    import gateway.status as status
    from hermes_cli.web_server import _start_desktop_cron_ticker
    from hermes_cli import web_server as ws

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock_path = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock_path)

    # Stand in for a live gateway by writing a record matching THIS pid — the
    # probe uses _pid_exists() + start_time + argv fingerprint to classify held.
    _write_record(lock_path, _make_live_gateway_record(tmp_path))

    # Force the argv fingerprint check to pass on this test process (we are
    # pytest, not hermes gateway run), keeping the test independent of live
    # cmdline shape while still exercising every other probe branch.
    monkeypatch.setattr(
        status,
        "_record_matches_live_gateway_pid",
        lambda record, pid, *, expected_home=None: True,
    )

    ticks = []
    stop = threading.Event()

    def fake_tick(*args, **kwargs):
        ticks.append(kwargs)
        return 0

    # Count gate calls so the quiet-interval assertion is guaranteed to run
    # AFTER the ticker started deciding to defer. Without this, a delayed
    # provider thread could reach its first tick only after the record is
    # removed and pass the quiet check without ever exercising the held path.
    gate_calls = []
    real_gate = ws._no_live_gateway

    def counting_gate():
        gate_calls.append(1)
        return real_gate()

    monkeypatch.setattr(ws, "_no_live_gateway", counting_gate)

    with patch("cron.scheduler.tick", side_effect=fake_tick):
        t = threading.Thread(
            target=_start_desktop_cron_ticker,
            args=(stop,),
            kwargs={"interval": 0.01},
            daemon=True,
        )
        t.start()
        try:
            assert _wait_until(lambda: len(gate_calls) >= 2, timeout=2.0), (
                "provider loop never consulted the gate within 2 s — the quiet "
                "interval assertion would be vacuous"
            )
            # A live gateway holds this HERMES_HOME -> the desktop ticker must stay quiet.
            time.sleep(0.2)
            assert ticks == [], (
                f"desktop ticker fired {len(ticks)} tick(s) while a live gateway "
                "was recorded (interactive cards would degrade to text)"
            )
            # Gateway stops -> record removed -> desktop cron resumes on its own.
            lock_path.unlink()
            assert _wait_until(lambda: len(ticks) >= 1), (
                "desktop ticker did not resume after the gateway record was cleared"
            )
        finally:
            stop.set()
            t.join(timeout=5)
    assert not t.is_alive()


def test_probe_never_touches_the_os_lock(monkeypatch, tmp_path):
    """By-construction guarantee (#66629 amend, Sol 3rd finding). The probe
    must never call ``_try_acquire_file_lock`` / ``msvcrt.locking`` /
    ``fcntl.flock`` on the runtime lock — an OS-lock probe would serialize
    against ``acquire_gateway_runtime_lock`` for microseconds and, under
    scheduler pressure, arbitrarily longer, making the observer kill the
    owner. Any future change that reintroduces such a probe fails this test
    deterministically (no timing dependency)."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)
    _write_record(lock, _make_live_gateway_record(tmp_path))
    monkeypatch.setattr(
        status,
        "_record_matches_live_gateway_pid",
        lambda record, pid, *, expected_home=None: True,
    )

    def _explode(_handle):
        pytest.fail(
            "probe must not call _try_acquire_file_lock (#66629 regression: "
            "an OS-lock probe would race gateway startup)"
        )

    monkeypatch.setattr(status, "_try_acquire_file_lock", _explode)
    if status._IS_WINDOWS:
        class _Explode:
            def locking(self, *a, **kw):
                pytest.fail("probe must not call msvcrt.locking (#66629 regression)")
        monkeypatch.setattr(status, "msvcrt", _Explode(), raising=False)
    else:
        class _Explode:
            LOCK_EX = 2
            LOCK_NB = 4
            LOCK_UN = 8
            LOCK_SH = 1
            def flock(self, *a, **kw):
                pytest.fail("probe must not call fcntl.flock (#66629 regression)")
        monkeypatch.setattr(status, "fcntl", _Explode(), raising=False)

    assert status.probe_gateway_runtime_lock() == "held"


def test_probe_reports_free_when_no_lock_file_exists(monkeypatch, tmp_path):
    """No lock file -> "free" (no live process owns it). The probe must not
    create the lock as a side effect."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)

    assert status.probe_gateway_runtime_lock() == "free"
    assert not lock.exists(), "probe must not create the lock file as a side effect"


def test_probe_reports_free_for_a_stale_record_from_a_dead_pid(monkeypatch, tmp_path):
    """After a crash the OS releases the lock; the JSON record is stale (pid
    dead). ``_pid_exists`` returns False -> probe returns "free" so the desktop
    cron takes over correctly."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)
    # Deliberately unassignable pid: max signed 32-bit minus one. Even on
    # systems that allow larger pids, no process will match this.
    _write_record(lock, {
        "pid": 2**31 - 2,
        "kind": status._GATEWAY_KIND,
        "argv": ["hermes", "gateway", "run"],
        "start_time": 1,
    })

    assert status.probe_gateway_runtime_lock() == "free"


def test_probe_reports_free_when_pid_reused_by_another_process(monkeypatch, tmp_path):
    """The record's pid is alive but the ``start_time`` in the record does not
    match the live process — the pid was recycled onto an unrelated process
    after the original gateway crashed. Return "free" so the desktop cron does
    not stall on a phantom gateway."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)
    # Point the record at the pytest process itself, but with a wrong
    # start_time so the PID-reuse guard fires.
    _write_record(lock, {
        "pid": os.getpid(),
        "kind": status._GATEWAY_KIND,
        "argv": ["hermes", "gateway", "run"],
        "start_time": 1,  # deliberately not the real start_time
    })

    assert status.probe_gateway_runtime_lock() == "free"


def test_probe_reports_free_when_live_pid_is_not_a_gateway(monkeypatch, tmp_path):
    """The pid is alive, the start_time matches, but the process is NOT a
    gateway (a PID was recycled and the fingerprint check
    ``_record_matches_live_gateway_pid`` rejects it). Return "free"."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)
    _write_record(lock, _make_live_gateway_record(tmp_path))
    monkeypatch.setattr(
        status,
        "_record_matches_live_gateway_pid",
        lambda record, pid, *, expected_home=None: False,
    )

    assert status.probe_gateway_runtime_lock() == "free"


def test_probe_reports_unknown_for_empty_or_unparseable_record(monkeypatch, tmp_path):
    """An empty file, or a file whose JSON cannot be parsed, means the record
    is unreadable — either a crashed mid-write or the sub-millisecond startup
    window between the OS lock and the record write. Return "unknown" so the
    gate fails open (dispatch + warning) and never stalls a desktop-only
    cron."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)

    lock.write_text("", encoding="utf-8")
    assert status.probe_gateway_runtime_lock() == "unknown"

    lock.write_text("not-json{", encoding="utf-8")
    assert status.probe_gateway_runtime_lock() == "unknown"


def test_probe_reports_held_when_this_process_holds_the_in_process_lock(monkeypatch, tmp_path):
    """Fast path: the in-process module global says this process is holding
    the lock. No file I/O; no OS lock touched."""
    import gateway.status as status

    lock = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)
    # Simulate 'this process holds it' by installing any non-None handle.
    monkeypatch.setattr(status, "_gateway_lock_handle", object())

    assert status.probe_gateway_runtime_lock() == "held"


def test_probe_reports_unknown_when_path_inspection_raises(monkeypatch, tmp_path):
    """A stat / permission error from ``Path.exists()`` must be caught as
    "unknown", not escape the documented three-valued API."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)

    def deny(self, *args, **kwargs):
        raise PermissionError("simulated stat denied")

    monkeypatch.setattr("pathlib.Path.exists", deny)
    assert status.probe_gateway_runtime_lock() == "unknown"
