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


def test_probe_reports_unknown_when_second_exists_raises_after_first_succeeds(
    monkeypatch, tmp_path,
):
    """Sol 4th finding 2 — TOCTOU. The outer Path.exists() succeeds; then
    permissions flip before _read_pid_record's OWN exists() runs, and its
    PermissionError escapes an inner "if not exists()" that lives outside the
    inner try. probe_gateway_runtime_lock() must catch it and stay tri-state."""
    import pathlib
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock = tmp_path / "gateway.lock"
    lock.write_text(json.dumps({"pid": 1, "kind": status._GATEWAY_KIND}), encoding="utf-8")
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)

    real_exists = pathlib.Path.exists
    calls = {"n": 0}

    def flipping_exists(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return True  # outer probe check: file is there.
        raise PermissionError("simulated permission flip")

    monkeypatch.setattr(pathlib.Path, "exists", flipping_exists)
    try:
        assert status.probe_gateway_runtime_lock() == "unknown"
    finally:
        monkeypatch.setattr(pathlib.Path, "exists", real_exists)


def test_probe_reports_unknown_when_recorded_start_time_is_null(monkeypatch, tmp_path):
    """Sol 4th finding 1 — a null recorded start_time cannot positively
    establish identity. probe must return "unknown" (fail open, dispatch), not
    "held", so a dead profile's stale record whose PID got recycled onto a
    live gateway on another profile does NOT stall this profile's cron."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)
    _write_record(lock, {
        "pid": os.getpid(),
        "kind": status._GATEWAY_KIND,
        "argv": ["hermes", "gateway", "run"],
        "start_time": None,  # a record that never received an OS timestamp.
        "hermes_home": str(tmp_path),
    })

    assert status.probe_gateway_runtime_lock() == "unknown"


def test_probe_reports_unknown_when_live_start_time_is_none(monkeypatch, tmp_path):
    """The live PID exists but its start_time cannot be read (permission,
    unsupported platform). Identity is unconfirmable -> "unknown"."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)
    _write_record(lock, _make_live_gateway_record(tmp_path))
    # Force _get_process_start_time() to return None for this pid.
    monkeypatch.setattr(status, "_get_process_start_time", lambda _pid: None)

    assert status.probe_gateway_runtime_lock() == "unknown"


def test_probe_reports_free_when_record_hermes_home_differs_from_probed(
    monkeypatch, tmp_path,
):
    """Sol 4th finding 1 (part 2) — cross-profile: the record's stored
    hermes_home does not match the probed lock's HERMES_HOME. This means the
    record was written by a gateway on a different profile (either a foreign
    write, or the record survived a profile move). The probe must NOT trust
    it; return "free"."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock = tmp_path / "gateway.lock"
    other_home = tmp_path / "other-profile"
    other_home.mkdir()
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)
    # Record is authentic for the CURRENT process but was written for a
    # different HERMES_HOME than this lock's parent (tmp_path).
    rec = _make_live_gateway_record(tmp_path)
    rec["hermes_home"] = str(other_home)
    _write_record(lock, rec)
    # Force the fingerprint check to pass so this test isolates the
    # hermes_home mismatch check itself. Without this pin, the pre-fix
    # implementation would return "free" via the fingerprint path (pytest
    # is not a gateway), producing a false-green.
    monkeypatch.setattr(
        status,
        "_record_matches_live_gateway_pid",
        lambda record, pid, *, expected_home=None: True,
    )

    assert status.probe_gateway_runtime_lock() == "free"


def test_probe_passes_expected_home_to_gateway_fingerprint(monkeypatch, tmp_path):
    """Sol 4th finding 1 (part 3) — the live-cmdline fingerprint check must be
    called with expected_home derived from the probed lock. Without it, a PID
    recycled onto another profile's live gateway would still pass the "is a
    gateway" check and return "held". Assert the argument is passed."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)
    _write_record(lock, _make_live_gateway_record(tmp_path))

    captured = {}

    def fake_match(record, pid, *, expected_home=None):
        captured["expected_home"] = expected_home
        return True

    monkeypatch.setattr(status, "_record_matches_live_gateway_pid", fake_match)

    assert status.probe_gateway_runtime_lock() == "held"
    assert captured["expected_home"] is not None, (
        "probe must pass expected_home so cross-profile PID reuse does not falsely fingerprint"
    )
    assert str(captured["expected_home"]) == str(status._canonical_hermes_home(tmp_path))


def test_probe_reports_unknown_when_recorded_home_is_not_a_usable_string(
    monkeypatch, tmp_path,
):
    """Sol 5th #1 — a record whose ``hermes_home`` is missing, empty, or not a
    string (a corrupt record, or one written before the field existed) cannot
    positively prove profile identity. The probe must return "unknown" (fail
    open) rather than fall through to the argv matcher. Pinned with the matcher
    forced True: pre-fix a homeless record reached it and answered "held"."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)
    monkeypatch.setattr(
        status,
        "_record_matches_live_gateway_pid",
        lambda record, pid, *, expected_home=None: True,
    )

    base = _make_live_gateway_record(tmp_path)
    for bad in ("__missing__", "", "   ", 12345, ["/x"], {"p": 1}):
        rec = dict(base)
        if bad == "__missing__":
            rec.pop("hermes_home", None)
        else:
            rec["hermes_home"] = bad
        _write_record(lock, rec)
        assert status.probe_gateway_runtime_lock() == "unknown", (
            f"homeless/garbage record (hermes_home={bad!r}) must be indeterminate, "
            "not trusted as held"
        )


def test_probe_delegates_hermes_home_equality_to_platform_comparator(
    monkeypatch, tmp_path,
):
    """Sol 5th #3 — HERMES_HOME identity must be compared with the host's
    path + case semantics (``_same_hermes_home``), not raw string equality, so
    representations that differ only in case (Windows) or normalization still
    name the same directory. Proven by forcing the comparator's verdict on a
    record whose home is raw-equal to the probed home and asserting the probe
    honors it — pre-fix used ``str(...) != str(...)`` and ignored the patch."""
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

    # Comparator says "different" for a raw-equal pair -> a probe that honors the
    # comparator returns "free"; one still using raw `==` ignores it and wrongly
    # returns "held".
    monkeypatch.setattr(status, "_same_hermes_home", lambda a, b: False)
    assert status.probe_gateway_runtime_lock() == "free"
    # Inverse: comparator says "same" -> held.
    monkeypatch.setattr(status, "_same_hermes_home", lambda a, b: True)
    assert status.probe_gateway_runtime_lock() == "held"


def test_probe_survives_recursion_error_from_deeply_nested_record(
    monkeypatch, tmp_path,
):
    """Sol 5th #4 — a deeply-nested corrupt JSON lock record makes ``json.loads``
    raise ``RecursionError`` (a RuntimeError subclass, NOT JSONDecodeError or
    OSError), which escaped the probe's ``except OSError``. The record read must
    treat it as unreadable and the probe must return "unknown" (fail open),
    not crash."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)
    depth = 20000  # well past the interpreter recursion limit for json's decoder
    lock.write_text("[" * depth + "]" * depth, encoding="utf-8")

    assert status.probe_gateway_runtime_lock() == "unknown"


def test_command_line_belongs_to_profile_requires_an_exact_profile_token(tmp_path):
    """Sol 5th #2 (root) — profile scoping must match an EXACT argv token, not a
    substring. ``work`` must not match a live ``--profile worker``; the real
    ``work`` selector (space / ``=`` / ``-p`` forms, and an explicit
    ``HERMES_HOME=``) must match."""
    import gateway.status as status

    work_home = tmp_path / "profiles" / "work"
    m = status._command_line_belongs_to_profile
    # The substring trap: sibling profile whose name has "work" as a prefix.
    assert m("hermes gateway run --profile worker", work_home) is False
    assert m("hermes gateway run -p worker", work_home) is False
    assert m("hermes gateway run --profile=worker", work_home) is False
    # Genuine selectors for "work" in every accepted form.
    assert m("hermes gateway run --profile work", work_home) is True
    assert m("hermes gateway run -p work", work_home) is True
    assert m("hermes gateway run --profile=work", work_home) is True
    assert m(f"HERMES_HOME={work_home} hermes gateway run", work_home) is True
    # A named profile with no selector at all does not match.
    assert m("hermes gateway run", work_home) is False


def test_command_line_belongs_to_profile_treats_explicit_profile_as_authoritative(tmp_path):
    """Sol 6th #2 (second pass) — an explicit ``--profile``/``-p`` selector is
    authoritative (Hermes' ``_apply_profile_override`` consumes it); a
    conflicting ``HERMES_HOME=`` token on the SAME argv must not override it. A
    live ``--profile worker`` process is not the "work" profile even if its argv
    also carries ``HERMES_HOME=<work>``."""
    import gateway.status as status

    work_home = tmp_path / "profiles" / "work"
    worker_home = tmp_path / "profiles" / "worker"
    m = status._command_line_belongs_to_profile
    cmd = f"HERMES_HOME={work_home} hermes gateway run --profile worker"
    assert m(cmd, work_home) is False   # explicit --profile worker wins over HERMES_HOME=work
    assert m(cmd, worker_home) is True  # it genuinely belongs to worker


def test_probe_reports_free_when_live_argv_names_a_sibling_profile(
    monkeypatch, tmp_path,
):
    """Sol 5th #2 (the exact reproduction) — probe profile "work" whose lock
    holds a record that HONESTLY recorded ``hermes_home=work`` (written by
    work's now-dead gateway) with a pid + start_time that, after PID reuse and a
    start-time collision, now identify a LIVE process running ``--profile
    worker``. Home matches (both work); pre-fix the substring matcher confirmed
    via ``--profile work`` ⊂ ``--profile worker`` and the probe answered "held",
    stalling work's cron. With exact-token matching the live sibling gateway is
    rejected and the probe returns "free" (work has no live gateway)."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    work_home = tmp_path / "profiles" / "work"
    work_home.mkdir(parents=True)
    lock = work_home / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)
    # The live process (this pid) is a gateway for the sibling profile "worker".
    monkeypatch.setattr(
        status, "_read_process_cmdline", lambda pid: "hermes gateway run --profile worker"
    )
    # Honest stale record: home=work, pid+start now collide with the live worker.
    rec = {
        "pid": os.getpid(),
        "kind": status._GATEWAY_KIND,
        "argv": ["hermes", "gateway", "run", "--profile", "work"],
        "start_time": status._get_process_start_time(os.getpid()),
        "hermes_home": str(work_home),
    }
    _write_record(lock, rec)

    assert status.probe_gateway_runtime_lock() == "free"


def test_probe_reports_unknown_when_a_homeless_record_cannot_scope_a_profile(
    monkeypatch, tmp_path,
):
    """A record that omits ``hermes_home`` cannot positively scope a profile, so
    the probe returns "unknown" (fail open) before the argv matcher is even
    consulted — independent of whatever the live argv looks like."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    work_home = tmp_path / "profiles" / "work"
    work_home.mkdir(parents=True)
    lock = work_home / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)
    monkeypatch.setattr(
        status, "_read_process_cmdline", lambda pid: "hermes gateway run --profile work"
    )
    rec = {
        "pid": os.getpid(),
        "kind": status._GATEWAY_KIND,
        "argv": ["hermes", "gateway", "run", "--profile", "work"],
        "start_time": status._get_process_start_time(os.getpid()),
        # deliberately no hermes_home
    }
    _write_record(lock, rec)

    assert status.probe_gateway_runtime_lock() == "unknown"


def test_probe_holds_for_a_named_profile_when_home_confirms_identity(
    monkeypatch, tmp_path,
):
    """The fix must not over-tighten. A record that DOES carry this named
    profile's ``hermes_home``, written by a live gateway-shaped process on the
    same profile, still resolves to "held"."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    work_home = tmp_path / "profiles" / "work"
    work_home.mkdir(parents=True)
    lock = work_home / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock)
    monkeypatch.setattr(
        status, "_read_process_cmdline", lambda pid: "hermes gateway run --profile work"
    )
    rec = {
        "pid": os.getpid(),
        "kind": status._GATEWAY_KIND,
        "argv": ["hermes", "gateway", "run", "--profile", "work"],
        "start_time": status._get_process_start_time(os.getpid()),
        "hermes_home": str(work_home),
    }
    _write_record(lock, rec)

    assert status.probe_gateway_runtime_lock() == "held"
