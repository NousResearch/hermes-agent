"""Regression tests for Desktop-owned ``hermes serve`` lifecycle tracking."""

from hermes_cli.web_server_lifecycle import (
    _is_serve_orphaned,
    _parent_start_marker_mismatch_is_conclusive,
    _valid_parent_start_marker,
)


def test_parent_watchdog_tracks_recorded_desktop_pid_not_immediate_ppid():
    """Windows venv launch shims must not make a live Desktop look orphaned."""

    assert _is_serve_orphaned(4242, pid_exists=lambda pid: pid == 4242) is False
    assert _is_serve_orphaned(4242, pid_exists=lambda _pid: False) is True


def test_parent_watchdog_fails_safe_when_liveness_probe_errors():
    def broken_probe(_pid: int) -> bool:
        raise OSError("process table temporarily unavailable")

    assert _is_serve_orphaned(4242, pid_exists=broken_probe) is False


def test_parent_watchdog_accepts_electron_windows_creation_time_marker():
    unix_ms = 1_723_456_789_123
    dotnet_ticks = 621_355_968_000_000_000 + unix_ms * 10_000 + 9_999

    assert _valid_parent_start_marker(f"winms:{unix_ms}") is True
    assert (
        _is_serve_orphaned(
            4242,
            f"winms:{unix_ms}",
            process_start_marker=lambda _pid: f"win:{dotnet_ticks}",
        )
        is False
    )


def test_parent_watchdog_rejects_reused_pid_with_different_windows_creation_time():
    unix_ms = 1_723_456_789_123
    next_process_ticks = 621_355_968_000_000_000 + (unix_ms + 1) * 10_000

    assert (
        _is_serve_orphaned(
            4242,
            f"winms:{unix_ms}",
            process_start_marker=lambda _pid: f"win:{next_process_ticks}",
        )
        is True
    )


def test_parent_watchdog_preserves_legacy_exact_windows_marker():
    marker = "win:638908765432109876"

    assert (
        _is_serve_orphaned(
            4242,
            marker,
            process_start_marker=lambda _pid: marker,
        )
        is False
    )


def test_parent_watchdog_does_not_kill_a_live_parent_on_macos_timezone_drift():
    """#95693: the SAME instant rendered by `ps -o lstart=` under EDT (cached by
    Electron before a TZ change) vs CEST (probed by a fresh backend after) must
    degrade to the PID-only check, not count as proof the parent died."""
    expected = "ps:Thu Aug 20 22:33:11 2026"
    actual = "ps:Fri Aug 21 04:33:11 2026"

    assert (
        _is_serve_orphaned(
            4242, expected, pid_exists=lambda _pid: True, process_start_marker=lambda _pid: actual
        )
        is False
    )


def test_parent_watchdog_still_detects_a_genuinely_dead_parent_despite_ps_marker_mismatch():
    expected = "ps:Thu Aug 20 22:33:11 2026"
    actual = "ps:Fri Aug 21 04:33:11 2026"

    assert (
        _is_serve_orphaned(
            4242, expected, pid_exists=lambda _pid: False, process_start_marker=lambda _pid: actual
        )
        is True
    )


def test_parent_watchdog_exact_ps_marker_match_still_short_circuits():
    marker = "ps:Thu Aug 20 22:33:11 2026"

    assert (
        _is_serve_orphaned(
            4242, marker, pid_exists=lambda _pid: False, process_start_marker=lambda _pid: marker
        )
        is False
    )


def test_parent_watchdog_still_rejects_recycled_pid_via_stable_linux_marker():
    """linux:/win: markers are machine values -- a mismatch stays conclusive."""
    assert (
        _is_serve_orphaned(
            4242,
            "linux:12345",
            pid_exists=lambda _pid: True,
            process_start_marker=lambda _pid: "linux:99999",
        )
        is True
    )
    assert _parent_start_marker_mismatch_is_conclusive("linux:123", "linux:456") is True
    assert _parent_start_marker_mismatch_is_conclusive("win:1", "winms:2") is True


def test_macos_ps_marker_requires_full_lstart_not_a_truncated_weekday():
    """#98132: a whitespace-split ``ps:Sat`` must not arm the watchdog."""
    assert _valid_parent_start_marker("ps:Sat Aug 29 15:04:31 2026") is True
    assert _valid_parent_start_marker("ps:Sat") is False
    assert _valid_parent_start_marker("ps:Sat Aug 29") is False


def test_parent_watchdog_treats_empty_marker_env_as_absent(monkeypatch):
    """Blank inherited HERMES_PARENT_START_MARKER/NONCE degrade to PID-only tracking."""
    from hermes_cli import web_server_lifecycle

    monkeypatch.setenv("HERMES_PARENT_PID", "4242")
    monkeypatch.setenv("HERMES_PARENT_START_MARKER", "")
    monkeypatch.setenv("HERMES_PARENT_NONCE", "")
    seen = {}

    def fake_orphaned(pid, marker):
        seen["args"] = (pid, marker)
        raise SystemExit  # stop the loop thread before it sleeps/exits

    monkeypatch.setattr(web_server_lifecycle, "_is_serve_orphaned", fake_orphaned)

    class _Thread:
        def __init__(self, target, **_kw):
            self.target = target

        def start(self):
            try:
                self.target()
            except SystemExit:
                pass

    monkeypatch.setattr(web_server_lifecycle.threading, "Thread", _Thread)
    web_server_lifecycle._start_parent_death_watchdog()
    assert seen["args"] == (4242, None)


def test_parent_watchdog_warns_when_disarmed_by_unusable_marker(monkeypatch, caplog):
    """Disarming is fail-safe but must leave a trace in the log."""
    import logging

    from hermes_cli import web_server_lifecycle

    monkeypatch.setenv("HERMES_PARENT_PID", "4242")
    monkeypatch.setenv("HERMES_PARENT_START_MARKER", "ps:Sat")
    monkeypatch.setenv("HERMES_PARENT_NONCE", "n")
    started = []
    monkeypatch.setattr(
        web_server_lifecycle.threading, "Thread", lambda *a, **k: started.append(1) or _NoThread()
    )
    with caplog.at_level(logging.WARNING, logger="hermes_cli.web_server"):
        web_server_lifecycle._start_parent_death_watchdog()
    assert started == [], "an unusable marker must disarm, not arm, the watchdog"
    assert any("watchdog disabled" in r.getMessage() for r in caplog.records)


class _NoThread:
    def start(self):
        raise AssertionError("watchdog thread must not start")


def test_parent_watchdog_detects_dead_parent_when_start_marker_probe_raises_oserror():
    """#80204: when marker probe fails with OSError on a dead parent, watchdog
    must degrade to PID liveness check and reap the orphan, not return False."""
    def broken_marker_probe(_pid: int) -> str:
        raise OSError("ps could not inspect PID 4242: ps: 4242: No such process")

    # Parent is dead: pid_exists returns False -> must return True (orphaned)
    assert (
        _is_serve_orphaned(
            4242,
            "ps:Thu Aug 20 22:33:11 2026",
            pid_exists=lambda _pid: False,
            process_start_marker=broken_marker_probe,
        )
        is True
    )

    # Parent is still alive: pid_exists returns True -> fail-safe False
    assert (
        _is_serve_orphaned(
            4242,
            "ps:Thu Aug 20 22:33:11 2026",
            pid_exists=lambda _pid: True,
            process_start_marker=broken_marker_probe,
        )
        is False
    )


def test_parent_watchdog_detects_dead_parent_on_process_lookup_error():
    """#80204: ProcessLookupError from probe immediately signals parent is gone."""
    def lookup_error_probe(pid: int) -> str:
        raise ProcessLookupError(pid)

    assert (
        _is_serve_orphaned(
            4242,
            "ps:Thu Aug 20 22:33:11 2026",
            pid_exists=lambda _pid: True,
            process_start_marker=lookup_error_probe,
        )
        is True
    )


def test_ps_process_start_marker_raises_process_lookup_error_on_missing_process(monkeypatch):
    """#80204: _process_start_marker raises ProcessLookupError when ps exits with known missing-process markers."""
    import subprocess
    import pytest
    from hermes_cli import web_server_lifecycle

    # returncode != 1 but explicit "No such process" in stderr
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=2, stdout="", stderr="ps: 4242: No such process")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(web_server_lifecycle.sys, "platform", "darwin")
    monkeypatch.setattr(web_server_lifecycle.os, "name", "posix")

    with pytest.raises(ProcessLookupError):
        web_server_lifecycle._process_start_marker(4242)


def test_ps_process_start_marker_raises_oserror_on_unrelated_nonzero_failure(monkeypatch):
    """#80204: unrelated ps failures (e.g. returncode 2 without missing-process message) must raise OSError."""
    import subprocess
    import pytest
    from hermes_cli import web_server_lifecycle

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=2, stdout="", stderr="ps: temporary process table failure")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(web_server_lifecycle.sys, "platform", "darwin")
    monkeypatch.setattr(web_server_lifecycle.os, "name", "posix")

    with pytest.raises(OSError) as excinfo:
        web_server_lifecycle._process_start_marker(4242)
    assert "ps could not inspect PID 4242" in str(excinfo.value)
    assert not isinstance(excinfo.value, ProcessLookupError)


def test_parent_watchdog_does_not_kill_live_parent_on_unrelated_ps_failure(monkeypatch):
    """#80204: an unrelated nonzero ps error raises OSError, degrading to pid_exists which keeps a live backend alive."""
    import subprocess
    from hermes_cli import web_server_lifecycle

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=2, stdout="", stderr="ps: temporary process table failure")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(web_server_lifecycle.sys, "platform", "darwin")
    monkeypatch.setattr(web_server_lifecycle.os, "name", "posix")

    # Parent is alive: must return False (do not kill healthy backend)
    assert (
        web_server_lifecycle._is_serve_orphaned(
            4242,
            "ps:Thu Aug 20 22:33:11 2026",
            pid_exists=lambda _pid: True,
        )
        is False
    )

    # Parent is dead: must return True (reap orphan)
    assert (
        web_server_lifecycle._is_serve_orphaned(
            4242,
            "ps:Thu Aug 20 22:33:11 2026",
            pid_exists=lambda _pid: False,
        )
        is True
    )


