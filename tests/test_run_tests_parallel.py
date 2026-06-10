"""Verify scripts/run_tests_parallel.py kills test-spawned grandchildren.

Setup
-----
A test in this file spawns a long-lived Python grandchild that writes
its PID + a nonce to a tempfile, then exits without cleaning up.
With the old ``subprocess.run`` runner, that grandchild would orphan
and outlive the test (and the whole runner). With the current Popen +
``start_new_session`` + ``_kill_tree`` runner, the grandchild gets
SIGKILL'd via process-group kill when its file's pytest exits.

The leaker test always passes — its only job is to spawn a grandchild
and walk away. The verifier runs the runner over the leaker file in a
subprocess, then waits for the grandchild PID to disappear from the
kernel's process table.

POSIX-only: Windows has its own grandchild lifecycle (no shared session,
``taskkill /F /T`` semantics). Marked accordingly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


# Both tests share the same handoff file: the leaker writes here, the
# verifier reads here. We park it in $TMPDIR with a unique-per-run name
# so concurrent invocations of the suite don't clobber each other.
_HANDOFF_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "hermes-isolation-probe"
_HANDOFF_DIR.mkdir(exist_ok=True)


def _handoff_path_for(nonce: str) -> Path:
    return _HANDOFF_DIR / f"grandchild-{nonce}.json"


def _pid_alive(pid: int) -> bool:
    """POSIX: send signal 0 to probe whether ``pid`` is still alive.

    ``os.kill(pid, 0)`` raises ``ProcessLookupError`` if the process is
    gone, ``PermissionError`` if it exists but we can't signal it
    (someone else's pid). We treat PermissionError as "alive" because
    the process exists and that's all we need to know.
    """
    if sys.platform == "win32":  # pragma: no cover — POSIX-only test
        # On Windows we'd use OpenProcess + GetExitCodeProcess; this
        # test is skipped on Windows so the path is unreachable.
        raise RuntimeError("_pid_alive POSIX-only")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only probe")
@pytest.mark.live_system_guard_bypass
def test_grandchild_leak_is_killed_by_runner(tmp_path: Path) -> None:
    """Run the parallel runner over a probe file and verify cleanup.

    1. Materialize a probe file that spawns a long-lived grandchild and
       writes its PID to disk before exiting.
    2. Invoke ``scripts/run_tests_parallel.py`` against the probe file.
    3. Wait for the grandchild PID to vanish (poll for ~5s).
    4. Assert the runner exited cleanly AND the grandchild is dead.
    """
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    assert runner.exists(), f"runner missing at {runner}"

    # Probe lives in a temp dir, NOT under tests/, so the regular suite
    # never picks it up — only our explicit invocation does.
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    probe = probe_dir / "test_probe_leaker.py"
    nonce = f"{os.getpid()}-{int(time.time() * 1000)}"
    handoff = _handoff_path_for(nonce)
    if handoff.exists():
        handoff.unlink()

    probe_src = textwrap.dedent(f"""
        import json, os, subprocess, sys, time
        from pathlib import Path

        HANDOFF = Path({str(handoff)!r})

        def test_spawns_grandchild_and_walks_away():
            # Long-lived grandchild: detached, ignores SIGTERM (we want
            # SIGKILL or process-group kill to be the only thing that
            # works, simulating a misbehaving server).
            child = subprocess.Popen(
                [
                    sys.executable, "-c",
                    "import os, signal, sys, time; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "sys.stdout.write(f'gc-pgid={{os.getpgid(0)}} gc-pid={{os.getpid()}}\\\\n'); "
                    "sys.stdout.flush(); "
                    "time.sleep(600)",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # IMPORTANT: do NOT pass start_new_session here. We want
                # the grandchild to inherit the pytest subprocess's
                # process group, so when the runner kills the group the
                # grandchild dies too.
            )
            # Read the first line so we can record gc's pgid in the
            # handoff, then walk away — don't close the pipe (would
            # signal EOF and let the child see SIGPIPE on next write).
            first_line = child.stdout.readline().decode().strip()
            HANDOFF.write_text(json.dumps({{
                "pid": child.pid,
                "diag": first_line,
                "test_pid": os.getpid(),
                "test_pgid": os.getpgid(0),
            }}))
            assert child.pid > 0
    """).strip()
    probe.write_text(probe_src + "\n")

    # Run the parallel runner against just the probe file. The runner
    # discovers under ``tests/`` by default, so we override via --paths.
    proc = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--paths",
            str(probe_dir),
            "-j",
            "1",
            # Tight per-file timeout: the probe finishes in <1s, no
            # need for 10min.
            "--file-timeout",
            "30",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )

    assert handoff.exists(), (
        f"probe never wrote handoff file; runner output:\n{proc.stdout}"
    )
    handoff_data = json.loads(handoff.read_text())
    grandchild_pid = handoff_data["pid"]
    diag = handoff_data.get("diag", "(no diag)")
    test_pid = handoff_data.get("test_pid")
    test_pgid = handoff_data.get("test_pgid")
    handoff.unlink()

    # The runner must have exited cleanly (probe test passes).
    assert proc.returncode == 0, (
        f"runner exited {proc.returncode}; output:\n{proc.stdout}"
    )

    # The grandchild must be gone. Poll for a bit because process-group
    # SIGKILL + reaping isn't synchronous; on a loaded box it can take
    # a beat.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _pid_alive(grandchild_pid):
            break
        time.sleep(0.05)
    else:
        # Test cleanup: kill the leaked grandchild ourselves so a
        # FAILED assertion doesn't leave a sleep(600) running.
        try:
            os.kill(grandchild_pid, 9)
        except ProcessLookupError:
            pass
        pytest.fail(
            f"grandchild PID {grandchild_pid} survived runner exit; "
            f"diag={diag!r} test_pid={test_pid} test_pgid={test_pgid}; "
            f"runner output:\n{proc.stdout}"
        )


# ---------------------------------------------------------------------------
# exit-4 retry loop (transient "file or directory not found" on loaded runners)
# ---------------------------------------------------------------------------

import importlib.util as _importlib_util  # noqa: E402


def _load_runner_module():
    """Import scripts/run_tests_parallel.py as a module for in-process tests."""
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "scripts" / "run_tests_parallel.py"
    spec = _importlib_util.spec_from_file_location("_rtp_under_test", path)
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exit4_retry_recovers_when_file_exists(tmp_path, monkeypatch):
    """A file that exits 4 transiently then passes must be retried and recover.

    Simulates the loaded-CI transient: the per-file pytest subprocess reports
    "file or directory not found" (exit 4) on the first attempts even though
    the file is on disk, then succeeds. The runner must retry and report pass.
    """
    rtp = _load_runner_module()
    f = tmp_path / "test_transient.py"
    f.write_text("def test_ok():\n    assert True\n")

    calls = {"n": 0}

    def fake_spawn(cmd, repo_root, file_timeout, *, timeout_note="per-file timeout"):
        calls["n"] += 1
        # First two attempts: transient exit-4. Third: success.
        if calls["n"] < 3:
            return 4, "ERROR: file or directory not found\nno tests ran in 0.00s"
        return 0, "1 passed"

    monkeypatch.setattr(rtp, "_spawn_pytest_once", fake_spawn)
    monkeypatch.setattr(rtp, "_EXIT4_RETRY_BACKOFF_SECONDS", 0.0)  # no real sleep

    file, rc, output, summary, _wall = rtp._run_one_file(f, [], tmp_path, 30.0)
    assert rc == 0, f"expected recovery to pass, got rc={rc}, output={output!r}"
    assert calls["n"] == 3, f"expected 3 attempts (1 + 2 retries), got {calls['n']}"


def test_exit4_no_retry_when_file_genuinely_missing(tmp_path, monkeypatch):
    """Exit 4 on a file that does NOT exist must fail fast without retrying.

    Guards the narrowing: we only retry while the file is present on disk, so a
    real typo / deleted file surfaces immediately instead of looping.
    """
    rtp = _load_runner_module()
    missing = tmp_path / "test_does_not_exist.py"  # never created

    calls = {"n": 0}

    def fake_spawn(cmd, repo_root, file_timeout, *, timeout_note="per-file timeout"):
        calls["n"] += 1
        return 4, "ERROR: file or directory not found"

    monkeypatch.setattr(rtp, "_spawn_pytest_once", fake_spawn)
    monkeypatch.setattr(rtp, "_EXIT4_RETRY_BACKOFF_SECONDS", 0.0)

    file, rc, output, summary, _wall = rtp._run_one_file(missing, [], tmp_path, 30.0)
    assert rc == 4, f"genuinely-missing file should keep rc=4, got {rc}"
    assert calls["n"] == 1, f"missing file must NOT be retried, got {calls['n']} calls"


def test_exit4_retry_gives_up_after_max_attempts(tmp_path, monkeypatch):
    """If the transient never clears, we stop after the bounded attempt count."""
    rtp = _load_runner_module()
    f = tmp_path / "test_persistent_transient.py"
    f.write_text("def test_ok():\n    assert True\n")

    calls = {"n": 0}

    def fake_spawn(cmd, repo_root, file_timeout, *, timeout_note="per-file timeout"):
        calls["n"] += 1
        return 4, "ERROR: file or directory not found"

    monkeypatch.setattr(rtp, "_spawn_pytest_once", fake_spawn)
    monkeypatch.setattr(rtp, "_EXIT4_RETRY_BACKOFF_SECONDS", 0.0)

    file, rc, output, summary, _wall = rtp._run_one_file(f, [], tmp_path, 30.0)
    assert rc == 4
    # 1 initial + _EXIT4_RETRY_ATTEMPTS retries.
    assert calls["n"] == 1 + rtp._EXIT4_RETRY_ATTEMPTS


def test_file_present_tolerates_transient_negative(tmp_path, monkeypatch):
    """_file_present must not conclude 'missing' on a single flaky stat."""
    rtp = _load_runner_module()
    f = tmp_path / "test_flaky_stat.py"
    f.write_text("x = 1\n")

    seq = iter([False, False, True])  # first two stats flake, third succeeds
    monkeypatch.setattr(rtp.Path, "exists", lambda self: next(seq))
    assert rtp._file_present(f, attempts=3, delay=0.0) is True


def test_file_present_reports_truly_missing(tmp_path, monkeypatch):
    """_file_present returns False when the file is absent across all checks."""
    rtp = _load_runner_module()
    f = tmp_path / "nope.py"
    monkeypatch.setattr(rtp.Path, "exists", lambda self: False)
    assert rtp._file_present(f, attempts=3, delay=0.0) is False
