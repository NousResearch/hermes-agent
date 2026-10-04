"""posix.sh owns the update marker like a lock (contract C1/C2) and reports a committed update as
committed (contract C3). Every case runs the real script against a disposable install; liveness
is decided against real processes, never a stub."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest

from tests.installation_launcher_fixture import publish_fixture_launcher

pytestmark = pytest.mark.platforms("linux")  # /proc creation times

POSIX = Path(__file__).resolve().parents[3] / "scripts" / "desktop-update" / "posix.sh"
FAKE_CLI = """
import os, sys
from pathlib import Path
if '--help' in sys.argv:
    print('update options --keep-stash'); sys.exit(0)
with open(os.environ['HANDOFF_CAPTURE'], 'a', encoding='utf-8') as f:
    f.write(' '.join(sys.argv[1:]) + '\\n')
if sys.argv[1:2] == ['desktop']:
    sys.exit(1)
completion = os.environ.get('HANDOFF_COMPLETION')
if completion and sys.argv[1:2] == ['update']:  # a survivor that keeps the checkout lock past our exit
    import subprocess
    subprocess.Popen([sys.executable, '-c', (
        'import fcntl, os, sys, time; from pathlib import Path; '
        'fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644); fcntl.flock(fd, fcntl.LOCK_EX); '
        'Path(sys.argv[2] + ".ready").write_text(str(os.getpid())); '
        'exec("while not Path(sys.argv[2]).exists(): time.sleep(0.05)")'),
        os.environ['HANDOFF_CHECKOUT_LOCK'], completion], start_new_session=True)
    while not Path(completion + '.ready').exists():
        __import__('time').sleep(0.02)
hold = os.environ.get('HANDOFF_HOLD')
if hold:  # an update still running: report the pid, then wait to be released
    Path(hold + '.pid').write_text(str(os.getpid()), encoding='utf-8')
    while not Path(hold).exists():
        __import__('time').sleep(0.05)
print(os.environ.get('HANDOFF_OUTPUT', ''))
sys.exit(int(os.environ.get('HANDOFF_EXIT', '0')))
"""


def _ct(pid: int) -> str:
    rest = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8-sig").rsplit(") ", 1)[1].split()
    btime = next(int(l.split()[1]) for l in Path("/proc/stat").read_text(encoding="utf-8-sig").splitlines() if l.startswith("btime "))
    return f"{btime + int(rest[19]) / os.sysconf('SC_CLK_TCK'):.3f}"


@pytest.fixture
def sleeper():
    procs: list[subprocess.Popen] = []

    def spawn() -> subprocess.Popen:
        proc = subprocess.Popen(["sleep", "300"])
        procs.append(proc)
        return proc

    yield spawn
    for proc in procs:
        proc.kill()
        proc.wait()


def _install(tmp_path: Path, *, legacy: bool = False) -> tuple[Path, Path]:
    home = tmp_path / "home"
    install = home / "hermes-agent"
    package = install / "hermes_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "main.py").write_text(FAKE_CLI, encoding="utf-8")
    if legacy:
        bin_dir = install / "venv" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "python3").symlink_to(sys.executable)
        hermes = bin_dir / "hermes"
        hermes.write_text(f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} -m hermes_cli.main "$@"\n', encoding="utf-8")
        hermes.chmod(0o755)
    else:
        (install / "pm").mkdir()
        publish_fixture_launcher(install, FAKE_CLI)
    return home, install


def _run(tmp_path: Path, home: Path, install: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    full_env = {**os.environ, "HOME": str(tmp_path), "TMPDIR": str(tmp_path), "HERMES_HOME": str(home),
                "HANDOFF_CAPTURE": str(tmp_path / "calls.txt"), "HERMES_RUNTIME_DIR": str(tmp_path / "store"),
                "HERMES_UPDATE_SHIM_GRACE_SECONDS": "0", **env}
    for key in ("PYTHONPATH", "PYTHONHOME", "HERMES_UPDATE_STARTED_AT"):
        full_env.pop(key, None)
    full_env.update(env)
    return subprocess.run(["bash", str(POSIX), "--daemonized", "--no-ui", "--install-root", str(install), *args],
                          env=full_env, cwd=tmp_path, capture_output=True, text=True, timeout=120)


def _calls(tmp_path: Path) -> list[str]:
    capture = tmp_path / "calls.txt"
    return capture.read_text(encoding="utf-8-sig").splitlines() if capture.exists() else []


@pytest.mark.parametrize("delegate", [False, True], ids=["live-owner-past-20min", "dead-owner-live-delegate"])
def test_live_marker_is_never_reclaimed_and_refuses_the_handoff(tmp_path, sleeper, delegate):
    home, install = _install(tmp_path)
    live = sleeper()
    if delegate:
        # The orchestrator died (pid 1 is never ours) while `hermes update` still runs as the
        # delegate: the marker stays LIVE through line 4.
        body = f"999999\n{int(time.time()) - 60}\nct:1.000\ndelegate:{live.pid} ct:{_ct(live.pid)}\n"
    else:
        body = f"{live.pid}\n{int(time.time()) - 3600}\nct:{_ct(live.pid)}\n"
    marker = home / ".hermes-update-in-progress"
    marker.write_text(body, encoding="utf-8")

    result = _run(tmp_path, home, install)

    assert result.returncode == 2, result.stdout + result.stderr
    assert marker.read_text(encoding="utf-8-sig") == body
    assert _calls(tmp_path) == []
    # A refused run changed nothing and owns no result (contract A4): the other update reports.
    assert not (home / ".hermes-update-result.json").exists()


def _now() -> int:
    return int(time.time())


# Contract A1/A2/A3 marker bodies, judged by the real posix.sh against a real live process; every
# other reader (Python, Electron, PowerShell, Rust) must give the same verdict.
_MATRIX = {
    "crlf-v2-matching-ct": (lambda pid, ct: f"{pid}\r\n{_now()}\r\nct:{ct}\r\n", "live"),
    "bom-v2-matching-ct": (lambda pid, ct: f"\ufeff{pid}\n{_now()}\nct:{ct}\n", "live"),
    "fractional-started-at": (lambda pid, ct: f"{pid}\n{_now()}.5\nct:{ct}\n", "dead"),
    "missing-line-2": (lambda pid, ct: f"{pid}\n", "dead"),
    "garbled-ct-is-v1-fresh": (lambda pid, ct: f"{pid}\n{_now()}\nct:garbage\n", "live"),
    "garbled-ct-is-v1-past-20min": (lambda pid, ct: f"{pid}\n{_now() - 1300}\nct:garbage\n", "dead"),
    "delegate-without-ct-is-ignored": (lambda pid, ct: f"999999\n{_now()}\nct:1.000\ndelegate:{pid}\n", "dead"),
    "delegate-with-ct": (lambda pid, ct: f"999999\n{_now()}\nct:1.000\ndelegate:{pid} ct:{ct}\n", "live"),
}


@pytest.mark.parametrize("case", sorted(_MATRIX))
def test_marker_bodies_are_parsed_positionally_like_every_other_reader(tmp_path, sleeper, case):
    home, install = _install(tmp_path)
    live = sleeper()
    make, verdict = _MATRIX[case]
    marker = home / ".hermes-update-in-progress"
    body = make(live.pid, _ct(live.pid)).encode("utf-8")
    marker.write_bytes(body)

    result = _run(tmp_path, home, install, "--self-test-marker", "--no-marker-cleanup")

    if verdict == "live":
        assert result.returncode == 2, result.stdout + result.stderr
        assert marker.read_bytes() == body
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert marker.read_text(encoding="utf-8-sig").split("\n")[0] != str(live.pid)


@pytest.mark.parametrize("age", [0, 60], ids=["young-claim-in-flight", "stale"])
def test_empty_marker_is_live_only_while_young(tmp_path, age):
    home, install = _install(tmp_path)
    marker = home / ".hermes-update-in-progress"
    marker.write_bytes(b"")
    os.utime(marker, (time.time() - age, time.time() - age))

    result = _run(tmp_path, home, install, "--self-test-marker", "--no-marker-cleanup")

    assert result.returncode == (2 if age == 0 else 0), result.stdout + result.stderr


@pytest.mark.parametrize("bridge", ["absent", "someone-else"])
def test_desktop_started_handoff_only_adopts_its_bridge(tmp_path, sleeper, bridge):
    """A4: the Desktop gave up on a late script (no bridge left, or another update took the marker):
    the script must not claim fresh, run, or write a result the next boot would show."""
    home, install = _install(tmp_path)
    desktop, other = sleeper(), sleeper()
    marker = home / ".hermes-update-in-progress"
    body = f"{other.pid}\n{_now()}\nct:{_ct(other.pid)}\n"
    if bridge == "someone-else":
        marker.write_text(body, encoding="utf-8")

    result = _run(tmp_path, home, install, "--desktop-pid", str(desktop.pid))

    assert result.returncode == 2, result.stdout + result.stderr
    assert _calls(tmp_path) == []
    assert not (home / ".hermes-update-result.json").exists()
    if bridge == "absent":
        assert not marker.exists()
    else:
        assert marker.read_text(encoding="utf-8-sig") == body


def test_desktop_bridge_marker_is_adopted_with_its_started_at(tmp_path, sleeper):
    home, install = _install(tmp_path)
    desktop = sleeper()
    started = int(time.time()) - 30
    marker = home / ".hermes-update-in-progress"
    marker.write_text(f"{desktop.pid}\n{started}\nct:{_ct(desktop.pid)}\n", encoding="utf-8")

    result = _run(tmp_path, home, install, "--desktop-pid", str(desktop.pid), "--self-test-marker")

    assert result.returncode == 0, result.stdout + result.stderr
    pid, started_at, ct = marker.read_text(encoding="utf-8-sig").splitlines()
    assert pid != str(desktop.pid) and int(pid) > 0
    assert started_at == str(started)
    assert ct.startswith("ct:") and len(ct.split(".")[-1]) == 3


def test_committed_update_with_failed_followup_is_ok_with_warnings(tmp_path):
    home, install = _install(tmp_path, legacy=True)

    result = _run(tmp_path, home, install, HANDOFF_OUTPUT="Desktop build failed")

    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((home / ".hermes-update-result.json").read_text(encoding="utf-8-sig"))
    assert receipt["ok"] is True and receipt["manual"] is True
    assert receipt["warnings"] and receipt["warnings"][0].startswith("desktop-rebuild:")
    assert "previous version" not in receipt["message"]
    assert isinstance(receipt["started_at"], int)
    assert not (home / ".hermes-update-in-progress").exists()


def test_interrupted_app_swap_is_rolled_back_at_the_next_run(tmp_path):
    home, install = _install(tmp_path)
    app = tmp_path / "Applications" / "Hermes.app"
    previous = app.with_name("Hermes.app.old")
    (previous / "Contents").mkdir(parents=True)
    (previous / "Contents" / "Info.plist").write_text("previous", encoding="utf-8")
    (app.with_name("Hermes.app.new") / "Contents").mkdir(parents=True)  # partial staged copy

    result = _run(tmp_path, home, install, "--relaunch-target", str(app))

    assert result.returncode == 0, result.stdout + result.stderr
    assert (app / "Contents" / "Info.plist").read_text(encoding="utf-8-sig") == "previous"
    assert not previous.exists() and not app.with_name("Hermes.app.new").exists()


def _ancestry(pid: int) -> list[int]:
    chain = []
    while pid > 1:
        chain.append(pid)
        pid = int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()[1])
    return chain


def test_script_killed_right_after_spawning_the_update_leaves_a_live_marker(tmp_path):
    """C1 rule 6, written by the script itself: posix.sh dies (SIGKILL) while its `hermes update`
    child is only starting up and has not taken the update lock. The marker must still read LIVE
    through the child named on line 4, and turn dead once that child is gone."""
    home, install = _install(tmp_path)
    hold = tmp_path / "release-update"
    marker = home / ".hermes-update-in-progress"
    env = {**os.environ, "HOME": str(tmp_path), "TMPDIR": str(tmp_path), "HERMES_HOME": str(home),
           "HANDOFF_CAPTURE": str(tmp_path / "calls.txt"), "HERMES_RUNTIME_DIR": str(tmp_path / "store"),
           "HERMES_UPDATE_SHIM_GRACE_SECONDS": "0", "HANDOFF_HOLD": str(hold)}
    for key in ("PYTHONPATH", "PYTHONHOME", "HERMES_UPDATE_STARTED_AT"):
        env.pop(key, None)
    script = subprocess.Popen(["bash", str(POSIX), "--daemonized", "--no-ui", "--install-root", str(install)],
                              env=env, cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True)
    child_pid_file = Path(str(hold) + ".pid")
    try:
        deadline = time.monotonic() + 60
        while not child_pid_file.exists():
            assert time.monotonic() < deadline and script.poll() is None, "update child never started"
            time.sleep(0.01)
        # Kill the script the moment the delegate line is there (at most 3 s after the spawn).
        settle = time.monotonic() + 3
        while time.monotonic() < settle and len(marker.read_text(encoding="utf-8-sig").splitlines()) < 4:
            time.sleep(0.005)
        script.kill()
        script.wait(timeout=10)
        child = int(child_pid_file.read_text(encoding="utf-8-sig"))
        lines = marker.read_text(encoding="utf-8-sig").splitlines()
        assert lines[0] == str(script.pid)
        assert len(lines) == 4 and lines[3].startswith("delegate:"), lines
        delegate = int(lines[3].split()[0].split(":")[1])
        assert delegate in _ancestry(child)  # the update process (or its exec-ing launcher)
        assert lines[3] == f"delegate:{delegate} ct:{_ct(delegate)}"

        # A real reader (the script itself) sees an update in progress and refuses.
        refused = _run(tmp_path, home, install, "--self-test-marker", "--no-marker-cleanup")
        assert refused.returncode == 2, refused.stdout + refused.stderr

        hold.touch()
        deadline = time.monotonic() + 30
        while Path(f"/proc/{delegate}").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        reclaimed = _run(tmp_path, home, install, "--self-test-marker", "--no-marker-cleanup")
        assert reclaimed.returncode == 0, reclaimed.stdout + reclaimed.stderr
    finally:
        hold.touch()
        if script.poll() is None:
            script.kill()
            script.wait()
