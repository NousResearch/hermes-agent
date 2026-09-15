"""The update fetch reports progress while it runs, and cleans up after an interrupt (#80049, #93732).

`_fetch_with_progress` passes git's `--progress` output straight through, so a multi-minute transfer
is visibly working instead of looking like a hang, and a Ctrl-C during it reaps the fetch tree and
the aborted-fetch pack debris instead of stranding both.
"""

import _thread
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

import hermes_cli.gitlock as gitlock
import hermes_cli.update_cmd as update_cmd
from hermes_cli import update_receipt

# Emits git-style frames, spawns a descendant (git's transport child) and reports both pids, then
# keeps the pipe busy until released, so frames are readable mid-transfer and an interrupt always
# lands between reads. The descendant detaches its stdio — holding this pipe would hide EOF.
CHILD = """\
import os, pathlib, subprocess, sys, time

ready = pathlib.Path(os.environ["HG_READY"])
release = pathlib.Path(os.environ["HG_RELEASE"])
transport = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
pathlib.Path(os.environ["HG_PID_FILE"]).write_text(f"{os.getpid()} {transport.pid}")
sys.stdout.write("Receiving objects:  10% (100/1000)\\r")
sys.stdout.flush()
ready.touch()
tick = 10
while not release.exists():
    tick = (tick + 1) % 100
    sys.stdout.write("Receiving objects: %d%%\\r" % tick)
    sys.stdout.flush()
    time.sleep(0.2)
"""


def _stub(monkeypatch, repo_root: Path, **dirs: str) -> None:
    """``_m()`` stand-in (PROJECT_ROOT for the git cwd, a real ``sys`` for the passthrough writes)."""
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(PROJECT_ROOT=str(repo_root), sys=sys))
    for name, value in dirs.items():
        monkeypatch.setenv(f"HG_{name.upper()}", value)


def _fetch(tmp_path: Path):
    return update_cmd._fetch_with_progress([sys.executable, "-c", CHILD], "main", timeout_seconds=30)


def test_fetch_output_reaches_the_terminal_while_the_transfer_is_still_running(monkeypatch, tmp_path, capsys):
    """Passed through, not dumped at exit: git's frame shows while the fetch is still running."""
    release = str(tmp_path / "release")
    _stub(monkeypatch, tmp_path, ready=str(tmp_path / "ready"), release=release,
          pid_file=str(tmp_path / "child.pid"))
    results = []
    worker = threading.Thread(target=lambda: results.append(_fetch(tmp_path)))
    worker.start()
    try:
        deadline, seen = time.monotonic() + 15, ""
        while time.monotonic() < deadline and "10%" not in seen:
            seen = capsys.readouterr().out
            time.sleep(0.02)
        assert "Receiving objects:  10%" in seen
        assert worker.is_alive()
    finally:
        Path(release).touch()
        worker.join(15)
    assert [result.returncode for result in results] == [0]


def test_interrupted_fetch_kills_the_child_and_sweeps_its_pack_debris(monkeypatch, tmp_path, capsys):
    """Ctrl-C mid-fetch leaves no orphaned git child, no pack debris in .git/objects/pack,
    and an open receipt's step log ends in ``fetch_interrupted`` (the no-op-when-closed seam
    had never been exercised with a receipt actually open)."""
    repo_root = tmp_path / "repo"
    pack_dir = repo_root / ".git" / "objects" / "pack"
    pack_dir.mkdir(parents=True)
    debris = pack_dir / "tmp_pack_aborted"
    debris.write_bytes(b"\0" * 4096)
    ready, pid_file = tmp_path / "ready", tmp_path / "child.pid"
    _stub(monkeypatch, repo_root, ready=str(ready), release=str(tmp_path / "release"),
          pid_file=str(pid_file))
    monkeypatch.setattr(gitlock, "_git_proc_running", lambda: False)  # the sweep's global git probe
    update_receipt.begin_update_receipt()

    def _interrupt_once_the_child_is_up():
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(0.01)
        _thread.interrupt_main()

    interrupter = threading.Thread(target=_interrupt_once_the_child_is_up, daemon=True)
    interrupter.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            _fetch(tmp_path)
    finally:
        interrupter.join(15)

    assert ready.exists(), "the child never came up"
    child_pid, descendant_pid = (int(pid) for pid in pid_file.read_text(encoding="utf-8").split())
    for _ in range(100):
        if not psutil.pid_exists(child_pid) and not psutil.pid_exists(descendant_pid):
            break
        time.sleep(0.05)
    # A tree kill, not a launcher kill: a surviving transport child holds the pipe and its pack.
    assert not psutil.pid_exists(child_pid), "the fetch child outlived the interrupt"
    assert not psutil.pid_exists(descendant_pid), "the fetch's transport child outlived the interrupt"
    assert not debris.exists(), "the aborted fetch left its pack debris behind"
    assert "Interrupted during the fetch" in capsys.readouterr().out
    # The receipt step landed on the OPEN receipt this run opened (was: unverified end to end).
    receipt = update_receipt._current
    assert receipt is not None, "no receipt was open during the interrupt"
    steps = receipt.data["steps"]
    assert steps[-1]["name"] == "fetch_interrupted"
    assert steps[-1]["ok"] is False
    assert steps[-1]["detail"] == "SIGINT during fetch"
    update_receipt._current = None  # never leak a receipt into other tests in this file
