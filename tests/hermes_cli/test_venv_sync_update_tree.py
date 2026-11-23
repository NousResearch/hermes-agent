"""A launch never runs the completion tail beside a live update tree whose marker is gone.

A killed ``hermes update`` leaves its completion child holding the checkout lock; its marker is
dead (and any reader deletes it). ``prepare_launch`` must read the checkout lock, not just the
marker, or it claims the free marker and runs a second tail beside the live one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import venv_sync

REPO = Path(__file__).resolve().parents[2]

_HOLDER = """
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock
assert UpdateLock(path=Path(sys.argv[3]), install_root=sys.argv[2]).acquire()
print("held", flush=True)
time.sleep(120)
"""


def test_launch_refuses_while_another_process_holds_the_checkout(tmp_path, monkeypatch):
    import pm

    root = tmp_path / "checkout"
    (root / ".git").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='example'\n", encoding="utf-8")
    (root / "install-stamp.json").write_text('{"updateMechanism": "self"}', encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: False)  # an interrupted update's tree
    finished = []
    monkeypatch.setattr(venv_sync, "_finish_source_update", lambda *a, **k: finished.append(a))

    # The orphaned completion child: a real process holding the checkout lock; its marker
    # lives elsewhere, so this home's marker is free.
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(REPO), str(root), str(tmp_path / "gone")],
                              stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True, encoding="utf-8")
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        with pytest.raises(RuntimeError, match="an update is still running"):
            venv_sync.prepare_launch(root, [])
        assert finished == [], "a second completion tail ran beside the live update tree"
        assert not (tmp_path / "home" / ".hermes-update-in-progress").exists(), "a refused launch left a claim"
    finally:
        holder.kill()
        holder.wait()


def _tail_root(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """A checkout owing the tail, whose completion child only records that it ran."""
    import pm.environments

    root = tmp_path / "checkout"
    (root / "hermes_cli").mkdir(parents=True)
    started = tmp_path / "tail-ran"
    child = f"from pathlib import Path\nPath({str(started)!r}).touch()\n"
    (root / "hermes_cli" / "source_completion.py").write_text(child, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(venv_sync, "_tree_matches_completed_stamp", lambda root: False)
    monkeypatch.setattr(pm.environments, "activation_environment", lambda root: None)
    return root, started


def test_a_tail_child_custody_refuses_never_runs_and_stays_owed(tmp_path, monkeypatch):
    """N10/R6b: the launch-time tail writes the checkout, so its child is started in the update's
    custody. One custody refuses (Windows: the job would not take it) never runs unfenced: the
    launch fails as a failed tail does, so the tail stays owed."""
    from hermes_cli import update_custody

    root, started = _tail_root(tmp_path, monkeypatch)

    def refused(argv, **kwargs):
        raise update_custody._refuse(argv, OSError(5, "Access is denied"))

    monkeypatch.setattr(update_custody, "run", refused)
    with pytest.raises(RuntimeError, match="source update completion failed"):
        venv_sync._finish_source_update(root, current=True, pending=root / ".update-pending")
    assert not started.exists(), "the tail child ran outside the update's custody"


@pytest.mark.platforms("windows")
def test_a_tail_child_the_job_refuses_never_runs(tmp_path, monkeypatch):
    """N10/R6b native: with the launch holding the checkout lock, the REAL bind is refused (the
    job handle is an event); the tail child must not run outside the job."""
    import ctypes

    from hermes_cli import update_lock

    root, started = _tail_root(tmp_path, monkeypatch)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateEventW.restype = ctypes.c_void_p
    event = kernel32.CreateEventW(None, True, False, None)
    monkeypatch.setattr(update_lock, "update_tree_job", lambda: event)
    lock = update_lock.UpdateLock(path=tmp_path / "marker", install_root=root, checkout_first=False)
    assert lock.acquire_checkout(root)
    try:
        with pytest.raises(RuntimeError, match="source update completion failed"):
            venv_sync._finish_source_update(root, current=True, pending=root / ".update-pending")
    finally:
        lock.release()
    assert not started.exists(), "the tail child ran outside the update's job"
