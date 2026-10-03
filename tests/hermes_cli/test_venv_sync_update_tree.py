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
