"""Regression tests for the s4-w1a extraction of the Windows-quarantine cluster.

The Windows hermes.exe quarantine helpers moved from ``hermes_cli/main.py`` to
``hermes_cli/win_quarantine.py`` (main.py decomposition, mechanical move,
wave-1 shard s4 cluster c7). This suite pins the extraction contract:

1. ``hermes_cli.main`` re-exports every moved name, so callers and test
   monkeypatches on ``hermes_cli.main.<name>`` keep resolving to the SAME
   function object as before the move.
2. References to helpers whose canonical home is ``hermes_cli.main`` (and
   moved-but-test-patched siblings) are routed through a lazy ``_m()`` main
   reference, so monkeypatching ``hermes_cli.main.<name>`` still reaches this
   code path (``PROJECT_ROOT``, ``_is_windows``, ``_load_console_script_names``,
   ``_run_install_with_heartbeat``, ...).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import hermes_cli.main as main_mod
from hermes_cli import win_quarantine as wq

MOVED = [
    "_cleanup_quarantined_exes",
    "_hermes_exe_shims",
    "_is_windows",
    "_quarantine_running_hermes_exe",
    "_restore_quarantined_exes",
    "_run_quarantined_install",
    "_schedule_replace_on_reboot",
    "_venv_scripts_dir",
]


def test_main_reexports_same_object():
    """Re-export seam: hermes_cli.main.<name> IS the moved function object."""
    for name in MOVED:
        assert getattr(main_mod, name) is getattr(wq, name)


def test_hermes_exe_shims_empty_off_windows(monkeypatch):
    """_m() routing: _is_windows patched on hermes_cli.main reaches the module."""
    monkeypatch.setattr(main_mod, "_is_windows", lambda: False)
    assert wq._hermes_exe_shims(Path("x")) == []


def test_hermes_exe_shims_windows_names(monkeypatch, tmp_path):
    monkeypatch.setattr(main_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(main_mod, "_load_console_script_names", lambda: ["hermes"])
    shims = wq._hermes_exe_shims(tmp_path)
    names = {p.name for p in shims}
    assert {"hermes.exe", "hermes-gateway.exe"} <= names
    assert all(p.parent == tmp_path for p in shims)


def test_venv_scripts_dir_follows_main_project_root(monkeypatch, tmp_path):
    monkeypatch.setattr(main_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main_mod, "_is_windows", lambda: True)
    assert wq._venv_scripts_dir() is None  # no venv dir yet
    (tmp_path / "venv" / "Scripts").mkdir(parents=True)
    assert wq._venv_scripts_dir() == tmp_path / "venv" / "Scripts"


def test_quarantine_uses_main_patched_seams(monkeypatch, tmp_path):
    """_m() routing: shims + reboot-schedule patches on hermes_cli.main reach it."""
    shim = tmp_path / "hermes.exe"
    shim.write_bytes(b"locked")

    monkeypatch.setattr(main_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(main_mod, "_hermes_exe_shims", lambda d: [shim])
    scheduled = []
    monkeypatch.setattr(
        main_mod,
        "_schedule_replace_on_reboot",
        lambda s, q: scheduled.append((s, q)) or True,
    )

    def always_fails(self, target):
        raise OSError(32, "simulated lock")

    with patch.object(Path, "rename", always_fails), patch(
        "time.sleep", lambda *_a, **_k: None
    ):
        pairs = wq._quarantine_running_hermes_exe(tmp_path)

    assert scheduled and scheduled[0][0] == shim
    assert pairs == []  # reboot-deferred pairs are not roll-back candidates


def test_run_quarantined_install_heartbeat_routes_through_main(monkeypatch):
    """_m() routing: _run_install_with_heartbeat patched on hermes_cli.main."""
    calls = []
    monkeypatch.setattr(
        main_mod,
        "_run_install_with_heartbeat",
        lambda cmd, **kw: calls.append((cmd, kw)),
    )
    wq._run_quarantined_install(["uv", "pip"], scripts_dir=None)
    assert calls and calls[0][0] == ["uv", "pip"]


def test_cleanup_quarantined_exes_uses_main_scripts_dir(monkeypatch, tmp_path):
    stale = tmp_path / "hermes.exe.old.123"
    stale.write_bytes(b"")
    other = tmp_path / "keep.exe"
    other.write_bytes(b"")

    monkeypatch.setattr(main_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(main_mod, "_venv_scripts_dir", lambda: tmp_path)
    wq._cleanup_quarantined_exes()
    assert not stale.exists()
    assert other.exists()
