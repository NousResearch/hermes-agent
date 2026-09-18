"""Tests for interrupted-install self-heal (the ``.update-incomplete`` marker).

Covers the breadcrumb lifecycle and the launch-time recovery guard added so a
``hermes update`` killed mid-install (Ctrl-C, terminal close, WSL OOM) gets
finished automatically on the next launch instead of leaving a half-built venv.
"""

from __future__ import annotations

from pathlib import Path

import hermes_cli.main as m
import hermes_cli.main_install_repair as hermes_cli_main_install_repair
from hermes_cli import main_install_repair
from hermes_cli import update_cmd


def test_marker_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    marker = m._update_marker_path()
    assert marker == tmp_path / ".update-incomplete"
    assert not marker.exists()

    m._write_update_incomplete_marker()
    assert marker.exists()
    body = marker.read_text()
    assert "started=" in body
    assert "pid=" in body

    m._clear_update_incomplete_marker()
    assert not marker.exists()






def _stub_install_env(monkeypatch, m, seen):
    """Common stubs so recovery's install path is inert and observable."""
    import hermes_cli.main_install_repair as hermes_cli_main_install_repair

    class R:
        returncode = 0

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(m, "_is_termux_env", lambda *a, **k: False)
    monkeypatch.setattr(hermes_cli_main_install_repair, "_is_termux_env", lambda *a, **k: False)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: None)
    # The install executor moved to hermes_cli._install_repair (shared between
    # the pre-import early pass and this late recovery path) — stub WHERE it
    # is executed, not the legacy main.py wrapper it replaced.
    import hermes_cli._install_repair as ir

    monkeypatch.setattr(
        ir, "run_core_install", lambda _root: seen.__setitem__("install", True)
    )


def test_recovery_self_lock_does_not_clear_core_marker_via_import_probes(
    tmp_path, monkeypatch
):
    # ``.update-incomplete`` is the generic core-install marker. Healthy
    # lazy-refresh import probes alone must NOT clear it and skip full
    # reinstall — a missing dep outside the 7-probe set would look healthy
    # (#58004 review blocker).
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    m._write_update_incomplete_marker()

    scripts_dir = tmp_path / "venv" / "Scripts"
    scripts_dir.mkdir(parents=True)
    shim = scripts_dir / "hermes.exe"
    shim.write_text("")

    monkeypatch.setattr(m, "_is_windows", lambda: True)
    monkeypatch.setattr(hermes_cli_main_install_repair, "_is_windows", lambda: True)
    monkeypatch.setattr(m, "_venv_scripts_dir", lambda: scripts_dir)
    monkeypatch.setattr(hermes_cli_main_install_repair, "_venv_scripts_dir", lambda: scripts_dir)
    monkeypatch.setattr(main_install_repair, "_hermes_exe_shims", lambda d: [shim])
    monkeypatch.setattr(main_install_repair, "_default_venv_install_target",
        lambda: (["uv", "pip"], {"VIRTUAL_ENV": str(tmp_path / "venv")}),
    )
    monkeypatch.setattr(
        m, "_repair_venv_via_import_probes", lambda *a, **k: "healthy"
    )
    monkeypatch.setattr(
        hermes_cli_main_install_repair, "_repair_venv_via_import_probes", lambda *a, **k: "healthy"
    )

    class FakeProc:
        def __init__(self, exe_path):
            self._exe = exe_path

        def exe(self):
            return self._exe

        def parents(self):
            return [FakeProc(str(shim))]

    monkeypatch.setattr("psutil.Process", lambda: FakeProc(sys_executable_path()))

    seen = {"install": False}
    _stub_install_env(monkeypatch, m, seen)

    m._recover_from_interrupted_install()

    assert seen["install"] is True, "core marker still requires full reinstall"
    assert not m._update_marker_path().exists(), "cleared only after full reinstall"




def sys_executable_path():
    import sys

    return sys.executable


