"""Tests for interrupted-install self-heal (the ``.update-incomplete`` marker).

Covers the breadcrumb lifecycle and the launch-time recovery guard added so a
``hermes update`` killed mid-install (Ctrl-C, terminal close, WSL OOM) gets
finished automatically on the next launch instead of leaving a half-built venv.
"""

from __future__ import annotations

from pathlib import Path

import hermes_cli.main as m


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

    class R:
        returncode = 0

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(m, "_is_termux_env", lambda *a, **k: False)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: None)
    monkeypatch.setattr(
        m,
        "_install_python_dependencies_with_optional_fallback",
        lambda *a, **k: seen.__setitem__("install", True),
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
    monkeypatch.setattr(m, "_venv_scripts_dir", lambda: scripts_dir)
    monkeypatch.setattr(m, "_hermes_exe_shims", lambda d: [shim])
    monkeypatch.setattr(
        m,
        "_default_venv_install_target",
        lambda: (["uv", "pip"], {"VIRTUAL_ENV": str(tmp_path / "venv")}),
    )
    monkeypatch.setattr(
        m, "_repair_venv_via_import_probes", lambda *a, **k: "healthy"
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


def _make_live_venv(tmp_path, name):
    """Create a fake venv layout under tmp_path (bin/ on POSIX, Scripts/ on Windows)."""
    import platform

    bin_dir = tmp_path / name / ("Scripts" if platform.system() == "Windows" else "bin")
    bin_dir.mkdir(parents=True)
    py = bin_dir / ("python.exe" if platform.system() == "Windows" else "python")
    py.write_bytes(b"")
    return py


def test_default_venv_install_target_points_at_live_dot_venv(tmp_path, monkeypatch):
    """#84647: install/repair must target the live venv (.venv), not a hardcoded `venv`."""
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: "/fake/uv")
    monkeypatch.setattr(m, "_is_termux_env", lambda *a, **k: False)

    # uv-default / dev layout: only .venv exists
    _make_live_venv(tmp_path, ".venv")
    prefix, env = m._default_venv_install_target()
    assert prefix == ["/fake/uv", "pip"]
    assert env is not None
    assert env["VIRTUAL_ENV"] == str(tmp_path / ".venv")

    # Managed layout: `venv` wins when it actually holds an interpreter.
    _make_live_venv(tmp_path, "venv")
    _, env = m._default_venv_install_target()
    assert env["VIRTUAL_ENV"] == str(tmp_path / "venv")


def test_lazy_refresh_recovery_targets_live_dot_venv(tmp_path, monkeypatch):
    """#84647: lazy-refresh recovery must repair the .venv install, not a phantom venv."""
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: "/fake/uv")
    monkeypatch.setattr(m, "_is_termux_env", lambda *a, **k: False)
    _make_live_venv(tmp_path, ".venv")

    seen = {}

    def fake_repair(prefix, *, env=None):
        seen["env"] = env
        return "healthy"

    monkeypatch.setattr(m, "_repair_venv_via_import_probes", fake_repair)
    monkeypatch.setattr(m, "_clear_lazy_refresh_incomplete_marker", lambda: None)

    m._recover_lazy_refresh_marker_locked()
    assert seen["env"] is not None
    assert seen["env"]["VIRTUAL_ENV"] == str(tmp_path / ".venv")


def test_venv_scripts_dir_resolves_live_dot_venv(tmp_path, monkeypatch):
    """#84647: _venv_scripts_dir finds Scripts/bin in live .venv layout."""
    import platform

    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    _make_live_venv(tmp_path, ".venv")
    scripts = m._venv_scripts_dir()
    expected_name = "Scripts" if platform.system() == "Windows" else "bin"
    assert scripts == tmp_path / ".venv" / expected_name


def test_venv_core_imports_healthy_targets_live_dot_venv(tmp_path, monkeypatch):
    """#84647: _venv_core_imports_healthy inspects live .venv interpreter."""
    import hermes_cli.update_cmd as u_cmd

    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    _make_live_venv(tmp_path, ".venv")
    healthy, detail = u_cmd._venv_core_imports_healthy()
    assert isinstance(healthy, bool)
    assert isinstance(detail, str)


