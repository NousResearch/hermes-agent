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


def test_clear_when_absent_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    # Must not raise when the marker was never written.
    m._clear_update_incomplete_marker()
    assert not m._update_marker_path().exists()


def test_recovery_noop_without_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    called = {"install": False}
    monkeypatch.setattr(
        m,
        "_install_python_dependencies_with_optional_fallback",
        lambda *a, **k: called.__setitem__("install", True),
    )
    m._recover_from_interrupted_install()
    assert called["install"] is False, "recovery must not install when no marker"


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


def test_lazy_marker_cleared_only_after_confirmed_import_repair(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    m._write_lazy_refresh_incomplete_marker()

    monkeypatch.setattr(
        m,
        "_default_venv_install_target",
        lambda: (["uv", "pip"], {"VIRTUAL_ENV": str(tmp_path / "venv")}),
    )
    monkeypatch.setattr(
        m, "_repair_venv_via_import_probes", lambda *a, **k: "repaired"
    )

    seen = {"install": False}
    _stub_install_env(monkeypatch, m, seen)

    m._recover_from_interrupted_install()

    assert seen["install"] is False, "lazy marker does not require full .[all] reinstall"
    assert not m._lazy_refresh_marker_path().exists()


def sys_executable_path():
    import sys

    return sys.executable


def test_recovery_output_goes_to_stderr(tmp_path, monkeypatch, capfd):
    # ACP speaks JSON-RPC on stdout — recovery output (including the streamed
    # install, which inherits fd 1) must land on stderr only.
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    m._write_update_incomplete_marker()

    seen = {"install": False}
    _stub_install_env(monkeypatch, m, seen)

    m._recover_from_interrupted_install()

    out, err = capfd.readouterr()
    assert "interrupted mid-install" not in out
    assert "interrupted mid-install" in err
    assert "recovered" in err
