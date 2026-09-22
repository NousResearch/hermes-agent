"""Tests for interrupted-install self-heal (the ``.update-incomplete`` marker).

Covers the breadcrumb lifecycle and the launch-time recovery guard added so a
``hermes update`` killed mid-install (Ctrl-C, terminal close, WSL OOM) gets
finished automatically on the next launch instead of leaving a half-built venv.
"""

from __future__ import annotations

import sys

import pytest

import hermes_cli.main as m
from hermes_cli import _install_repair, main_install_repair


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


@pytest.mark.windows_only
def test_recovery_self_lock_does_not_clear_core_marker_via_import_probes(
    tmp_path, monkeypatch
):
    # Healthy package-only probes are first aid, not proof that the full core
    # install finished (#58004). Exercise the real native self-lock predicate.
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    m._write_update_incomplete_marker()
    marker = m._update_marker_path()
    original_marker = marker.read_bytes()

    scripts_dir = tmp_path / "venv" / "Scripts"
    scripts_dir.mkdir(parents=True)
    shim = scripts_dir / "hermes.exe"
    shim.write_text("")
    monkeypatch.setattr(main_install_repair, "_venv_scripts_dir", lambda: scripts_dir)
    monkeypatch.setattr(main_install_repair, "_hermes_exe_shims", lambda d: [shim])
    install_prefix = ["uv", "pip"]
    install_env = {"VIRTUAL_ENV": str(tmp_path / "venv")}
    monkeypatch.setattr(
        main_install_repair, "_default_venv_install_target",
        lambda: (install_prefix, install_env),
    )

    class FakeProc:
        def __init__(self, pid, exe_path, parents=()):
            self.pid = pid
            self._exe = exe_path
            self._parents = parents

        def exe(self):
            return self._exe

        def parents(self):
            return list(self._parents)

    launcher = FakeProc(101, str(shim))
    interpreter = FakeProc(102, sys.executable, (launcher,))
    monkeypatch.setattr("psutil.Process", lambda: interpreter)

    # Only the package and full-install executors are replaced: no real
    # installer, uv bootstrap, or host process inventory may run in this test.
    def unexpected_subprocess(*args, **kwargs):
        pytest.fail("recovery escaped the inert install seams")

    monkeypatch.setattr(main_install_repair.subprocess, "run", unexpected_subprocess)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: None)
    events = []
    install_succeeds = False

    def first_aid(prefix, *, env):
        assert prefix == install_prefix
        assert env == install_env
        events.append(("first-aid", marker.read_bytes()))
        return "healthy"

    def full_install(root):
        assert root == tmp_path
        events.append(("full-install", marker.read_bytes()))
        if not install_succeeds:
            raise RuntimeError("fixture full install failed")

    monkeypatch.setattr(main_install_repair, "_repair_venv_via_import_probes", first_aid)
    monkeypatch.setattr(_install_repair, "run_core_install", full_install)

    assert main_install_repair._windows_running_hermes_launcher_locked() is True
    m._recover_from_interrupted_install()
    assert events == [("first-aid", original_marker), ("full-install", original_marker)]
    assert marker.read_bytes() == original_marker, "failed full install must retain core marker"

    install_succeeds = True
    events.clear()
    m._recover_from_interrupted_install()
    assert events == [("first-aid", original_marker), ("full-install", original_marker)]
    assert not marker.exists(), "cleared only after successful full reinstall"
