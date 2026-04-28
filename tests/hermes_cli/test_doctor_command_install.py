"""Tests for the Command Installation check in hermes doctor."""

import os
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

import hermes_cli.doctor as doctor_mod


def _setup_doctor_env(monkeypatch, tmp_path, venv_name="venv"):
    """Create a minimal HERMES_HOME + PROJECT_ROOT for doctor tests."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")

    project = tmp_path / "project"
    project.mkdir(exist_ok=True)

    # Create a fake venv entry point
    venv_bin_dir = project / venv_name / "bin"
    venv_bin_dir.mkdir(parents=True, exist_ok=True)
    hermes_bin = venv_bin_dir / "hermes"
    hermes_bin.write_text("#!/usr/bin/env python\n# entry point\n")
    hermes_bin.chmod(0o755)

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))

    # Stub model_tools so doctor doesn't fail on import
    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    # Stub auth checks
    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
    except Exception:
        pass

    # Stub httpx.get to avoid network calls
    try:
        import httpx
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: types.SimpleNamespace(status_code=200))
    except Exception:
        pass

    return home, project, hermes_bin


def _run_doctor(fix=False):
    """Run doctor and capture stdout."""
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=fix))
    return buf.getvalue()


class TestDoctorCommandInstallation:
    """Tests for the ◆ Command Installation section."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlink check is Unix-only")
    def test_correct_symlink_shows_ok(self, monkeypatch, tmp_path):
        home, project, hermes_bin = _setup_doctor_env(monkeypatch, tmp_path)

        # Create the command link dir with correct symlink
        cmd_link_dir = tmp_path / ".local" / "bin"
        cmd_link_dir.mkdir(parents=True)
        cmd_link = cmd_link_dir / "hermes"
        cmd_link.symlink_to(hermes_bin)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out = _run_doctor(fix=False)
        assert "Command Installation" in out
        assert "Venv entry point exists" in out
        assert "correct target" in out

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlink check is Unix-only")
    def test_missing_symlink_shows_fail(self, monkeypatch, tmp_path):
        home, project, hermes_bin = _setup_doctor_env(monkeypatch, tmp_path)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # Don't create the symlink — it should be missing

        out = _run_doctor(fix=False)
        assert "Command Installation" in out
        assert "Venv entry point exists" in out
        assert "not found" in out
        assert "hermes doctor --fix" in out

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlink check is Unix-only")
    def test_fix_creates_missing_symlink(self, monkeypatch, tmp_path):
        home, project, hermes_bin = _setup_doctor_env(monkeypatch, tmp_path)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out = _run_doctor(fix=True)
        assert "Command Installation" in out
        assert "Created symlink" in out

        # Verify the symlink was actually created
        cmd_link = tmp_path / ".local" / "bin" / "hermes"
        assert cmd_link.is_symlink()
        assert cmd_link.resolve() == hermes_bin.resolve()

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlink check is Unix-only")
    def test_wrong_target_symlink_shows_warn(self, monkeypatch, tmp_path):
        home, project, hermes_bin = _setup_doctor_env(monkeypatch, tmp_path)

        # Create a symlink pointing to the wrong target
        cmd_link_dir = tmp_path / ".local" / "bin"
        cmd_link_dir.mkdir(parents=True)
        cmd_link = cmd_link_dir / "hermes"
        wrong_target = tmp_path / "wrong_hermes"
        wrong_target.write_text("#!/usr/bin/env python\n")
        cmd_link.symlink_to(wrong_target)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out = _run_doctor(fix=False)
        assert "Command Installation" in out
        assert "wrong target" in out

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlink check is Unix-only")
    def test_fix_repairs_wrong_symlink(self, monkeypatch, tmp_path):
        home, project, hermes_bin = _setup_doctor_env(monkeypatch, tmp_path)

        # Create a symlink pointing to wrong target
        cmd_link_dir = tmp_path / ".local" / "bin"
        cmd_link_dir.mkdir(parents=True)
        cmd_link = cmd_link_dir / "hermes"
        wrong_target = tmp_path / "wrong_hermes"
        wrong_target.write_text("#!/usr/bin/env python\n")
        cmd_link.symlink_to(wrong_target)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out = _run_doctor(fix=True)
        assert "Fixed symlink" in out

        # Verify the symlink now points to the correct target
        assert cmd_link.is_symlink()
        assert cmd_link.resolve() == hermes_bin.resolve()

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlink check is Unix-only")
    def test_missing_venv_entry_point_shows_warn(self, monkeypatch, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")

        project = tmp_path / "project"
        project.mkdir(exist_ok=True)
        # Do NOT create any venv entry point

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)
        try:
            from hermes_cli import auth as _auth_mod
            monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
            monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        except Exception:
            pass
        try:
            import httpx
            monkeypatch.setattr(httpx, "get", lambda *a, **kw: types.SimpleNamespace(status_code=200))
        except Exception:
            pass

        out = _run_doctor(fix=False)
        assert "Command Installation" in out
        assert "Venv entry point not found" in out

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlink check is Unix-only")
    def test_dot_venv_dir_is_found(self, monkeypatch, tmp_path):
        """The check finds entry points in .venv/ as well as venv/."""
        home, project, _ = _setup_doctor_env(monkeypatch, tmp_path, venv_name=".venv")

        # Create the command link with correct symlink
        hermes_bin = project / ".venv" / "bin" / "hermes"
        cmd_link_dir = tmp_path / ".local" / "bin"
        cmd_link_dir.mkdir(parents=True)
        cmd_link = cmd_link_dir / "hermes"
        cmd_link.symlink_to(hermes_bin)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out = _run_doctor(fix=False)
        assert "Venv entry point exists" in out
        assert ".venv/bin/hermes" in out

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlink check is Unix-only")
    def test_non_symlink_regular_file_shows_ok(self, monkeypatch, tmp_path):
        """If ~/.local/bin/hermes is a regular file (not symlink), accept it."""
        home, project, hermes_bin = _setup_doctor_env(monkeypatch, tmp_path)

        cmd_link_dir = tmp_path / ".local" / "bin"
        cmd_link_dir.mkdir(parents=True)
        cmd_link = cmd_link_dir / "hermes"
        cmd_link.write_text("#!/bin/sh\nexec python -m hermes_cli.main \"$@\"\n")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out = _run_doctor(fix=False)
        assert "non-symlink" in out

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlink check is Unix-only")
    def test_termux_uses_prefix_bin(self, monkeypatch, tmp_path):
        """On Termux, the command link dir is $PREFIX/bin."""
        prefix_dir = tmp_path / "termux_prefix"
        prefix_bin = prefix_dir / "bin"
        prefix_bin.mkdir(parents=True)

        home, project, hermes_bin = _setup_doctor_env(monkeypatch, tmp_path)

        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", str(prefix_dir))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out = _run_doctor(fix=False)
        assert "Command Installation" in out
        assert "$PREFIX/bin" in out

    def test_windows_skips_check(self, monkeypatch, tmp_path):
        """On Windows, the Command Installation section is skipped."""
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")

        project = tmp_path / "project"
        project.mkdir(exist_ok=True)

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))
        monkeypatch.setattr(sys, "platform", "win32")

        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)
        try:
            from hermes_cli import auth as _auth_mod
            monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
            monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        except Exception:
            pass
        try:
            import httpx
            monkeypatch.setattr(httpx, "get", lambda *a, **kw: types.SimpleNamespace(status_code=200))
        except Exception:
            pass

        out = _run_doctor(fix=False)
        assert "Command Installation" not in out
