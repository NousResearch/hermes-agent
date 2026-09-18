"""Launchd Local Network helper .app (#71206).

The helper is a real .app bundle so ``nehelper`` has an application ID and
``NSLocalNetworkUsageDescription``. Layout and argv wrapping are host-independent;
install is boot-gated in production and is not required for these contracts.
"""

from __future__ import annotations

import plistlib
from pathlib import Path

import hermes_cli.macos_gateway_app as helper
from hermes_cli import doctor_platform


def _darwin(monkeypatch):
    monkeypatch.setattr(helper.platform, "system", lambda: "Darwin")


def _linux(monkeypatch):
    monkeypatch.setattr(helper.platform, "system", lambda: "Linux")


def _fake_venv(tmp_path: Path) -> tuple[Path, Path]:
    venv = tmp_path / "venv"
    venv_bin = venv / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    python.write_bytes(b"#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    (venv / "pyvenv.cfg").write_text("home = /tmp/fake\n")
    (venv / "lib").mkdir()
    return venv, python


class TestWrapLaunchdPython:
    def test_explicit_helper_replaces_python_entries(self):
        helper_exe = "/tmp/HermesGateway.app/Contents/MacOS/HermesGateway"
        command = ["/venv/bin/python", "-m", "hermes_cli.stderr_timestamp", "--", "/venv/bin/python", "-m", "hermes_cli.main"]
        wrapped = helper.wrap_launchd_python(command, "/venv/bin/python", helper_exe=helper_exe)
        assert wrapped[0] == helper_exe
        assert wrapped[-3] == helper_exe
        assert wrapped[1] == "-m"

    def test_non_macos_leaves_argv_unchanged(self, monkeypatch):
        _linux(monkeypatch)
        command = ["/venv/bin/python", "-m", "hermes_cli.main"]
        assert helper.wrap_launchd_python(command, "/venv/bin/python") == command


class TestInfoPlist:
    def test_declares_local_network_usage_and_bundle_id(self):
        payload = helper._info_plist_payload()
        assert payload["CFBundleIdentifier"] == helper.BUNDLE_ID
        assert payload["NSLocalNetworkUsageDescription"] == helper.LOCAL_NETWORK_USAGE
        assert payload["LSUIElement"] is True
        assert payload["CFBundleExecutable"] == helper.EXECUTABLE_NAME


class TestInstallLayout:
    def test_install_writes_bundle_identity(self, tmp_path):
        venv, python = _fake_venv(tmp_path)
        app = tmp_path / "home" / "macos" / "HermesGateway.app"
        helper._install_app(app, python, venv)
        exe = helper.gateway_app_executable(app)
        assert exe.is_file()
        info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
        assert info["CFBundleIdentifier"] == helper.BUNDLE_ID
        assert "NSLocalNetworkUsageDescription" in info
        assert (app / "Contents" / "pyvenv.cfg").is_file()
        lib = app / "Contents" / "lib"
        assert lib.is_symlink()
        assert lib.resolve() == (venv / "lib").resolve()
        marker = (app / "Contents" / helper._MARKER_NAME).read_text(encoding="utf-8").strip()
        assert marker == str(python)

    def test_state_skip_off_macos(self, monkeypatch):
        _linux(monkeypatch)
        status, detail = helper.gateway_app_state()
        assert status == "skip"
        assert "not macOS" in detail

    def test_state_missing_when_bundle_absent(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        venv, python = _fake_venv(tmp_path)
        monkeypatch.setattr(helper, "_venv_python_source", lambda: (venv, python))
        monkeypatch.setattr(helper, "get_hermes_home", lambda: tmp_path / "home")
        status, detail = helper.gateway_app_state(tmp_path / "home")
        assert status == "missing"
        assert str(tmp_path / "home" / "macos" / "HermesGateway.app") in detail


class TestGenerateLaunchdPlist:
    def test_program_arguments_use_helper_when_wrap_returns_it(self, tmp_path, monkeypatch):
        import plistlib as pl

        import hermes_cli.gateway as gateway_cli

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: home)
        monkeypatch.setattr(gateway_cli, "get_python_path", lambda: "/venv/bin/python")
        helper_exe = str(tmp_path / "HermesGateway")

        def fake_wrap(command, python_path):
            assert python_path == "/venv/bin/python"
            return helper.wrap_launchd_python(command, python_path, helper_exe=helper_exe)

        monkeypatch.setattr("hermes_cli.macos_gateway_app.launchd_python_command", fake_wrap)
        parsed = pl.loads(gateway_cli.generate_launchd_plist().encode("utf-8"))
        args = parsed["ProgramArguments"]
        assert args[0] == helper_exe
        assert "/venv/bin/python" not in args
        assert args.count(helper_exe) == 2
        assert "gateway" in args
        assert "run" in args


class TestDoctorCheck:
    def test_missing_warns_without_fix(self, monkeypatch, capsys):
        monkeypatch.setattr(
            helper, "gateway_app_state", lambda *a, **k: ("missing", "/x/macos/HermesGateway.app")
        )
        doctor_platform.check_macos_gateway_app(should_fix=False)
        out = capsys.readouterr().out
        assert "macOS gateway Local Network helper missing" in out
        assert "Local Network" in out

    def test_fix_installs_helper(self, monkeypatch, capsys):
        monkeypatch.setattr(
            helper, "gateway_app_state", lambda *a, **k: ("missing", "/x/macos/HermesGateway.app")
        )
        monkeypatch.setattr(
            helper, "ensure_gateway_app", lambda *a, **k: Path("/x/macos/HermesGateway.app/Contents/MacOS/HermesGateway")
        )
        doctor_platform.check_macos_gateway_app(should_fix=True)
        out = capsys.readouterr().out
        assert "macOS gateway Local Network helper installed" in out

    def test_active_reports_ok(self, monkeypatch, capsys):
        monkeypatch.setattr(
            helper,
            "gateway_app_state",
            lambda *a, **k: ("active", "/x/macos/HermesGateway.app/Contents/MacOS/HermesGateway"),
        )
        doctor_platform.check_macos_gateway_app(should_fix=False)
        out = capsys.readouterr().out
        assert "macOS gateway Local Network helper active" in out

    def test_skip_is_silent(self, monkeypatch, capsys):
        monkeypatch.setattr(helper, "gateway_app_state", lambda *a, **k: ("skip", "not macOS"))
        doctor_platform.check_macos_gateway_app(should_fix=False)
        assert capsys.readouterr().out == ""
