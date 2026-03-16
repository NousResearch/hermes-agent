"""Tests for hermes_cli.doctor."""

import os
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.doctor as doctor
import hermes_cli.gateway as gateway_cli
from hermes_cli import doctor as doctor_mod
from hermes_cli.doctor import _has_provider_env_config


class TestProviderEnvDetection:
    def test_detects_openai_api_key(self):
        content = "OPENAI_BASE_URL=http://localhost:1234/v1\nOPENAI_API_KEY=***"
        assert _has_provider_env_config(content)

    def test_detects_custom_endpoint_without_openrouter_key(self):
        content = "OPENAI_BASE_URL=http://localhost:8080/v1\n"
        assert _has_provider_env_config(content)

    def test_returns_false_when_no_provider_settings(self):
        content = "TERMINAL_ENV=local\n"
        assert not _has_provider_env_config(content)


class TestDoctorToolAvailabilityOverrides:
    def test_marks_honcho_available_when_configured(self, monkeypatch):
        monkeypatch.setattr(doctor, "_honcho_is_configured_for_doctor", lambda: True)

        available, unavailable = doctor._apply_doctor_tool_availability_overrides(
            [],
            [{"name": "honcho", "env_vars": [], "tools": ["query_user_context"]}],
        )

        assert available == ["honcho"]
        assert unavailable == []

    def test_leaves_honcho_unavailable_when_not_configured(self, monkeypatch):
        monkeypatch.setattr(doctor, "_honcho_is_configured_for_doctor", lambda: False)

        honcho_entry = {"name": "honcho", "env_vars": [], "tools": ["query_user_context"]}
        available, unavailable = doctor._apply_doctor_tool_availability_overrides(
            [],
            [honcho_entry],
        )

        assert available == []
        assert unavailable == [honcho_entry]


class TestHonchoDoctorConfigDetection:
    def test_reports_configured_when_enabled_with_api_key(self, monkeypatch):
        fake_config = SimpleNamespace(enabled=True, api_key="***")

        monkeypatch.setattr(
            "honcho_integration.client.HonchoClientConfig.from_global_config",
            lambda: fake_config,
        )

        assert doctor._honcho_is_configured_for_doctor()

    def test_reports_not_configured_without_api_key(self, monkeypatch):
        fake_config = SimpleNamespace(enabled=True, api_key="")

        monkeypatch.setattr(
            "honcho_integration.client.HonchoClientConfig.from_global_config",
            lambda: fake_config,
        )

        assert not doctor._honcho_is_configured_for_doctor()


def test_effective_terminal_backend_prefers_env(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "windows-sandbox")
    monkeypatch.setattr(doctor, "load_config", lambda: {"terminal": {"backend": "local"}})
    assert doctor._effective_terminal_backend() == "windows-sandbox"


def test_effective_terminal_backend_falls_back_to_config(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setattr(doctor, "load_config", lambda: {"terminal": {"backend": "windows-sandbox"}})
    assert doctor._effective_terminal_backend() == "windows-sandbox"


def test_check_writable_directory_appends_issue_on_failure(monkeypatch):
    monkeypatch.setattr(doctor, "probe_writable_directory", lambda _path, create=False: (False, "access denied"))

    issues = []
    doctor._check_writable_directory("HERMES_HOME writable", Path("/tmp/fail"), issues, should_fix=True)

    assert issues == [f"Make {Path('/tmp/fail')} writable for the current Hermes user"]


def test_run_doctor_sets_interactive_env_for_tool_checks(monkeypatch, tmp_path):
    """Doctor should present CLI-gated tools as available in CLI context."""
    project_root = tmp_path / "project"
    hermes_home = tmp_path / ".hermes"
    project_root.mkdir()
    hermes_home.mkdir()

    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", hermes_home)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    seen = {}

    def fake_check_tool_availability(*args, **kwargs):
        seen["interactive"] = os.getenv("HERMES_INTERACTIVE")
        raise SystemExit(0)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=fake_check_tool_availability,
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    with pytest.raises(SystemExit):
        doctor_mod.run_doctor(Namespace(fix=False))

    assert seen["interactive"] == "1"


def test_check_gateway_service_linger_warns_when_disabled(monkeypatch, tmp_path, capsys):
    unit_path = tmp_path / "hermes-gateway.service"
    unit_path.write_text("[Unit]\n")

    monkeypatch.setattr(gateway_cli, "is_linux", lambda: True)
    monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda: unit_path)
    monkeypatch.setattr(gateway_cli, "get_systemd_linger_status", lambda: (False, ""))

    issues = []
    doctor._check_gateway_service_linger(issues)

    out = capsys.readouterr().out
    assert "Gateway Service" in out
    assert "Systemd linger disabled" in out
    assert "loginctl enable-linger" in out
    assert issues == [
        "Enable linger for the gateway user service: sudo loginctl enable-linger $USER"
    ]


def test_check_gateway_service_linger_skips_when_service_not_installed(monkeypatch, tmp_path, capsys):
    unit_path = tmp_path / "missing.service"

    monkeypatch.setattr(gateway_cli, "is_linux", lambda: True)
    monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda: unit_path)

    issues = []
    doctor._check_gateway_service_linger(issues)

    out = capsys.readouterr().out
    assert out == ""
    assert issues == []

def test_check_windows_sandbox_backend_reports_missing_wrapper(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(doctor.os, "name", "nt", raising=False)
    monkeypatch.setattr(doctor, "get_hermes_bin_dir", lambda: tmp_path / "hermes-bin")
    monkeypatch.setattr(doctor, "find_wrapper_executable", lambda _bin_dir="": None)
    monkeypatch.setattr(doctor, "find_setup_helper_executable", lambda *_args, **_kwargs: None)

    issues = []
    doctor._check_windows_sandbox_backend(issues)

    out = capsys.readouterr().out
    assert "Windows Sandbox Backend" in out
    assert "windows-sandbox wrapper" in out
    assert any("hermes-windows-sandbox-wrapper.exe" in issue for issue in issues)


def test_check_windows_sandbox_backend_reports_setup_required(monkeypatch, tmp_path, capsys):
    wrapper = tmp_path / "hermes-windows-sandbox-wrapper.exe"
    helper = tmp_path / "codex-windows-sandbox-setup.exe"
    wrapper.write_text("stub", encoding="utf-8")
    helper.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(doctor.os, "name", "nt", raising=False)
    monkeypatch.setattr(doctor, "find_wrapper_executable", lambda _bin_dir="": wrapper)
    monkeypatch.setattr(doctor, "find_setup_helper_executable", lambda *_args, **_kwargs: helper)
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="hermes-windows-sandbox-wrapper 0.1.0", stderr=""),
    )
    monkeypatch.setattr(
        doctor,
        "get_windows_sandbox_status",
        lambda **_kwargs: {
            "error": "Windows sandbox setup is required before execution.",
            "error_type": "setup_required",
            "diagnostics": {"setup_complete": False, "setup_code": "orchestrator_helper_launch_canceled"},
        },
    )

    issues = []
    doctor._check_windows_sandbox_backend(issues)

    out = capsys.readouterr().out
    assert "Setup code: orchestrator_helper_launch_canceled" in out
    assert "windows-sandbox setup" in out
    assert issues == ["Complete windows-sandbox setup before using this backend"]


def test_check_windows_sandbox_backend_confirms_execution_viability(monkeypatch, tmp_path, capsys):
    wrapper = tmp_path / "hermes-windows-sandbox-wrapper.exe"
    helper = tmp_path / "codex-windows-sandbox-setup.exe"
    wrapper.write_text("stub", encoding="utf-8")
    helper.write_text("stub", encoding="utf-8")

    class FakeEnvironment:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def execute(self, command, timeout=None):
            assert command == "Write-Output hermes-windows-sandbox-doctor"
            assert timeout == 20
            return {"output": "hermes-windows-sandbox-doctor", "returncode": 0}

    monkeypatch.setattr(doctor.os, "name", "nt", raising=False)
    monkeypatch.setattr(doctor, "find_wrapper_executable", lambda _bin_dir="": wrapper)
    monkeypatch.setattr(doctor, "find_setup_helper_executable", lambda *_args, **_kwargs: helper)
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="hermes-windows-sandbox-wrapper 0.1.0", stderr=""),
    )
    monkeypatch.setattr(
        doctor,
        "get_windows_sandbox_status",
        lambda **_kwargs: {"error": None, "error_type": None, "diagnostics": {"setup_complete": True}},
    )
    monkeypatch.setattr(doctor, "WindowsSandboxEnvironment", FakeEnvironment)

    issues = []
    doctor._check_windows_sandbox_backend(issues)

    out = capsys.readouterr().out
    assert "windows-sandbox execution viability" in out
    assert issues == []
