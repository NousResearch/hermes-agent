"""Tests for hermes_cli.doctor."""

import io
import json
import os
import sys
import types
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

import hermes_cli.doctor as doctor
import hermes_cli.gateway as gateway_cli
from hermes_cli import doctor as doctor_mod
from hermes_cli.doctor import _has_provider_env_config


class TestDoctorPlatformHints:
    def test_termux_package_hint(self, monkeypatch):
        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        assert doctor._is_termux() is True
        assert doctor._python_install_cmd() == "python -m pip install"
        assert doctor._system_package_install_cmd("ripgrep") == "pkg install ripgrep"

    def test_non_termux_package_hint_defaults_to_apt(self, monkeypatch):
        monkeypatch.delenv("TERMUX_VERSION", raising=False)
        monkeypatch.setenv("PREFIX", "/usr")
        monkeypatch.setattr(sys, "platform", "linux")
        assert doctor._is_termux() is False
        assert doctor._python_install_cmd() == "uv pip install"
        assert doctor._system_package_install_cmd("ripgrep") == "sudo apt install ripgrep"


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
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            lambda: fake_config,
        )

        assert doctor._honcho_is_configured_for_doctor()

    def test_reports_not_configured_without_api_key(self, monkeypatch):
        fake_config = SimpleNamespace(enabled=True, api_key="")

        monkeypatch.setattr(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            lambda: fake_config,
        )

        assert not doctor._honcho_is_configured_for_doctor()


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


# ── Memory provider section (doctor should only check the *active* provider) ──


class TestDoctorMemoryProviderSection:
    """The ◆ Memory Provider section should respect memory.provider config."""

    def _make_hermes_home(self, tmp_path, provider=""):
        """Create a minimal HERMES_HOME with config.yaml."""
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        import yaml
        config = {"memory": {"provider": provider}} if provider else {"memory": {}}
        (home / "config.yaml").write_text(yaml.dump(config))
        return home

    def _run_doctor_and_capture(self, monkeypatch, tmp_path, provider=""):
        """Run doctor and capture stdout."""
        home = self._make_hermes_home(tmp_path, provider)
        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))
        (tmp_path / "project").mkdir(exist_ok=True)

        # Stub tool availability (returns empty) so doctor runs past it
        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        # Stub auth checks to avoid real API calls
        try:
            from hermes_cli import auth as _auth_mod
            monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
            monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        except Exception:
            pass

        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))
        return buf.getvalue()

    def test_no_provider_shows_builtin_ok(self, monkeypatch, tmp_path):
        out = self._run_doctor_and_capture(monkeypatch, tmp_path, provider="")
        assert "Memory Provider" in out
        assert "Built-in memory active" in out
        # Should NOT mention Honcho or Mem0 errors
        assert "Honcho API key" not in out
        assert "Mem0" not in out

    def test_honcho_provider_not_installed_shows_fail(self, monkeypatch, tmp_path):
        # Make honcho import fail
        monkeypatch.setitem(
            sys.modules, "plugins.memory.honcho.client", None
        )
        out = self._run_doctor_and_capture(monkeypatch, tmp_path, provider="honcho")
        assert "Memory Provider" in out
        # Should show failure since honcho is set but not importable
        assert "Built-in memory active" not in out

    def test_mem0_provider_not_installed_shows_fail(self, monkeypatch, tmp_path):
        # Make mem0 import fail
        monkeypatch.setitem(sys.modules, "plugins.memory.mem0", None)
        out = self._run_doctor_and_capture(monkeypatch, tmp_path, provider="mem0")
        assert "Memory Provider" in out
        assert "Built-in memory active" not in out


def test_run_doctor_termux_treats_docker_and_browser_warnings_as_expected(monkeypatch, tmp_path):
    helper = TestDoctorMemoryProviderSection()
    monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")

    real_which = doctor_mod.shutil.which

    def fake_which(cmd):
        if cmd in {"docker", "node", "npm"}:
            return None
        return real_which(cmd)

    monkeypatch.setattr(doctor_mod.shutil, "which", fake_which)

    out = helper._run_doctor_and_capture(monkeypatch, tmp_path, provider="")

    assert "Docker backend is not available inside Termux" in out
    assert "Node.js not found (browser tools are optional in the tested Termux path)" in out
    assert "Install Node.js on Termux with: pkg install nodejs" in out
    assert "Termux browser setup:" in out
    assert "1) pkg install nodejs" in out
    assert "2) npm install -g agent-browser" in out
    assert "3) agent-browser install" in out
    assert "docker not found (optional)" not in out


def test_run_doctor_termux_does_not_mark_browser_available_without_agent_browser(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)

    monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda cmd: "/data/data/com.termux/files/usr/bin/node" if cmd in {"node", "npm"} else None)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: (["terminal"], [{"name": "browser", "env_vars": [], "tools": ["browser_navigate"]}]),
        TOOLSET_REQUIREMENTS={
            "terminal": {"name": "terminal"},
            "browser": {"name": "browser"},
        },
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
    except Exception:
        pass

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    out = buf.getvalue()

    assert "✓ browser" not in out
    assert "browser" in out
    assert "system dependency not met" in out
    assert "agent-browser is not installed (expected in the tested Termux path)" in out
    assert "npm install -g agent-browser && agent-browser install" in out


# =============================================================================
# npm audit checks in hermes doctor
# =============================================================================

def _make_audit_json(critical=0, high=0, moderate=0, low=0):
    """Build a minimal npm audit --json payload."""
    return json.dumps({
        "metadata": {
            "vulnerabilities": {
                "critical": critical,
                "high": high,
                "moderate": moderate,
                "low": low,
                "total": critical + high + moderate + low,
            }
        }
    })


def _make_completed_process(stdout="", returncode=0):
    proc = MagicMock()
    proc.stdout = stdout
    proc.returncode = returncode
    return proc


class TestDoctorNpmAudit:
    """Tests for the npm audit section of run_doctor."""

    def _run_audit_section(self, monkeypatch, tmp_path, audit_json, fix=False, audit_fix_rc=0):
        """
        Run only the npm audit section of run_doctor by patching everything else
        away and providing a fake project root with a node_modules directory.
        """
        # Create a fake project root with node_modules so the loop enters
        project_root = tmp_path / "project"
        node_modules = project_root / "node_modules"
        node_modules.mkdir(parents=True)

        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project_root)
        monkeypatch.setattr(doctor_mod, "HERMES_HOME", tmp_path / ".hermes")
        (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(doctor_mod, "_DHH", str(tmp_path / ".hermes"))

        # npm is available; pip-audit and node are absent so only npm section runs
        real_which = doctor_mod.shutil.which
        monkeypatch.setattr(
            doctor_mod.shutil, "which",
            lambda cmd: "/usr/bin/npm" if cmd == "npm" else (None if cmd in {"pip-audit", "node"} else real_which(cmd)),
        )

        audit_proc = _make_completed_process(stdout=audit_json, returncode=0 if json.loads(audit_json).get("metadata", {}).get("vulnerabilities", {}).get("total", 0) == 0 else 1)
        fix_proc = _make_completed_process(stdout="", returncode=audit_fix_rc)

        call_results = [audit_proc, fix_proc] if fix else [audit_proc]

        subprocess_mock = MagicMock(side_effect=call_results)
        monkeypatch.setattr(doctor_mod.subprocess, "run", subprocess_mock)

        # Stub tool availability so doctor can complete the run
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

        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=fix))
        return buf.getvalue(), subprocess_mock

    def test_no_vulnerabilities(self, monkeypatch, tmp_path):
        """Zero vulns → clean checkmark with 'no known vulnerabilities'."""
        out, _ = self._run_audit_section(monkeypatch, tmp_path, _make_audit_json())
        assert "no known vulnerabilities" in out

    def test_critical_high_warns_and_adds_issue(self, monkeypatch, tmp_path):
        """Critical and high vulns → warning line and issue in summary."""
        out, _ = self._run_audit_section(
            monkeypatch, tmp_path, _make_audit_json(critical=1, high=2, moderate=1)
        )
        assert "⚠" in out or "warning" in out.lower() or "npm audit fix" in out
        assert "npm audit fix" in out
        assert "1 critical" in out or "critical/high" in out.lower() or "issue" in out.lower()

    def test_moderate_only_warns(self, monkeypatch, tmp_path):
        """Moderate-only vulns are now a warning (not a clean checkmark)."""
        out, _ = self._run_audit_section(
            monkeypatch, tmp_path, _make_audit_json(moderate=2)
        )
        # Must NOT show "✓" for this label without a follow-up fix
        # Must include a warning and the npm audit fix hint
        assert "npm audit fix" in out
        assert "moderate" in out

    def test_npm_not_on_path(self, monkeypatch, tmp_path):
        """When npm is not available the audit section is skipped entirely."""
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
        monkeypatch.setattr(doctor_mod, "HERMES_HOME", tmp_path / ".hermes")
        (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(doctor_mod, "_DHH", str(tmp_path / ".hermes"))
        monkeypatch.setattr(doctor_mod.shutil, "which", lambda cmd: None)

        subprocess_mock = MagicMock()
        monkeypatch.setattr(doctor_mod.subprocess, "run", subprocess_mock)

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

        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))

        # npm audit should never have been called
        audit_calls = [c for c in subprocess_mock.call_args_list if "npm" in str(c) and "audit" in str(c)]
        assert audit_calls == []

    def test_no_node_modules(self, monkeypatch, tmp_path):
        """Directories without node_modules are skipped (no npm audit call)."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        # Deliberately no node_modules directory

        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project_root)
        monkeypatch.setattr(doctor_mod, "HERMES_HOME", tmp_path / ".hermes")
        (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(doctor_mod, "_DHH", str(tmp_path / ".hermes"))

        real_which = doctor_mod.shutil.which
        monkeypatch.setattr(doctor_mod.shutil, "which", lambda cmd: "/usr/bin/npm" if cmd == "npm" else real_which(cmd))

        subprocess_mock = MagicMock()
        monkeypatch.setattr(doctor_mod.subprocess, "run", subprocess_mock)

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

        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))

        audit_calls = [c for c in subprocess_mock.call_args_list if "npm" in str(c) and "audit" in str(c)]
        assert audit_calls == []

    def test_audit_exception_shows_warning(self, monkeypatch, tmp_path):
        """If subprocess.run raises, a warning is shown instead of silently passing."""
        project_root = tmp_path / "project"
        (project_root / "node_modules").mkdir(parents=True)

        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project_root)
        monkeypatch.setattr(doctor_mod, "HERMES_HOME", tmp_path / ".hermes")
        (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(doctor_mod, "_DHH", str(tmp_path / ".hermes"))

        real_which = doctor_mod.shutil.which
        monkeypatch.setattr(doctor_mod.shutil, "which", lambda cmd: "/usr/bin/npm" if cmd == "npm" else real_which(cmd))

        monkeypatch.setattr(doctor_mod.subprocess, "run", MagicMock(side_effect=OSError("npm not found")))

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

        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))
        out = buf.getvalue()

        assert "npm audit failed" in out

    def test_fix_flag_runs_npm_audit_fix_on_success(self, monkeypatch, tmp_path):
        """--fix with high vulns: npm audit fix is invoked and the issue is marked fixed."""
        out, subprocess_mock = self._run_audit_section(
            monkeypatch, tmp_path,
            _make_audit_json(high=1, moderate=1),
            fix=True,
            audit_fix_rc=0,
        )
        fix_calls = [c for c in subprocess_mock.call_args_list if "audit" in str(c) and "fix" in str(c)]
        assert fix_calls, "npm audit fix was not called when --fix is set and high vulns exist"
        assert "fixed" in out.lower() or "no known vulnerabilities" in out

    def test_fix_flag_reports_failure_when_audit_fix_fails(self, monkeypatch, tmp_path):
        """--fix with npm audit fix failure → surfaced as a manual issue."""
        out, subprocess_mock = self._run_audit_section(
            monkeypatch, tmp_path,
            _make_audit_json(high=1),
            fix=True,
            audit_fix_rc=1,
        )
        fix_calls = [c for c in subprocess_mock.call_args_list if "audit" in str(c) and "fix" in str(c)]
        assert fix_calls, "npm audit fix was not called"
        assert "failed" in out.lower() or "manually" in out.lower()


# =============================================================================
# Python (pip-audit) checks in hermes doctor
# =============================================================================

def _make_pip_audit_json(vulnerable_packages=None):
    """
    Build a minimal pip-audit --json payload.

    vulnerable_packages: list of (name, version, vuln_ids) tuples.
    """
    if vulnerable_packages is None:
        vulnerable_packages = []

    deps = []
    for name, version, vuln_ids in vulnerable_packages:
        deps.append({
            "name": name,
            "version": version,
            "vulns": [
                {"id": vid, "fix_versions": [], "description": f"Test vuln {vid}", "aliases": []}
                for vid in vuln_ids
            ],
        })
    # Also include a clean package to confirm mixed results work
    deps.append({"name": "safe-package", "version": "1.0.0", "vulns": []})
    return json.dumps({"dependencies": deps})


class TestDoctorPythonAudit:
    """Tests for the pip-audit section of run_doctor."""

    def _run_python_audit_section(
        self,
        monkeypatch,
        tmp_path,
        pip_audit_json,
        fix=False,
        audit_fix_rc=0,
        uv_lock_rc=0,
        pip_audit_available=True,
    ):
        """
        Run run_doctor with pip-audit mocked out.
        Returns (stdout_str, subprocess_mock).
        """
        project_root = tmp_path / "project"
        project_root.mkdir(parents=True)

        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project_root)
        monkeypatch.setattr(doctor_mod, "HERMES_HOME", tmp_path / ".hermes")
        (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(doctor_mod, "_DHH", str(tmp_path / ".hermes"))

        real_which = doctor_mod.shutil.which

        def fake_which(cmd):
            if cmd == "pip-audit":
                return "/usr/bin/pip-audit" if pip_audit_available else None
            if cmd in {"npm", "node"}:
                return None  # skip npm audit section
            if cmd == "uv":
                return "/usr/bin/uv"
            return real_which(cmd)

        monkeypatch.setattr(doctor_mod.shutil, "which", fake_which)

        parsed = json.loads(pip_audit_json)
        has_vulns = any(d.get("vulns") for d in parsed.get("dependencies", []))
        audit_proc = _make_completed_process(stdout=pip_audit_json, returncode=1 if has_vulns else 0)
        fix_proc = _make_completed_process(stdout="", returncode=audit_fix_rc)
        uv_lock_proc = _make_completed_process(stdout="", returncode=uv_lock_rc)

        # Build call_results: audit always first; fix + uv_lock only when --fix and vulns found
        if fix and has_vulns:
            call_results = [audit_proc, fix_proc, uv_lock_proc]
        else:
            call_results = [audit_proc]

        subprocess_mock = MagicMock(side_effect=call_results)
        monkeypatch.setattr(doctor_mod.subprocess, "run", subprocess_mock)

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

        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=fix))
        return buf.getvalue(), subprocess_mock

    def test_no_python_vulnerabilities(self, monkeypatch, tmp_path):
        """Zero vulns → clean checkmark."""
        out, _ = self._run_python_audit_section(
            monkeypatch, tmp_path, _make_pip_audit_json()
        )
        assert "no known vulnerabilities" in out

    def test_python_vulnerabilities_warn_and_add_issue(self, monkeypatch, tmp_path):
        """Found vulns → warning + fix hint in output."""
        out, _ = self._run_python_audit_section(
            monkeypatch, tmp_path,
            _make_pip_audit_json([("requests", "2.28.0", ["GHSA-1234", "GHSA-5678"])]),
        )
        assert "pip-audit --fix" in out
        assert "requests" in out

    def test_pip_audit_not_installed_shows_info(self, monkeypatch, tmp_path):
        """When pip-audit is missing, print an info hint (not a hard error)."""
        out, subprocess_mock = self._run_python_audit_section(
            monkeypatch, tmp_path,
            _make_pip_audit_json(),
            pip_audit_available=False,
        )
        audit_calls = [c for c in subprocess_mock.call_args_list if "pip-audit" in str(c)]
        assert audit_calls == [], "pip-audit should not be called when not on PATH"
        assert "pip-audit" in out
        assert "pip install pip-audit" in out

    def test_pip_audit_exception_shows_warning(self, monkeypatch, tmp_path):
        """If pip-audit raises, a warning is shown instead of silently passing."""
        project_root = tmp_path / "project"
        project_root.mkdir(parents=True)

        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project_root)
        monkeypatch.setattr(doctor_mod, "HERMES_HOME", tmp_path / ".hermes")
        (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(doctor_mod, "_DHH", str(tmp_path / ".hermes"))

        real_which = doctor_mod.shutil.which
        monkeypatch.setattr(
            doctor_mod.shutil, "which",
            lambda cmd: "/usr/bin/pip-audit" if cmd == "pip-audit" else (None if cmd in {"npm", "node"} else real_which(cmd)),
        )
        monkeypatch.setattr(doctor_mod.subprocess, "run", MagicMock(side_effect=OSError("pip-audit crashed")))

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

        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))
        assert "pip-audit failed" in buf.getvalue()

    def test_fix_flag_runs_pip_audit_fix_and_uv_lock(self, monkeypatch, tmp_path):
        """--fix with vulns: pip-audit --fix is called, then uv lock is run."""
        out, subprocess_mock = self._run_python_audit_section(
            monkeypatch, tmp_path,
            _make_pip_audit_json([("cryptography", "3.0", ["CVE-2023-0001"])]),
            fix=True,
            audit_fix_rc=0,
            uv_lock_rc=0,
        )
        calls_str = str(subprocess_mock.call_args_list)
        assert "pip-audit" in calls_str and "--fix" in calls_str
        assert "uv" in calls_str and "lock" in calls_str
        assert "fixed" in out.lower() or "synced" in out.lower()

    def test_fix_flag_reports_failure_when_pip_audit_fix_fails(self, monkeypatch, tmp_path):
        """--fix with pip-audit --fix failure → surfaced as a manual issue."""
        out, subprocess_mock = self._run_python_audit_section(
            monkeypatch, tmp_path,
            _make_pip_audit_json([("urllib3", "1.26.0", ["CVE-2023-0002"])]),
            fix=True,
            audit_fix_rc=1,
        )
        assert "failed" in out.lower() or "manually" in out.lower()

    def test_fix_flag_warns_when_uv_lock_fails(self, monkeypatch, tmp_path):
        """--fix: pip-audit --fix succeeds but uv lock fails → warning about manual uv lock."""
        out, _ = self._run_python_audit_section(
            monkeypatch, tmp_path,
            _make_pip_audit_json([("pyyaml", "5.3", ["CVE-2020-1234"])]),
            fix=True,
            audit_fix_rc=0,
            uv_lock_rc=1,
        )
        assert "uv lock" in out
