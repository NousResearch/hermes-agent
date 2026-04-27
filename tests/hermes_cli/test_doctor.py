"""Tests for hermes_cli.doctor."""

import os
import sys
import types
import io
import contextlib
from argparse import Namespace
from types import SimpleNamespace

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

    def test_detects_kimi_cn_api_key(self):
        content = "KIMI_CN_API_KEY=sk-test\n"
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


def test_run_doctor_accepts_named_provider_from_providers_section(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)

    import yaml

    (home / "config.yaml").write_text(
        yaml.dump(
            {
                "model": {
                    "provider": "volcengine-plan",
                    "default": "doubao-seed-2.0-code",
                },
                "providers": {
                    "volcengine-plan": {
                        "name": "volcengine-plan",
                        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
                        "default_model": "doubao-seed-2.0-code",
                        "models": {"doubao-seed-2.0-code": {}},
                    }
                },
            }
        )
    )

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    (tmp_path / "project").mkdir(exist_ok=True)

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

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert "model.provider 'volcengine-plan' is not a recognised provider" not in out


def test_run_doctor_accepts_bare_custom_provider(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: custom\n"
        "  default: local-model\n"
        "  base_url: http://localhost:8000/v1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    (tmp_path / "project").mkdir(exist_ok=True)

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

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert "model.provider 'custom' is not a recognised provider" not in out


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


def test_run_doctor_kimi_cn_env_is_detected_and_probe_is_null_safe(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
    (home / ".env").write_text("KIMI_CN_API_KEY=sk-test\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setenv("KIMI_CN_API_KEY", "sk-test")

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

    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers, timeout))
        return types.SimpleNamespace(status_code=200)

    import httpx
    monkeypatch.setattr(httpx, "get", fake_get)

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    out = buf.getvalue()

    assert "API key or custom endpoint configured" in out
    assert "Kimi / Moonshot (China)" in out
    assert "str expected, not NoneType" not in out
    assert any(url == "https://api.moonshot.cn/v1/models" for url, _, _ in calls)


@pytest.mark.parametrize("base_url", [None, "https://opencode.ai/zen/go/v1"])
def test_run_doctor_opencode_go_skips_invalid_models_probe(monkeypatch, tmp_path, base_url):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
    (home / ".env").write_text("OPENCODE_GO_API_KEY=***\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-test")
    if base_url:
        monkeypatch.setenv("OPENCODE_GO_BASE_URL", base_url)
    else:
        monkeypatch.delenv("OPENCODE_GO_BASE_URL", raising=False)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
    except ImportError:
        pass

    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers, timeout))
        return types.SimpleNamespace(status_code=200)

    import httpx
    monkeypatch.setattr(httpx, "get", fake_get)

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    out = buf.getvalue()

    assert any(
        "OpenCode Go" in line and "(key configured)" in line
        for line in out.splitlines()
    )
    assert not any(url == "https://opencode.ai/zen/go/v1/models" for url, _, _ in calls)
    assert not any("opencode" in url.lower() and "models" in url.lower() for url, _, _ in calls)
