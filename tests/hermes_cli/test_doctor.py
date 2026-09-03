"""Tests for hermes_cli.doctor."""

import os
import subprocess
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


    def test_sqlite_upgrade_hint_recreates_docker_containers(self, monkeypatch):
        monkeypatch.setattr(doctor, "detect_install_method", lambda _root: "docker")

        hint = doctor._sqlite_upgrade_hint()

        assert "docker pull nousresearch/hermes-agent:latest" in hint
        assert "recreate all Hermes containers" in hint
        assert "hermes update" not in hint

    def test_sqlite_upgrade_hint_keeps_git_runtime_repair(self):
        hint = doctor._sqlite_upgrade_hint("git")

        assert "run `hermes update`" in hint

    def test_sqlite_upgrade_hint_uses_pkg_for_apt_managed_install(self):
        hint = doctor._sqlite_upgrade_hint("apt")

        assert "run `pkg upgrade hermes-agent`" in hint
        assert "hermes update" not in hint

    def test_sqlite_upgrade_hint_preserves_nix_guidance_as_prose(self):
        guidance = doctor.recommended_update_command_for_method("nix")
        hint = doctor._sqlite_upgrade_hint("nix")

        assert guidance in hint
        assert f"run `{guidance}`" not in hint
        assert "hermes update" not in hint


class TestProviderEnvDetection:
    def test_detects_openai_api_key(self):
        content = "OPENAI_BASE_URL=http://localhost:1234/v1\nOPENAI_API_KEY=***"
        assert _has_provider_env_config(content)


    def test_returns_false_when_no_provider_settings(self):
        content = "TERMINAL_ENV=local\n"
        assert not _has_provider_env_config(content)


class TestDoctorToolAvailabilitySummary:
    def test_missing_api_key_summary_ignores_disabled_toolsets(self, monkeypatch):
        unavailable = [
            {"name": "rl", "missing_vars": ["TINKER_API_KEY"]},
            {"name": "web", "missing_vars": ["EXA_API_KEY"]},
        ]
        monkeypatch.setattr(doctor, "_enabled_cli_toolsets_for_doctor", lambda: {"web"})

        filtered = doctor._missing_api_key_toolsets_for_summary(unavailable)

        assert [item["name"] for item in filtered] == ["web"]

    def test_web_capability_rows_warn_when_selected_provider_not_ready(self, monkeypatch):
        """#78412: selected firecrawl with is_available=False must warn."""
        class _Unavailable:
            name = "firecrawl"

            def is_available(self):
                return False

        unavailable = _Unavailable()
        monkeypatch.setattr(
            "agent.web_search_registry.get_active_search_provider",
            lambda: unavailable,
        )
        monkeypatch.setattr(
            "agent.web_search_registry.get_active_extract_provider",
            lambda: unavailable,
        )

        rows = doctor._doctor_web_capability_rows()
        assert rows
        assert all(status == "warn" for status, _, _ in rows)
        assert any("firecrawl selected; provider not configured" in detail for _, _, detail in rows)

    def test_web_capability_rows_ok_when_provider_ready(self, monkeypatch):
        class _Ready:
            name = "ddgs"

            def is_available(self):
                return True

        ready = _Ready()
        monkeypatch.setattr(
            "agent.web_search_registry.get_active_search_provider",
            lambda: ready,
        )
        monkeypatch.setattr(
            "agent.web_search_registry.get_active_extract_provider",
            lambda: ready,
        )

        rows = doctor._doctor_web_capability_rows()
        assert rows == [
            ("ok", "web search", "(ddgs)"),
            ("ok", "web extract", "(ddgs)"),
        ]


class TestDoctorEnvFileEncoding:
    """Regression for #18637 (bug 3): `hermes doctor` crashed on Windows
    Chinese locale (GBK) because `.env` was read with Path.read_text() which
    defaults to the system locale encoding, not UTF-8."""

    def test_doctor_reads_env_as_utf8_even_when_locale_is_not_utf8(
        self, monkeypatch, tmp_path
    ):
        import pathlib

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        # Write a UTF-8 .env containing an em dash (U+2014 = e2 80 94). The
        # 0x94 byte is exactly the one the issue reporter hit: it's invalid
        # as a GBK trailing byte in this position, so locale-default reads
        # raise UnicodeDecodeError on Chinese Windows.
        env_path = hermes_home / ".env"
        env_path.write_text(
            "OPENAI_API_KEY=sk-test  # em-dash here — should not crash\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", hermes_home)

        orig_read_text = pathlib.Path.read_text

        def gbk_like_read_text(self, encoding=None, errors=None, **kwargs):
            # Simulate a GBK locale: refuse to decode this specific UTF-8
            # .env unless the caller pins encoding="utf-8".
            if self == env_path and encoding != "utf-8":
                raise UnicodeDecodeError(
                    "gbk", b"\x94", 0, 1, "illegal multibyte sequence"
                )
            return orig_read_text(self, encoding=encoding, errors=errors, **kwargs)

        monkeypatch.setattr(pathlib.Path, "read_text", gbk_like_read_text)

        # Short-circuit the expensive tool-availability probe — we only
        # need doctor to reach the .env read without crashing.
        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: (_ for _ in ()).throw(SystemExit(0)),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        # Run doctor. If the .env read still uses locale encoding, this
        # raises UnicodeDecodeError and the test fails.
        with pytest.raises(SystemExit):
            doctor_mod.run_doctor(Namespace(fix=False))


    def test_doctor_reads_invalid_utf8_env_via_latin1_fallback(
        self, monkeypatch, tmp_path
    ):
        """cp1252/latin-1 .env with ASCII provider hints must not abort doctor."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        env_path = hermes_home / ".env"
        # 0xff is invalid UTF-8; latin-1 decodes it. Keep an ASCII provider key
        # so the scan still reports a configured endpoint/key.
        env_path.write_bytes(b"OPENAI_API_KEY=sk-test\xff\n")

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", hermes_home)

        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: (_ for _ in ()).throw(SystemExit(0)),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        with pytest.raises(SystemExit):
            doctor_mod.run_doctor(Namespace(fix=False))


class TestDoctorToolAvailabilityOverrides:


    def test_marks_kanban_available_only_when_missing_worker_env_gate(self, monkeypatch):
        monkeypatch.setattr(doctor, "_honcho_is_configured_for_doctor", lambda: False)
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        available, unavailable = doctor._apply_doctor_tool_availability_overrides(
            [],
            [{"name": "kanban", "env_vars": [], "tools": ["kanban_show"]}],
        )

        assert available == ["kanban"]
        assert unavailable == []

    def test_leaves_kanban_unavailable_when_worker_env_is_set(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "probe")
        kanban_entry = {"name": "kanban", "env_vars": [], "tools": ["kanban_show"]}

        available, unavailable = doctor._apply_doctor_tool_availability_overrides(
            [],
            [kanban_entry],
        )

        assert available == []
        assert unavailable == [kanban_entry]



    def test_suppresses_warning_for_default_off_toolset_missing_creds(self, monkeypatch):
        # Home Assistant is opt-in (_DEFAULT_OFF_TOOLSETS) — a clean install
        # that never configured/enabled it should not surface a doctor
        # warning. Resolver returns an enabled set that does NOT include
        # homeassistant, matching a clean install.
        monkeypatch.setattr(doctor, "_enabled_cli_toolsets_for_doctor", lambda: set())
        homeassistant_entry = {
            "name": "homeassistant",
            "missing_vars": ["HOMEASSISTANT_URL", "HOMEASSISTANT_TOKEN"],
            "tools": ["homeassistant_call_service"],
        }

        available, unavailable = doctor._apply_doctor_tool_availability_overrides(
            [],
            [homeassistant_entry],
        )

        assert available == []
        assert unavailable == []

    def test_suppresses_warning_for_each_default_off_toolset(self, monkeypatch):
        from hermes_cli.tools_config import _DEFAULT_OFF_TOOLSETS

        monkeypatch.setattr(doctor, "_enabled_cli_toolsets_for_doctor", lambda: set())
        entries = [
            {"name": name, "missing_vars": ["SOME_KEY"], "tools": ["some_tool"]}
            for name in _DEFAULT_OFF_TOOLSETS
        ]

        available, unavailable = doctor._apply_doctor_tool_availability_overrides([], entries)

        assert available == []
        assert unavailable == []

    def test_leaves_normally_enabled_toolset_unavailable_when_missing_creds(self, monkeypatch):
        # A non-opt-in toolset (e.g. a core web-search provider) missing its
        # API key must still be reported so the user can fix a real problem.
        monkeypatch.setattr(doctor, "_enabled_cli_toolsets_for_doctor", lambda: set())
        search_entry = {
            "name": "web_search",
            "missing_vars": ["SEARCH_API_KEY"],
            "tools": ["web_search"],
        }

        available, unavailable = doctor._apply_doctor_tool_availability_overrides(
            [],
            [search_entry],
        )

        assert available == []
        assert unavailable == [search_entry]

    def test_does_not_suppress_default_off_toolset_the_user_explicitly_enabled(self, monkeypatch):
        # Regression for sweeper review on PR #60364: a user who explicitly
        # enables a default-off toolset via `hermes tools` has opted into
        # wanting its missing-credential problems surfaced. Suppressing the
        # warning here would hide a real, actionable setup issue for an
        # integration the user is actively relying on.
        monkeypatch.setattr(
            doctor, "_enabled_cli_toolsets_for_doctor", lambda: {"homeassistant"}
        )
        homeassistant_entry = {
            "name": "homeassistant",
            "missing_vars": ["HOMEASSISTANT_URL", "HOMEASSISTANT_TOKEN"],
            "tools": ["homeassistant_call_service"],
        }

        available, unavailable = doctor._apply_doctor_tool_availability_overrides(
            [],
            [homeassistant_entry],
        )

        assert available == []
        assert unavailable == [homeassistant_entry]

    def test_does_not_suppress_when_enabled_resolver_fails(self, monkeypatch):
        # If config resolution fails (returns None), fail toward showing the
        # warning rather than guessing the toolset is safely unopted-into.
        monkeypatch.setattr(doctor, "_enabled_cli_toolsets_for_doctor", lambda: None)
        homeassistant_entry = {
            "name": "homeassistant",
            "missing_vars": ["HOMEASSISTANT_URL", "HOMEASSISTANT_TOKEN"],
            "tools": ["homeassistant_call_service"],
        }

        available, unavailable = doctor._apply_doctor_tool_availability_overrides(
            [],
            [homeassistant_entry],
        )

        assert available == []
        assert unavailable == [homeassistant_entry]

    def test_default_off_toolset_detection_helper(self):
        assert doctor._is_default_off_toolset({"name": "spotify"}, set()) is True
        assert doctor._is_default_off_toolset({"name": "spotify"}, {"spotify"}) is False
        assert doctor._is_default_off_toolset({"name": "spotify"}, None) is False
        assert doctor._is_default_off_toolset({"name": "web_search"}, set()) is False


class TestHonchoDoctorConfigDetection:
    def test_reports_configured_when_enabled_with_api_key(self, monkeypatch):
        fake_config = SimpleNamespace(enabled=True, api_key="***")

        monkeypatch.setattr(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            lambda: fake_config,
        )

        assert doctor._honcho_is_configured_for_doctor()








def test_doctor_reports_vercel_backend_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setenv("TERMINAL_VERCEL_RUNTIME", "python3.13")
    monkeypatch.setenv("TERMINAL_CONTAINER_DISK", "2048")
    monkeypatch.setenv("VERCEL_TOKEN", "super-secret-value")
    monkeypatch.delenv("VERCEL_PROJECT_ID", raising=False)
    monkeypatch.setenv("VERCEL_TEAM_ID", "team")
    monkeypatch.setattr(doctor_mod.importlib.util, "find_spec", lambda name: object() if name == "vercel" else None)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert "Vercel runtime" in out
    assert "python3.13" in out
    assert "Vercel custom disk unsupported" in out
    assert "Vercel auth incomplete" in out
    assert "VERCEL_PROJECT_ID" in out
    assert "Vercel auth mode: incomplete access token" in out
    assert "Vercel auth present env: VERCEL_TOKEN, VERCEL_TEAM_ID" in out
    assert "Vercel auth missing env: VERCEL_PROJECT_ID" in out
    assert "super-secret-value" not in out
    assert "snapshot filesystem only" in out


# ── Memory provider section (doctor should only check the *active* provider) ──


class TestDoctorMemoryProviderSection:
    """The ◆ Memory Provider section should respect memory.provider config."""

    def _make_hermes_home(self, tmp_path, provider="", memory_config=None):
        """Create a minimal HERMES_HOME with config.yaml."""
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        import yaml
        config = dict(memory_config or {})
        if provider:
            config["provider"] = provider
        config = {"memory": config}
        (home / "config.yaml").write_text(yaml.dump(config))
        return home

    def _run_doctor_and_capture(
        self,
        monkeypatch,
        tmp_path,
        provider="",
        *,
        memory_config=None,
        stale_builtin_files=False,
    ):
        """Run doctor and capture stdout."""
        home = self._make_hermes_home(tmp_path, provider, memory_config)
        if stale_builtin_files:
            memories = home / "memories"
            memories.mkdir()
            (memories / "MEMORY.md").write_text("stale memory", encoding="utf-8")
            (memories / "USER.md").write_text("stale user", encoding="utf-8")
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
            monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {})
            monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
            monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
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


    def test_mem0_provider_not_installed_shows_fail(self, monkeypatch, tmp_path):
        # Make mem0 import fail
        monkeypatch.setitem(sys.modules, "plugins.memory.mem0", None)
        out = self._run_doctor_and_capture(monkeypatch, tmp_path, provider="mem0")
        assert "Memory Provider" in out
        assert "Built-in memory active" not in out

    @pytest.mark.parametrize("memory_enabled", [False, True])
    def test_stale_builtin_files_reported_only_when_store_enabled(
        self, monkeypatch, tmp_path, memory_enabled
    ):
        # #100668: disabled built-in stores must not surface stale files as active.
        out = self._run_doctor_and_capture(
            monkeypatch,
            tmp_path,
            provider="mnemosyne",
            memory_config={
                "memory_enabled": memory_enabled,
                "user_profile_enabled": False,
            },
            stale_builtin_files=True,
        )

        assert ("MEMORY.md exists" in out) is memory_enabled
        assert "USER.md exists" not in out
        assert ("Built-in memory files disabled by config" in out) is not memory_enabled


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
    assert "Termux compatibility fallbacks:" in out
    assert "use .[termux-all] for broad compatibility" in out
    assert "Matrix E2EE extra is excluded on Termux" in out
    assert "Local faster-whisper extra is excluded on Termux" in out
    assert "STT fallback: use Groq Whisper (set GROQ_API_KEY) or OpenAI Whisper (set VOICE_TOOLS_OPENAI_KEY)." in out
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
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert "model.provider 'volcengine-plan' is not a recognised provider" not in out


def test_run_doctor_accepts_stable_key_when_provider_name_differs(
    monkeypatch, tmp_path
):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: custom:local-127.0.0.1:11434\n"
        "  default: qwen3.5:9b\n"
        "providers:\n"
        "  local-127.0.0.1:11434:\n"
        "    name: Local Ollama\n"
        "    base_url: http://127.0.0.1:11434/v1\n"
        "    default_model: qwen3.5:9b\n",
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
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert (
        "model.provider 'custom:local-127.0.0.1:11434' is not a recognised provider"
        not in out
    )
    assert "model.provider 'custom:local-127.0.0.1:11434' is unknown" not in out


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
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert "model.provider 'custom' is not a recognised provider" not in out


def test_run_doctor_flags_missing_credentials_for_active_openrouter_provider(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: openrouter\n"
        "  default: openai/gpt-4.1-mini\n",
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
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        from hermes_cli import auth as _auth_mod

        monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_minimax_oauth_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_gemini_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert "model.provider 'openrouter' is set but no API key is configured" in out
    assert "No credentials found for provider 'openrouter'." in out


@pytest.mark.parametrize(
    ("provider", "default_model"),
    [
        ("ai-gateway", "anthropic/claude-sonnet-4.6"),
        ("opencode-zen", "anthropic/claude-sonnet-4.6"),
        ("kilocode", "anthropic/claude-sonnet-4.6"),
        ("kimi-coding", "kimi-k2"),
        ("nvidia", "qwen/qwen3.5-122b-a10b"),
        ("moa", "anthropic/claude-sonnet-4.6"),
    ],
)
def test_run_doctor_accepts_hermes_provider_ids_that_catalog_aliases(
    monkeypatch, tmp_path, provider, default_model
):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        f"  provider: {provider}\n"
        f"  default: {default_model}\n",
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
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert f"model.provider '{provider}' is not a recognised provider" not in out
    assert f"model.provider '{provider}' is unknown" not in out
    if provider in {"ai-gateway", "opencode-zen", "kilocode", "nvidia"}:
        assert (
            f"model.default '{default_model}' uses a vendor/model slug but provider is '{provider}'"
            not in out
        )


def test_run_doctor_accepts_vendor_slugs_for_named_custom_provider(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: custom:hpc-ai\n"
        "  default: deepseek/deepseek-v4-flash\n"
        "custom_providers:\n"
        "  - name: hpc-ai\n"
        "    base_url: https://hpc-ai.example/v1\n"
        "    api_key: test-key\n",
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
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert "model.provider 'custom:hpc-ai' is not a recognised provider" not in out
    assert "model.provider 'custom:hpc-ai' is unknown" not in out
    assert (
        "model.default 'deepseek/deepseek-v4-flash' uses a vendor/model slug but provider is "
        "'custom:hpc-ai'"
        not in out
    )
    assert "Either set model.provider to 'openrouter', or drop the vendor prefix." not in out




def test_run_doctor_accepts_kimi_coding_cn_provider(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text("KIMI_CN_API_KEY=***\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: kimi-coding-cn\n"
        "  default: kimi-k2.6\n",
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
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_auth_status", lambda provider: {"logged_in": True})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert "model.provider 'kimi-coding-cn' is not a recognised provider" not in out


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
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
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


def _doctor_env_for_agent_browser(monkeypatch, tmp_path):
    """Shared non-Termux fixture setup for the agent-browser npx-resolution
    branch in run_doctor (hermes_cli/doctor.py ~1557-1605)."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)

    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "/usr")
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setattr(
        doctor_mod.shutil,
        "which",
        lambda cmd: "/usr/bin/node" if cmd in {"node", "npm"} else None,
    )

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass


def test_run_doctor_reports_agent_browser_resolves_via_npx(monkeypatch, tmp_path):
    """When agent-browser has no local/global install, _find_agent_browser
    falls through to 'npx agent-browser' — doctor must report that as OK
    (#43564: agent-browser is no longer a root package.json dependency, so
    this is the expected common case now, not a warning)."""
    _doctor_env_for_agent_browser(monkeypatch, tmp_path)

    import tools.browser_tool as bt
    monkeypatch.setattr(bt, "_find_agent_browser", lambda **_kw: "npx agent-browser")
    warm_calls = []
    monkeypatch.setattr(
        bt, "warm_agent_browser_npx_cache", lambda *a, **kw: warm_calls.append(1) or True
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    out = buf.getvalue()

    assert "agent-browser" in out
    assert "resolves via npx on first use" in out
    assert "agent-browser not installed" not in out
    # --fix was not requested: the warm-up must not fire on a plain check.
    assert not warm_calls


def test_run_doctor_fix_warms_npx_cache_when_agent_browser_resolves_via_npx(
    monkeypatch, tmp_path
):
    """`hermes doctor --fix` must actually call warm_agent_browser_npx_cache()
    when agent-browser resolves via npx, and report success."""
    _doctor_env_for_agent_browser(monkeypatch, tmp_path)

    import tools.browser_tool as bt
    monkeypatch.setattr(bt, "_find_agent_browser", lambda **_kw: "npx agent-browser")
    warm_calls = []
    monkeypatch.setattr(
        bt, "warm_agent_browser_npx_cache", lambda *a, **kw: warm_calls.append(1) or True
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=True))
    out = buf.getvalue()

    assert warm_calls, "warm_agent_browser_npx_cache() must be called under --fix"
    assert "Warmed npx cache for agent-browser" in out
    assert "Could not warm npx cache" not in out


def test_run_doctor_fix_reports_when_npx_warmup_fails(monkeypatch, tmp_path):
    """If warm_agent_browser_npx_cache() fails (offline, npx missing from
    PATH at call time, etc.), doctor must say so instead of silently
    claiming success — and must not count it as a fix."""
    _doctor_env_for_agent_browser(monkeypatch, tmp_path)

    import tools.browser_tool as bt
    monkeypatch.setattr(bt, "_find_agent_browser", lambda **_kw: "npx agent-browser")
    monkeypatch.setattr(bt, "warm_agent_browser_npx_cache", lambda *a, **kw: False)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=True))
    out = buf.getvalue()

    assert "Could not warm npx cache (offline or npx unavailable)" in out
    assert "Warmed npx cache for agent-browser" not in out


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
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
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


def test_run_doctor_dashscope_retries_china_endpoint_after_intl_unauthorized(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
    (home / ".env").write_text("DASHSCOPE_API_KEY=sk-test\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except ImportError:
        pass

    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers, timeout))
        status = 200 if "dashscope.aliyuncs.com" in url else 401
        return types.SimpleNamespace(status_code=status)

    import httpx
    monkeypatch.setattr(httpx, "get", fake_get)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    out = buf.getvalue()

    assert "Alibaba/DashScope" in out
    assert "invalid API key" not in out
    assert any(
        url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models"
        for url, _, _ in calls
    )
    assert any(
        url == "https://dashscope.aliyuncs.com/compatible-mode/v1/models"
        for url, _, _ in calls
    )


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
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
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


class TestGitHubTokenCheck:
    """Tests for GitHub token / gh auth detection in doctor."""

    @staticmethod
    def _isolate_home(monkeypatch, home):
        """Point doctor at the temp HERMES_HOME.

        ``run_doctor`` reads the module-level ``HERMES_HOME`` constant (cached
        at import time), NOT the env var — so ``setenv("HERMES_HOME")`` alone
        leaves doctor probing the REAL ~/.hermes. On a dev machine with a
        large state.db that meant a multi-minute ``PRAGMA integrity_check``
        that blew the 300s per-file budget and killed the whole file.
        """
        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))
        monkeypatch.setenv("HERMES_HOME", str(home))

    def test_no_token_and_not_gh_authenticated_shows_warn(self, monkeypatch, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        self._isolate_home(monkeypatch, home)
        monkeypatch.setenv("PATH", "/nonexistent")  # gh not found

        from hermes_cli.doctor import run_doctor
        import io, contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_doctor(Namespace(fix=False))
        out = buf.getvalue()

        assert "No GITHUB_TOKEN" in out
        assert "60 req/hr" in out


    def test_gh_authenticated_without_env_token_shows_ok(self, monkeypatch, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        self._isolate_home(monkeypatch, home)
        # No GITHUB_TOKEN or GH_TOKEN
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)

        # Mock gh to return success
        import shutil
        real_which = shutil.which
        def mock_which(cmd):
            return "/usr/local/bin/gh" if cmd == "gh" else real_which(cmd)
        monkeypatch.setattr(shutil, "which", mock_which)

        call_log = []
        def mock_run(cmd, **kwargs):
            call_log.append(cmd)
            if cmd[:2] == ["gh", "auth"]:
                result = types.SimpleNamespace(returncode=0, stdout="", stderr="")
            else:
                result = types.SimpleNamespace(returncode=1, stdout="", stderr="")
            return result

        import subprocess
        monkeypatch.setattr(subprocess, "run", mock_run)

        from hermes_cli.doctor import run_doctor
        import io, contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_doctor(Namespace(fix=False))
        out = buf.getvalue()

        assert "gh auth" in str(call_log) or any(c[0] == "gh" for c in call_log), f"gh not called: {call_log}"
        assert "GitHub authenticated via gh CLI" in out or "token configured" in out


def _run_doctor_with_healthy_oauth_fallback(
    monkeypatch,
    tmp_path,
    *,
    env_key: str,
    bad_key: str,
    failing_host: str,
    minimax_oauth_status: dict,
    xai_oauth_status: dict | None = None,
) -> str:
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: nous\n"
        "  default: moonshotai/kimi-k2.6\n",
        encoding="utf-8",
    )
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setenv(env_key, bad_key)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_CN_API_KEY", raising=False)
    monkeypatch.setenv(env_key, bad_key)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    from hermes_cli import auth as _auth_mod

    monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {"logged_in": True})
    monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
    monkeypatch.setattr(_auth_mod, "get_minimax_oauth_auth_status", lambda: minimax_oauth_status)
    _xai_status = xai_oauth_status if xai_oauth_status is not None else {}
    monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: _xai_status)

    def fake_get(url, headers=None, timeout=None):
        status = 401 if failing_host in url else 200
        return types.SimpleNamespace(status_code=status)

    import httpx

    monkeypatch.setattr(httpx, "get", fake_get)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    return buf.getvalue()


@pytest.mark.parametrize(
    ("env_key", "bad_key", "failing_host", "minimax_oauth_status", "xai_oauth_status", "unexpected_issue"),
    [
        (
            "MINIMAX_API_KEY",
            "bad-minimax-key",
            "minimax.io",
            {"logged_in": True, "region": "global"},
            None,
            "Check MINIMAX_API_KEY in .env",
        ),
        (
            "XAI_API_KEY",
            "bad-xai-key",
            "api.x.ai",
            {},
            {"logged_in": True, "auth_mode": "oauth_pkce"},
            "Check XAI_API_KEY in .env",
        ),
    ],
)
def test_run_doctor_ignores_invalid_direct_keys_when_oauth_fallback_is_healthy(
    monkeypatch,
    tmp_path,
    env_key,
    bad_key,
    failing_host,
    minimax_oauth_status,
    xai_oauth_status,
    unexpected_issue,
):
    out = _run_doctor_with_healthy_oauth_fallback(
        monkeypatch,
        tmp_path,
        env_key=env_key,
        bad_key=bad_key,
        failing_host=failing_host,
        minimax_oauth_status=minimax_oauth_status,
        xai_oauth_status=xai_oauth_status,
    )

    assert "invalid API key" in out
    assert unexpected_issue not in out


def test_has_healthy_oauth_fallback_returns_false_for_unknown_provider():
    from hermes_cli.doctor import _has_healthy_oauth_fallback_for_apikey_provider
    assert _has_healthy_oauth_fallback_for_apikey_provider("unknown-provider") is False


class TestHasHealthyOauthFallbackForXai:


    def test_returns_false_when_xai_import_unavailable(self, monkeypatch):
        import sys
        # Simulate get_xai_oauth_auth_status missing from auth module
        monkeypatch.delattr("hermes_cli.auth.get_xai_oauth_auth_status", raising=False)
        # Force doctor module to re-import the function
        monkeypatch.delitem(sys.modules, "hermes_cli.doctor", raising=False)
        from hermes_cli.doctor import _has_healthy_oauth_fallback_for_apikey_provider
        assert _has_healthy_oauth_fallback_for_apikey_provider("xai") is False


# ---------------------------------------------------------------------------
# ◆ Auth Providers — xAI OAuth display in run_doctor()
# ---------------------------------------------------------------------------


class TestDoctorXaiOAuthStatus:
    """The ◆ Auth Providers section must show xAI OAuth login state.

    xAI OAuth is checked in a *separate* try/except block so that an import
    failure (or runtime exception) cannot silence the Nous / Codex / Gemini /
    MiniMax rows that were already printed above it.
    """

    def _run(self, monkeypatch, tmp_path, *, xai_auth_fn) -> str:
        """Run doctor with a controlled xAI auth callable; return stdout."""
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))

        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {"logged_in": False})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {"logged_in": False})
        monkeypatch.setattr(_auth_mod, "get_minimax_oauth_auth_status", lambda: {"logged_in": False})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", xai_auth_fn)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))
        return buf.getvalue()


    def test_logged_in_does_not_emit_not_logged_in_on_xai_line(self, monkeypatch, tmp_path):
        out = self._run(
            monkeypatch, tmp_path,
            xai_auth_fn=lambda: {"logged_in": True},
        )
        assert "xAI OAuth" in out
        # The xAI OAuth line itself must say "(logged in)", not "(not logged in)".
        xai_line = next(l for l in out.splitlines() if "xAI OAuth" in l)
        assert "(logged in)" in xai_line
        assert "(not logged in)" not in xai_line


    def test_import_failure_does_not_affect_other_providers(self, monkeypatch, tmp_path):
        """Nous / Codex / Gemini / MiniMax rows must survive an xAI import failure."""
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))

        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {"logged_in": True})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {"logged_in": False})
        monkeypatch.setattr(_auth_mod, "get_minimax_oauth_auth_status", lambda: {"logged_in": False})
        monkeypatch.delattr(_auth_mod, "get_xai_oauth_auth_status", raising=False)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))
        out = buf.getvalue()
        assert "Nous Portal auth" in out
        assert "logged in" in out

    def test_function_raises_does_not_crash_doctor(self, monkeypatch, tmp_path):
        """A runtime exception from get_xai_oauth_auth_status must be swallowed."""
        def _raise():
            raise RuntimeError("simulated xAI status failure")

        out = self._run(monkeypatch, tmp_path, xai_auth_fn=_raise)
        assert "Auth Providers" in out


# ---------------------------------------------------------------------------
# ◆ Auth Providers — codex CLI import hint placement (issue #27975)
# ---------------------------------------------------------------------------


class TestDoctorCodexCliHintPlacement:
    """The `codex CLI not installed` hint belongs under OpenAI Codex auth.

    Regression for #27975: the hint used to be emitted as a standalone block
    after all auth-provider rows, so it visually attached to whichever
    provider happened to print last (MiniMax OAuth in the reported repro),
    reading as remediation for an unrelated provider.
    """

    def _run(self, monkeypatch, tmp_path, *, codex_logged_in: bool, codex_cli_present: bool) -> str:
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))

        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {"logged_in": False})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {"logged_in": codex_logged_in})
        monkeypatch.setattr(_auth_mod, "get_minimax_oauth_auth_status", lambda: {"logged_in": False})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {"logged_in": False})

        real_which = doctor_mod.shutil.which
        monkeypatch.setattr(
            doctor_mod.shutil,
            "which",
            lambda cmd: ("/usr/local/bin/codex" if codex_cli_present else None) if cmd == "codex" else real_which(cmd),
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))
        return buf.getvalue()

    @staticmethod
    def _hint_line() -> str:
        return "codex CLI not installed"

    def test_hint_appears_under_codex_auth_when_missing(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path, codex_logged_in=False, codex_cli_present=False)
        lines = out.splitlines()
        codex_idx = next(i for i, l in enumerate(lines) if "OpenAI Codex auth" in l)
        hint_idx = next(i for i, l in enumerate(lines) if self._hint_line() in l)
        minimax_idx = next(i for i, l in enumerate(lines) if "MiniMax OAuth" in l)
        # Hint must sit between Codex auth and the next provider row (#27975).
        assert codex_idx < hint_idx < minimax_idx

    def test_hint_suppressed_when_codex_cli_present(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path, codex_logged_in=False, codex_cli_present=True)
        assert "OpenAI Codex auth" in out
        assert self._hint_line() not in out


class TestDoctorStaleMaxIterationsDrift:
    """Regression for #17534: a stale HERMES_MAX_ITERATIONS in .env shadows
    agent.max_turns in config.yaml. The repro symptom is config.yaml saying
    400 while the gateway activity line reads N/90. Doctor must detect the
    drift, and `--fix` must remove the .env ghost (config.yaml wins).

    The detector reads the .env FILE directly, NOT os.environ — the gateway
    startup bridge can already have overridden os.environ to the config value,
    so the ghost is only visible in the file.
    """

    def _run_config_section(self, monkeypatch, tmp_path, *, fix, ghost, cfg_turns,
                            os_environ_value=None):
        import pathlib
        import contextlib
        import io
        from argparse import Namespace

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True)
        (hermes_home / "config.yaml").write_text(
            f"agent:\n  max_turns: {cfg_turns}\n", encoding="utf-8"
        )
        env_lines = ["OPENAI_API_KEY=sk-test\n"]
        if ghost is not None:
            env_lines.append(f"HERMES_MAX_ITERATIONS={ghost}\n")
        (hermes_home / ".env").write_text("".join(env_lines), encoding="utf-8")

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", hermes_home)
        monkeypatch.setattr(doctor_mod, "get_hermes_home", lambda: hermes_home)
        # Point the config helpers at the temp home.
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        if os_environ_value is not None:
            # Simulate the gateway bridge having already overridden os.environ.
            monkeypatch.setenv("HERMES_MAX_ITERATIONS", str(os_environ_value))
        else:
            monkeypatch.delenv("HERMES_MAX_ITERATIONS", raising=False)

        # Short-circuit at the Tool Availability stage — the drift check runs
        # well before it in the Configuration Files section.
        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: (_ for _ in ()).throw(SystemExit(0)),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
            doctor_mod.run_doctor(Namespace(fix=fix))
        return buf.getvalue(), hermes_home

    def test_detects_drift_warn_only(self, monkeypatch, tmp_path):
        out, hermes_home = self._run_config_section(
            monkeypatch, tmp_path, fix=False, ghost=90, cfg_turns=400,
            os_environ_value=400,  # bridge contaminated os.environ
        )
        assert "HERMES_MAX_ITERATIONS=90" in out
        assert "shadows" in out
        # Warn-only must NOT mutate .env.
        assert "HERMES_MAX_ITERATIONS=90" in (hermes_home / ".env").read_text(encoding="utf-8")

    def test_fix_removes_ghost(self, monkeypatch, tmp_path):
        out, hermes_home = self._run_config_section(
            monkeypatch, tmp_path, fix=True, ghost=90, cfg_turns=400,
            os_environ_value=400,
        )
        assert "Removed stale HERMES_MAX_ITERATIONS" in out
        env_after = (hermes_home / ".env").read_text(encoding="utf-8")
        assert "HERMES_MAX_ITERATIONS" not in env_after
        assert "OPENAI_API_KEY=sk-test" in env_after  # other keys preserved


    def test_no_drift_when_ghost_absent(self, monkeypatch, tmp_path):
        out, _ = self._run_config_section(
            monkeypatch, tmp_path, fix=False, ghost=None, cfg_turns=400,
        )
        assert "shadows" not in out




class TestDoctorDeprecatedConfigAndEnv:
    """Doctor must surface deprecated/legacy config keys and env vars with
    modern replacements as non-failing warnings — without auto-migrating.
    """



    def test_collect_deprecated_env_vars_ignores_empty(self):
        assert doctor_mod.collect_deprecated_env_vars({"TERMINAL_CWD": "  "}) == []
        assert doctor_mod.collect_deprecated_env_vars({}) == []
        assert doctor_mod.collect_deprecated_env_vars(None) == []

    def test_hermes_tool_progress_warning_says_unsupported_since_floor(self):
        """HERMES_TOOL_PROGRESS lost its last consumer (the retired v3→4
        migration) when the v12 support floor landed — doctor must say the
        variable is ignored rather than merely 'deprecated but read'."""
        findings = dict(
            doctor_mod.collect_deprecated_env_vars({"HERMES_TOOL_PROGRESS": "true"})
        )
        assert "ignored/unsupported since config floor v12" in findings["HERMES_TOOL_PROGRESS"]
        # The MODE variant is still read by the gateway fallback → keeps the
        # plain deprecation wording.
        mode = dict(
            doctor_mod.collect_deprecated_env_vars({"HERMES_TOOL_PROGRESS_MODE": "all"})
        )
        assert mode["HERMES_TOOL_PROGRESS_MODE"] == "display.tool_progress in config.yaml"

    def _run_doctor_with_config(self, monkeypatch, tmp_path, *, config_yaml: str, env_text: str = ""):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True)
        (hermes_home / "config.yaml").write_text(config_yaml, encoding="utf-8")
        env_body = env_text if env_text else "OPENAI_API_KEY=sk-test\n"
        (hermes_home / ".env").write_text(env_body, encoding="utf-8")

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", hermes_home)
        monkeypatch.setattr(doctor_mod, "get_hermes_home", lambda: hermes_home)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        # Clear process-level legacy env so tests only see the on-disk .env.
        for k in (
            "HERMES_TOOL_PROGRESS",
            "HERMES_TOOL_PROGRESS_MODE",
            "TERMINAL_CWD",
            "MESSAGING_CWD",
            "QQ_HOME_CHANNEL",
            "QQ_HOME_CHANNEL_NAME",
        ):
            monkeypatch.delenv(k, raising=False)

        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: (_ for _ in ()).throw(SystemExit(0)),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
            doctor_mod.run_doctor(Namespace(fix=False))
        return buf.getvalue(), hermes_home




    def test_report_does_not_count_as_blocking_issue(self, monkeypatch, tmp_path, capsys):
        """report_deprecated_config_and_env is warn-only — no issues list mutation."""
        findings = doctor_mod.report_deprecated_config_and_env(
            {"delegation": {"max_async_children": 2}},
            {"HERMES_TOOL_PROGRESS_MODE": "verbose"},
        )
        out = capsys.readouterr().out
        assert len(findings) == 2
        assert "Deprecated: delegation.max_async_children" in out
        assert "Deprecated: HERMES_TOOL_PROGRESS_MODE" in out
        assert "⚠" in out or "Deprecated" in out


class TestMacOSTCCGrants:
    """macOS TCC grant persistence check (issue #86385)."""

    def test_silent_on_non_macos(self, monkeypatch, capsys, tmp_path):
        """Non-macOS: the check must produce no output even with a bundle present."""
        monkeypatch.setattr(doctor_mod.sys, "platform", "linux")
        monkeypatch.setattr(
            doctor_mod,
            "_desktop_app_bundle",
            lambda: tmp_path / "Hermes.app",
        )
        doctor_mod.check_macos_tcc_grants()
        assert capsys.readouterr().out == ""

    def test_silent_when_no_desktop_bundle(self, monkeypatch, capsys):
        """No locally-built desktop bundle: nothing to check, no output."""
        monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
        monkeypatch.setattr(doctor_mod, "_desktop_app_bundle", lambda: None)
        doctor_mod.check_macos_tcc_grants()
        assert capsys.readouterr().out == ""

    def test_warns_on_cdhash_pinned_dr(self, monkeypatch, capsys, tmp_path):
        """Pre-#73681 builds have a cdhash-pinned DR → warn that grants reset."""
        monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
        monkeypatch.setattr(
            doctor_mod,
            "_desktop_app_bundle",
            lambda: tmp_path / "Hermes.app",
        )
        monkeypatch.setattr(
            doctor_mod,
            "_macos_desktop_dr",
            lambda app: 'designated => identifier "com.nousresearch.hermes" and cdhash H"97e692f3890f781fa0ad5ad6cb9d769cfaf42628"',
        )
        doctor_mod.check_macos_tcc_grants()
        out = capsys.readouterr().out
        assert "TCC grants will reset after every update" in out
        assert "cdhash-pinned" in out
        assert "hermes update" in out

    def test_ok_and_repair_info_on_identifier_dr(self, monkeypatch, capsys, tmp_path):
        """Post-#73681 identifier-only DR → stable + stale-grant repair info."""
        monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
        monkeypatch.setattr(
            doctor_mod,
            "_desktop_app_bundle",
            lambda: tmp_path / "Hermes.app",
        )
        monkeypatch.setattr(
            doctor_mod,
            "_macos_desktop_dr",
            lambda app: 'designated => identifier "com.nousresearch.hermes"',
        )
        doctor_mod.check_macos_tcc_grants()
        out = capsys.readouterr().out
        assert "TCC signing identity is stable" in out
        assert "identifier-pinned" in out
        # Identifier-pinned is stable but not the strongest anchor — the check
        # should point at the cert-anchored upgrade path.
        assert "--setup-tcc-identity" in out
        assert "tccutil reset ScreenCapture com.nousresearch.hermes" in out
        assert "toggle" in out
        assert "relaunch" in out

    def test_ok_on_certificate_anchored_dr(self, monkeypatch, capsys, tmp_path):
        """A cert-anchored DR (hermes desktop --setup-tcc-identity, or a
        notarized release) classifies as stable in its own class — no upgrade
        hint, still prints the stale-grant repair info."""
        monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
        monkeypatch.setattr(
            doctor_mod,
            "_desktop_app_bundle",
            lambda: tmp_path / "Hermes.app",
        )
        monkeypatch.setattr(
            doctor_mod,
            "_macos_desktop_dr",
            lambda app: 'designated => identifier "com.nousresearch.hermes" and certificate root = H"aabbcc"',
        )
        doctor_mod.check_macos_tcc_grants()
        out = capsys.readouterr().out
        assert "TCC signing identity is stable" in out
        assert "certificate-anchored" in out
        assert "--setup-tcc-identity" not in out
        assert "tccutil reset ScreenCapture com.nousresearch.hermes" in out

    def test_warns_when_dr_unreadable(self, monkeypatch, capsys, tmp_path):
        """codesign failure → warn, never crash."""
        monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
        monkeypatch.setattr(
            doctor_mod,
            "_desktop_app_bundle",
            lambda: tmp_path / "Hermes.app",
        )
        monkeypatch.setattr(doctor_mod, "_macos_desktop_dr", lambda app: None)
        doctor_mod.check_macos_tcc_grants()
        out = capsys.readouterr().out
        assert "could not read code-signing requirement" in out

    def test_warns_when_dr_empty_string(self, monkeypatch, capsys, tmp_path):
        """Empty DR output must not false-positive as a stable identity."""
        monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
        monkeypatch.setattr(
            doctor_mod,
            "_desktop_app_bundle",
            lambda: tmp_path / "Hermes.app",
        )
        monkeypatch.setattr(doctor_mod, "_macos_desktop_dr", lambda app: "")
        doctor_mod.check_macos_tcc_grants()
        out = capsys.readouterr().out
        assert "could not read code-signing requirement" in out
        assert "stable" not in out

    def test_warns_when_codesign_times_out(self, monkeypatch, capsys, tmp_path):
        """A hanging codesign must degrade to the unreadable-DR warning, never crash."""
        monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
        monkeypatch.setattr(
            doctor_mod,
            "_desktop_app_bundle",
            lambda: tmp_path / "Hermes.app",
        )

        def _timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=["codesign"], timeout=15)

        monkeypatch.setattr(doctor_mod.subprocess, "run", _timeout)
        doctor_mod.check_macos_tcc_grants()
        out = capsys.readouterr().out
        assert "could not read code-signing requirement" in out
        assert "stable" not in out

    def test_warns_when_codesign_missing(self, monkeypatch, capsys, tmp_path):
        """No codesign binary → same graceful unreadable-DR warning."""
        monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
        monkeypatch.setattr(
            doctor_mod,
            "_desktop_app_bundle",
            lambda: tmp_path / "Hermes.app",
        )
        monkeypatch.setattr(doctor_mod.shutil, "which", lambda _name: None)
        doctor_mod.check_macos_tcc_grants()
        out = capsys.readouterr().out
        assert "could not read code-signing requirement" in out
        assert "stable" not in out


def test_run_doctor_reports_shadowed_lightpanda_engine(monkeypatch, tmp_path):
    helper = TestDoctorMemoryProviderSection()
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_using_lightpanda_engine", lambda: True)
    monkeypatch.setattr(
        bt, "lightpanda_engine_status",
        lambda: (False, "cloud provider Browserbase is selected"),
    )
    out = helper._run_doctor_and_capture(monkeypatch, tmp_path)
    assert "browser.engine=lightpanda is shadowed" in out
    assert "Browserbase" in out


def test_run_doctor_reports_lightpanda_ok(monkeypatch, tmp_path):
    helper = TestDoctorMemoryProviderSection()
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_using_lightpanda_engine", lambda: True)
    monkeypatch.setattr(bt, "lightpanda_engine_status", lambda: (True, "Browser Use mode"))
    monkeypatch.setattr("tools.browser_lightpanda.find_lightpanda_binary", lambda: "/opt/lightpanda")
    out = helper._run_doctor_and_capture(monkeypatch, tmp_path)
    assert "Lightpanda" in out
    assert "shadowed" not in out


def test_run_doctor_warns_when_lightpanda_binary_missing(monkeypatch, tmp_path):
    helper = TestDoctorMemoryProviderSection()
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_using_lightpanda_engine", lambda: True)
    monkeypatch.setattr(bt, "lightpanda_engine_status", lambda: (True, "Browser Use mode"))
    monkeypatch.setattr("tools.browser_lightpanda.find_lightpanda_binary", lambda: None)
    out = helper._run_doctor_and_capture(monkeypatch, tmp_path)
    assert "Lightpanda selected but binary not found" in out
