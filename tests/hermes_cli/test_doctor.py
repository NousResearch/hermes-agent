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


    def test_sqlite_upgrade_hint_recreates_docker_containers(self, monkeypatch):
        monkeypatch.setattr(doctor, "detect_install_method", lambda _root: "docker")

        hint = doctor._sqlite_upgrade_hint()

        assert "docker pull nousresearch/hermes-agent:latest" in hint
        assert "recreate all Hermes containers" in hint
        assert "hermes update" not in hint

    def test_sqlite_upgrade_hint_keeps_git_runtime_repair(self):
        hint = doctor._sqlite_upgrade_hint("git")

        assert "run `hermes update`" in hint


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




class TestHonchoDoctorConfigDetection:
    def test_reports_configured_when_enabled_with_api_key(self, monkeypatch):
        fake_config = SimpleNamespace(enabled=True, api_key="***")

        monkeypatch.setattr(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            lambda: fake_config,
        )

        assert doctor._honcho_is_configured_for_doctor()








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




def _run_doctor_with_managed_agent_browser(monkeypatch, tmp_path, runnable):
    """Set up run_doctor with node present, agent-browser only in the
    Hermes-managed node bin (~/.hermes/node/bin), not on PATH or in
    PROJECT_ROOT/node_modules. Returns the captured stdout."""
    home = tmp_path / ".hermes"
    (home / "node" / "bin").mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
    managed_ab = home / "node" / "bin" / "agent-browser"
    managed_ab.write_text("#!/bin/sh\n", encoding="utf-8")
    managed_ab.chmod(0o755)
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)  # no node_modules/agent-browser here

    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))

    # node on PATH, agent-browser is NOT on PATH (only in the managed bin).
    # The managed-dir rung resolves via shutil.which(..., path=<dir>) so
    # Windows picks the .cmd shim — mirror that shape here.
    def _fake_which(cmd, path=None):
        if path is not None:
            if cmd == "agent-browser" and str(managed_ab.parent) == str(path):
                return str(managed_ab)
            return None
        return "/usr/bin/node" if cmd in {"node", "npm"} else None

    monkeypatch.setattr(doctor_mod.shutil, "which", _fake_which)
    # agent_browser_runnable is imported into doctor's namespace
    monkeypatch.setattr(
        doctor_mod,
        "agent_browser_runnable",
        lambda path: runnable and str(path) == str(managed_ab),
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

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    return buf.getvalue()




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
