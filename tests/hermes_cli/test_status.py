from types import SimpleNamespace

from hermes_cli.status import show_status
import subprocess


def test_show_status_all_does_not_print_keenable_key_value(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sentinel = "NONSECRET_SENTINEL_VALUE_DO_NOT_PRINT_123456"
    monkeypatch.setenv("KEENABLE_API_KEY", sentinel)

    show_status(SimpleNamespace(all=True, deep=False))

    output = capsys.readouterr().out
    assert "Keenable" in output
    assert sentinel not in output


def test_show_status_all_does_not_print_tavily_key_value(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sentinel = "NONSECRET_SENTINEL_VALUE_DO_NOT_PRINT_TAVILY_123456"
    monkeypatch.setenv("TAVILY_API_KEY", sentinel)

    show_status(SimpleNamespace(all=True, deep=False))

    output = capsys.readouterr().out
    assert "Tavily" in output
    assert sentinel not in output


def test_show_status_termux_gateway_section_skips_systemctl(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status_local", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)

    def _unexpected_systemctl(*args, **kwargs):
        raise AssertionError("systemctl should not be called in the Termux status view")

    monkeypatch.setattr(subprocess, "run", _unexpected_systemctl)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Manager:      Termux / manual process" in output
    assert "Start with:   hermes gateway" in output
    assert "systemd (user)" not in output
def test_show_status_reports_vercel_backend_contract(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setenv("TERMINAL_VERCEL_RUNTIME", "python3.13")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
    monkeypatch.setattr(status_mod.importlib.util, "find_spec", lambda name: object() if name == "vercel" else None)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"terminal": {"backend": "vercel_sandbox"}}, raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Backend:      vercel_sandbox" in output
    assert "Runtime:      python3.13" in output
    assert "Auth:" in output and "OIDC token via VERCEL_OIDC_TOKEN" in output
    assert "Auth detail:  mode: OIDC" in output
    assert "Auth detail:  active env: VERCEL_OIDC_TOKEN" in output
    assert "oidc-token" not in output
    assert "snapshot filesystem" in output
    assert "live processes do not survive" in output


# ---------------------------------------------------------------------------
# Helpers shared by xAI OAuth status tests
# ---------------------------------------------------------------------------

def _base_xai_mocks(monkeypatch, tmp_path):
    """Set up the minimal environment for show_status, returning status_mod."""
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status_local", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_minimax_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)
    return status_mod


class TestShowStatusXaiOAuth:
    """xAI OAuth row in hermes status."""

    # ------------------------------------------------------------------
    # Logged-in branch
    # ------------------------------------------------------------------


    def test_logged_in_shows_auth_store(self, monkeypatch, capsys, tmp_path):
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": True, "auth_store": "/home/u/.hermes/auth.json"},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "Auth file:  /home/u/.hermes/auth.json" in out


    def test_no_auth_store_line_when_field_absent(self, monkeypatch, capsys, tmp_path):
        """Auth file line must not appear when auth_store is missing."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": True},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        xai_section = out.split("xAI OAuth", 1)[1].split("◆", 1)[0]
        assert "Auth file:" not in xai_section


    # ------------------------------------------------------------------
    # Not-logged-in branch
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Resilience: import failure and runtime exception
    # ------------------------------------------------------------------

    def test_import_failure_does_not_crash_show_status(self, monkeypatch, capsys, tmp_path):
        """show_status must complete even when get_xai_oauth_auth_status cannot be imported."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.delattr(auth_mod, "get_xai_oauth_auth_status", raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "◆ Auth Providers" in out

    def test_import_failure_does_not_break_other_oauth_providers(self, monkeypatch, capsys, tmp_path):
        """Nous/Codex/MiniMax rows must still appear when xAI import fails."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_nous_auth_status_local",
                            lambda: {"logged_in": True}, raising=False)
        monkeypatch.delattr(auth_mod, "get_xai_oauth_auth_status", raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "Nous Portal" in out
        assert "MiniMax OAuth" in out

    def test_status_function_exception_does_not_crash(self, monkeypatch, capsys, tmp_path):
        """show_status must not propagate an exception raised by get_xai_oauth_auth_status."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)

        def _raises():
            raise RuntimeError("backend unreachable")

        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", _raises, raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "◆ Auth Providers" in out

    def test_status_function_returns_none_does_not_crash(self, monkeypatch, capsys, tmp_path):
        """get_xai_oauth_auth_status returning None must be handled gracefully."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: None, raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "xAI OAuth" in out
        assert "not logged in (run: hermes auth add xai-oauth)" in out


def test_show_status_reports_gateway_session_last_activity(monkeypatch, capsys, tmp_path):
    """hermes status should surface freshest gateway last_active (#72016)."""
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod
    import hermes_state
    import time

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)

    class _FakeDB:
        def __init__(self, **_kwargs):
            pass

        def list_gateway_sessions(self, active_only=True):
            return [
                {"id": "gw-old", "last_active": time.time() - 7200},
                {"id": "gw-new", "last_active": time.time() - 90},
            ]

        def close(self):
            return None

    monkeypatch.setattr(hermes_state, "SessionDB", _FakeDB)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))
    output = capsys.readouterr().out
    assert "Active:       2 session(s)" in output
    assert "Last activity:" in output
    assert "1m ago" in output


def _strip_ansi(text):
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_status_rows_wrap_to_the_terminal_with_hanging_indent(monkeypatch, capsys):
    """Long values wrap at COLUMNS with the continuation aligned under the value (#109625)."""
    from hermes_cli import status as status_mod

    monkeypatch.setenv("COLUMNS", "40")
    status_mod._row("xAI OAuth", False, "not logged in (run: hermes auth add xai-oauth) first second third")
    out = capsys.readouterr().out.splitlines()
    stripped = [_strip_ansi(line) for line in out]
    assert all(len(line) <= 40 for line in stripped), stripped
    # Continuation lines align under the value column, not the label.
    assert stripped[0].startswith("  xAI OAuth")
    assert stripped[1].startswith(" " * 18)
    assert not stripped[1].startswith("  xAI OAuth")


def test_status_kv_wraps_and_keeps_alignment(monkeypatch, capsys):
    from hermes_cli import status as status_mod

    monkeypatch.setenv("COLUMNS", "40")
    status_mod._kv("Error:", "No xAI OAuth credentials stored. Select xAI Grok OAuth (SuperGrok / Premium+) in `hermes model`.")
    out = capsys.readouterr().out.splitlines()
    assert all(len(line) <= 40 for line in out), out
    assert out[0].startswith("  Error:")
    assert out[1].startswith(" " * 16)


def test_status_wrapping_survives_colored_values(monkeypatch, capsys):
    """SGR escapes carry zero width and are never split across lines."""
    from hermes_cli import status as status_mod

    monkeypatch.setenv("COLUMNS", "40")
    # A raw escape, not color(): colors disable themselves when stdout is captured.
    value = "\x1b[2mstopped\x1b[0m " + " ".join(f"word{i}" for i in range(20))
    status_mod._kv("Status:", value)
    out = capsys.readouterr().out
    stripped = [_strip_ansi(line) for line in out.splitlines()]
    assert all(len(line) <= 40 for line in stripped), stripped
    # Every escape that opened is still present somewhere in the output.
    assert "\x1b[2m" in out and "\x1b[0m" in out


def test_status_wrapping_disabled_when_terminal_is_wide(monkeypatch, capsys):
    """A value shorter than the terminal prints exactly as before, single line."""
    from hermes_cli import status as status_mod

    monkeypatch.setenv("COLUMNS", "120")
    status_mod._kv("Manager:", "systemd (user)")
    out = capsys.readouterr().out
    assert out == "  Manager:      systemd (user)\n"
