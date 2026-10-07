from types import SimpleNamespace

from hermes_cli.status import show_status
import subprocess


def test_show_status_all_does_not_print_keenable_key_value(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sentinel = "NONSECRET_SENTINEL_VALUE_DO_NOT_PRINT_123456"
    monkeypatch.setenv("KEENABLE_API_KEY", sentinel)

    show_status(SimpleNamespace(full=True, deep=False))

    output = capsys.readouterr().out
    assert "Keenable" in output
    assert sentinel not in output


def test_show_status_all_does_not_print_tavily_key_value(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sentinel = "NONSECRET_SENTINEL_VALUE_DO_NOT_PRINT_TAVILY_123456"
    monkeypatch.setenv("TAVILY_API_KEY", sentinel)

    show_status(SimpleNamespace(full=True, deep=False))

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

    status_mod.show_status(SimpleNamespace(full=True, deep=False))

    output = capsys.readouterr().out
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

    status_mod.show_status(SimpleNamespace(full=True, deep=False))

    output = capsys.readouterr().out
    assert "vercel_sandbox" in output
    assert "python3.13" in output
    assert "VERCEL_OIDC_TOKEN" in output
    assert "oidc-token" not in output


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

        status_mod.show_status(SimpleNamespace(full=True, deep=False))
        out = capsys.readouterr().out

        assert "Auth file:  /home/u/.hermes/auth.json" in out




    # ------------------------------------------------------------------
    # Not-logged-in branch
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Resilience: import failure and runtime exception
    # ------------------------------------------------------------------


    def test_import_failure_does_not_break_other_oauth_providers(self, monkeypatch, capsys, tmp_path):
        """Nous/Codex/MiniMax rows must still appear when xAI import fails."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_nous_auth_status_local",
                            lambda: {"logged_in": True}, raising=False)
        monkeypatch.delattr(auth_mod, "get_xai_oauth_auth_status", raising=False)

        status_mod.show_status(SimpleNamespace(full=True, deep=False))
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

        status_mod.show_status(SimpleNamespace(full=True, deep=False))
        out = capsys.readouterr().out

        assert "◆ Auth Providers" in out

    def test_status_function_returns_none_does_not_crash(self, monkeypatch, capsys, tmp_path):
        """get_xai_oauth_auth_status returning None must be handled gracefully."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: None, raising=False)

        status_mod.show_status(SimpleNamespace(full=True, deep=False))
        out = capsys.readouterr().out

        assert "xAI OAuth" in out


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

    status_mod.show_status(SimpleNamespace(full=True, deep=False))
    output = capsys.readouterr().out
    assert "Active:       2 session(s)" in output
    assert "Last activity:" in output
    assert "1m ago" in output


def test_show_status_json_schema_and_keys(monkeypatch, capsys, tmp_path):
    """hermes status --json emits valid JSON with expected contract keys (#103176)."""
    import json
    from hermes_cli import status as status_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "test-model-42"}, raising=False)

    status_mod.show_status(_status_args("--json"))
    out = capsys.readouterr().out
    data = json.loads(out)

    assert isinstance(data, dict)
    assert "version" in data
    assert "model" in data
    assert data["model"] == "test-model-42"
    assert "provider" in data
    assert "estop" in data
    assert isinstance(data["estop"], dict)
    assert "paused" in data["estop"]
    assert "gateway" in data
    assert isinstance(data["gateway"], dict)
    assert "running" in data["gateway"]
    assert "manager" in data["gateway"]
    assert "auth" in data
    assert isinstance(data["auth"], dict)
    assert "api_keys" in data["auth"]
    assert "oauth" in data["auth"]
    assert "sessions" in data
    assert isinstance(data["sessions"], dict)
    assert "active" in data["sessions"]
    assert "total" in data["sessions"]
    assert "recent" in data["sessions"]
    assert "usage" in data
    assert isinstance(data["usage"], dict)
    assert "tokens" in data["usage"]
    assert "cost_usd" in data["usage"]


def test_show_status_json_does_not_leak_secrets(monkeypatch, capsys, tmp_path):
    """hermes status --json must never output secret API keys or token strings."""
    import json
    from hermes_cli import status as status_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sentinel = "SUPER_SECRET_TOKEN_DO_NOT_LEAK_999888"
    monkeypatch.setenv("OPENROUTER_API_KEY", sentinel)

    status_mod.show_status(SimpleNamespace(all=True, deep=False, json=True))
    out = capsys.readouterr().out
    assert sentinel not in out

    data = json.loads(out)
    assert "OpenRouter" in data["auth"]["api_keys"]


def test_show_status_json_estop_active(monkeypatch, capsys, tmp_path):
    """hermes status --json reflects active ESTOP pause state and reason."""
    import json
    import agent.estop as estop_mod
    from hermes_cli import status as status_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        estop_mod,
        "get_state",
        lambda: {"reason": "Maintenance window", "timestamp": 1757000000.0},
        raising=False,
    )

    status_mod.show_status(_status_args("--json"))
    out = capsys.readouterr().out
    data = json.loads(out)

    assert data["estop"]["paused"] is True
    assert data["estop"]["reason"] == "Maintenance window"
    assert data["estop"]["timestamp"] == 1757000000.0


def _status_args(*argv):
    import argparse
    from hermes_cli.subcommands.status import build_status_parser

    parser = argparse.ArgumentParser()
    build_status_parser(parser.add_subparsers(), cmd_status=None)
    return parser.parse_args(["status", *argv])


def test_status_defaults_to_summary_and_full_or_all_prints_every_section(monkeypatch, capsys, tmp_path):
    import hermes_cli.model_switch as model_switch

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(model_switch, "list_authenticated_providers", lambda **_: [{"name": "Acme AI"}])

    show_status(_status_args())
    summary = capsys.readouterr().out
    assert "Providers:    Acme AI" in summary
    assert "◆ Environment" not in summary
    for flag in ("--full", "--all"):
        show_status(_status_args(flag))
        assert "◆ Environment" in capsys.readouterr().out


def test_platform_rows_follow_the_gateway_verdict_not_check_fn(monkeypatch, capsys, tmp_path):
    """check_fn only says the SDK imports; each platform gets ONE row, judged like the summary line."""
    import gateway.config as gateway_config
    from gateway.platform_registry import platform_registry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entries = [SimpleNamespace(name=n, label=n.title(), check_fn=lambda: True) for n in ("telegram", "discord")]
    monkeypatch.setattr(platform_registry, "plugin_entries", lambda: entries)
    monkeypatch.setattr(gateway_config, "load_gateway_config", lambda: SimpleNamespace(
        platforms={}, get_connected_platforms=lambda: [SimpleNamespace(value="telegram")]))

    show_status(SimpleNamespace(full=True, deep=False))
    out = capsys.readouterr().out
    rows = {name: [line for line in out.splitlines() if line.strip().startswith(name)] for name in ("Telegram", "Discord")}
    assert [("not configured" in line) for line in rows["Discord"]] == [True]
    assert [("not configured" in line) for line in rows["Telegram"]] == [False]
    show_status(SimpleNamespace())
    assert "Platforms:    Telegram\n" in capsys.readouterr().out


    # JSON uses the same verdict, even when full/deep flags are supplied.
    import json
    show_status(_status_args("--json", "--full", "--deep"))
    platforms = json.loads(capsys.readouterr().out)["platforms"]
    assert platforms["Telegram"]["configured"] is True
    assert platforms["Discord"]["configured"] is False


def test_status_json_reads_sessions_without_creating_a_writer(monkeypatch, tmp_path):
    import hermes_state
    from hermes_cli.status import get_status_data

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert get_status_data()["sessions"]["total"] == 0
    assert not (tmp_path / "state.db").exists()

    db = hermes_state.SessionDB()
    try:
        db.create_session("json-status-session", "cli")
        db.append_message("json-status-session", "user", "Hello")
        db.update_token_counts("json-status-session", input_tokens=30, output_tokens=12)
    finally:
        db.close()

    real_db = hermes_state.SessionDB
    opened = []

    def tracked_db(*args, **kwargs):
        handle = real_db(*args, **kwargs)
        opened.append(handle.read_only)
        return handle

    monkeypatch.setattr(hermes_state, "SessionDB", tracked_db)
    data = get_status_data()
    assert opened == [True]
    assert data["sessions"]["total"] == 1
    assert data["sessions"]["recent"][0]["id"] == "json-status-session"
    assert data["usage"]["tokens"] == 42
