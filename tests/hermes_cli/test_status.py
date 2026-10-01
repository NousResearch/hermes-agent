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

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
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


# ── `hermes status --short` ────────────────────────────────────────────────


def _short_env(monkeypatch, tmp_path):
    """Quiet, hermetic environment for --short runs; returns (status_mod, auth_mod, gateway_mod)."""
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "test-model"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "test", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "test", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "Test Provider", raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status_local", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_minimax_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(
        gateway_mod, "get_gateway_runtime_snapshot",
        lambda: SimpleNamespace(running=False, manager="unknown", gateway_pids=[],
                                has_process_service_mismatch=False, service_installed=False,
                                service_running=False),
        raising=False)
    monkeypatch.setattr(gateway_mod, "named_profile_served_by_running_multiplexer", lambda: False, raising=False)
    return status_mod, auth_mod, gateway_mod


def test_status_parser_accepts_short_flag():
    import argparse
    from hermes_cli.subcommands.status import build_status_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    build_status_parser(subparsers, cmd_status=lambda args: None)

    assert parser.parse_args(["status", "--short"]).short is True
    assert parser.parse_args(["status"]).short is False
    both = parser.parse_args(["status", "--all", "--deep"])
    assert both.all is True and both.deep is True


def test_short_reports_model_provider_and_connected_providers(monkeypatch, capsys, tmp_path):
    status_mod, auth_mod, _ = _short_env(monkeypatch, tmp_path)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status_local", lambda: {"logged_in": True}, raising=False)
    monkeypatch.setenv("GLM_API_KEY", "glm-key-1234567890")

    status_mod.show_status(SimpleNamespace(all=False, deep=False, short=True))
    out = capsys.readouterr().out

    assert "status (short)" in out
    assert "Model:        test-model" in out
    assert "Provider:     Test Provider" in out
    assert "Providers:    Nous Portal, Z.AI / GLM" in out
    assert "none connected" not in out
    assert "hermes doctor" not in out and "hermes setup" not in out


def test_short_new_registry_provider_appears_when_env_var_set(monkeypatch, capsys, tmp_path):
    status_mod, auth_mod, _ = _short_env(monkeypatch, tmp_path)
    fake = auth_mod.ProviderConfig("acme", "Acme AI", "api_key", api_key_env_vars=("ACME_API_KEY",))
    monkeypatch.setattr(auth_mod, "PROVIDER_REGISTRY", {**auth_mod.PROVIDER_REGISTRY, "acme": fake}, raising=False)
    monkeypatch.delenv("ACME_API_KEY", raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False, short=True))
    assert "Acme AI" not in capsys.readouterr().out

    monkeypatch.setenv("ACME_API_KEY", "acme-key-1234567890")
    status_mod.show_status(SimpleNamespace(all=False, deep=False, short=True))
    assert "Acme AI" in capsys.readouterr().out


def test_short_shows_none_connected_without_credentials(monkeypatch, capsys, tmp_path):
    status_mod, auth_mod, _ = _short_env(monkeypatch, tmp_path)
    monkeypatch.setattr(auth_mod, "PROVIDER_REGISTRY", {}, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False, short=True))
    out = capsys.readouterr().out
    assert "Providers:    none connected" in out


def test_short_does_not_print_key_values(monkeypatch, capsys, tmp_path):
    status_mod, auth_mod, _ = _short_env(monkeypatch, tmp_path)
    sentinel = "NONSECRET_SENTINEL_VALUE_DO_NOT_PRINT_SHORT_123456"
    monkeypatch.setenv("GLM_API_KEY", sentinel)

    status_mod.show_status(SimpleNamespace(all=False, deep=False, short=True))
    out = capsys.readouterr().out
    assert "Z.AI / GLM" in out
    assert sentinel not in out


def test_short_gateway_row_reflects_snapshot(monkeypatch, capsys, tmp_path):
    status_mod, auth_mod, gateway_mod = _short_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        gateway_mod, "get_gateway_runtime_snapshot",
        lambda: SimpleNamespace(running=True, manager="systemd (user)", gateway_pids=[123],
                                has_process_service_mismatch=False, service_installed=True,
                                service_running=True),
        raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False, short=True))
    out = capsys.readouterr().out
    assert "Gateway:" in out and "running" in out
    assert "stopped" not in out


def test_short_platforms_row_lists_configured_only(monkeypatch, capsys, tmp_path):
    status_mod, auth_mod, _ = _short_env(monkeypatch, tmp_path)
    monkeypatch.setattr(status_mod, "_PLATFORMS", {"TestNet": ("TESTNET_TOKEN", None)}, raising=False)
    monkeypatch.setenv("TESTNET_TOKEN", "token-123")
    from gateway.platform_registry import platform_registry
    monkeypatch.setattr(platform_registry, "plugin_entries", lambda: [], raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False, short=True))
    out = capsys.readouterr().out
    assert "Platforms:    TestNet" in out


def test_full_status_output_unchanged_without_short_attr(monkeypatch, capsys, tmp_path):
    status_mod, auth_mod, _ = _short_env(monkeypatch, tmp_path)
    status_mod.show_status(SimpleNamespace(all=False, deep=False))
    out = capsys.readouterr().out
    assert "☤ Hermes Agent Status" in out
    assert "◆ Environment" in out
