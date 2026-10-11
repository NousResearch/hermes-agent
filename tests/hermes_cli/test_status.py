from types import SimpleNamespace

from hermes_cli.status import show_status
import subprocess

import pytest


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
    monkeypatch.setattr(auth_mod, "get_nous_auth_status_local", dict, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", dict, raising=False)
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", dict, raising=False)
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
    monkeypatch.setattr(auth_mod, "get_nous_auth_status", dict, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", dict, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", dict, raising=False)
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", dict, raising=False)
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
    monkeypatch.setattr(auth_mod, "get_nous_auth_status_local", dict, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", dict, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", dict, raising=False)
    monkeypatch.setattr(auth_mod, "get_minimax_oauth_auth_status", dict, raising=False)
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
    monkeypatch.setattr(auth_mod, "get_nous_auth_status", dict, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", dict, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", dict, raising=False)
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", dict, raising=False)
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


@pytest.mark.parametrize("invalid", ["last_activity_at", "nan", "inf", "-inf", "8.4e252"])
@pytest.mark.parametrize("include_valid", [False, True])
def test_show_status_ignores_legacy_non_numeric_gateway_last_activity(
    monkeypatch, capsys, tmp_path, invalid, include_valid
):
    """A malformed legacy state.db timestamp must not crash ``hermes status``."""
    from hermes_cli import status as status_mod
    import hermes_state
    import time

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class _FakeDB:
        def __init__(self, **_kwargs):
            pass

        def list_gateway_sessions(self, active_only=True):
            rows = [{"id": "legacy", "last_active": invalid}]
            if include_valid:
                rows.extend([
                    {"id": "older", "last_active": time.time() - 7200},
                    {"id": "latest", "last_active": str(time.time() - 90)},
                ])
            return rows

        def close(self):
            return None

    monkeypatch.setattr(hermes_state, "SessionDB", _FakeDB)

    status_mod.show_status(SimpleNamespace(full=True, deep=False))

    output = capsys.readouterr().out
    assert f"Active:       {3 if include_valid else 1} session(s)" in output
    if include_valid:
        assert "Last activity:" in output
        assert "1m ago" in output
    else:
        assert "Last activity:" not in output


def _status_args(*argv):
    import argparse
    from hermes_cli.subcommands.status import build_status_parser

    parser = argparse.ArgumentParser()
    build_status_parser(parser.add_subparsers(), cmd_status=None)
    return parser.parse_args(["status", *argv])


def test_status_defaults_to_summary_and_full_or_all_prints_every_section(monkeypatch, capsys, tmp_path):
    from hermes_cli import model_switch

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
