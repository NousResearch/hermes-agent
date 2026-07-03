import shutil
import types

import pytest

import agent.anthropic_adapter as aa


@pytest.fixture(autouse=True)
def _clean_anthropic_refresh_state(monkeypatch):
    for key in (
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "HERMES_ANTHROPIC_DISABLE_CLAUDE_CLI_REFRESH",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(aa, "_claude_code_cli_refresh_last_attempt", 0.0)


def test_resolve_anthropic_token_uses_official_claude_cli_refresh(monkeypatch):
    """If Hermes cannot resolve creds directly, run Claude CLI once.

    This protects Fable/Anthropic subprocesses from failing with
    "No Anthropic credentials found" when the official Claude Code session can
    refresh its store.
    """
    calls = {"read": 0, "run": 0}

    def fake_read_creds():
        calls["read"] += 1
        if calls["run"] == 0:
            return None
        return {
            "accessToken": "fresh-cli-token",
            "refreshToken": "fresh-refresh-token",
            "expiresAt": 9999999999999,
            "source": "claude_code_credentials_file",
        }

    def fake_run(*args, **kwargs):
        calls["run"] += 1
        return types.SimpleNamespace(returncode=0, stdout="OK_CLAUDE_REFRESH\n", stderr="")

    monkeypatch.setattr(aa, "read_claude_code_credentials", fake_read_creds)
    monkeypatch.setattr(
        aa,
        "is_claude_code_token_valid",
        lambda creds: bool(creds and creds.get("accessToken") == "fresh-cli-token"),
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(aa.subprocess, "run", fake_run)
    monkeypatch.setattr(aa, "_resolve_anthropic_pool_token", lambda: None)

    assert aa.resolve_anthropic_token() == "fresh-cli-token"
    assert calls["run"] == 1


def test_expired_claude_code_creds_with_failed_pure_refresh_use_cli(monkeypatch):
    calls = {"run": 0}

    expired = {
        "accessToken": "expired-token",
        "refreshToken": "stale-refresh",
        "expiresAt": 1,
        "source": "macos_keychain",
    }
    fresh = {
        "accessToken": "fresh-after-cli",
        "refreshToken": "fresh-refresh",
        "expiresAt": 9999999999999,
        "source": "macos_keychain",
    }

    def fake_read_creds():
        return fresh if calls["run"] else expired

    def fake_run(*args, **kwargs):
        calls["run"] += 1
        return types.SimpleNamespace(returncode=0, stdout="OK_CLAUDE_REFRESH\n", stderr="")

    monkeypatch.setattr(aa, "read_claude_code_credentials", fake_read_creds)
    monkeypatch.setattr(aa, "is_claude_code_token_valid", lambda creds: creds.get("accessToken") == "fresh-after-cli")
    monkeypatch.setattr(aa, "_refresh_oauth_token", lambda creds: None)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(aa.subprocess, "run", fake_run)
    monkeypatch.setattr(aa, "_resolve_anthropic_pool_token", lambda: None)

    assert aa.resolve_anthropic_token() == "fresh-after-cli"
    assert calls["run"] == 1


def test_claude_cli_nonzero_falls_through_to_pool(monkeypatch):
    calls = {"run": 0}

    monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: None)
    monkeypatch.setattr(aa, "is_claude_code_token_valid", lambda creds: False)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/claude" if name == "claude" else None)

    def fake_run(*args, **kwargs):
        calls["run"] += 1
        return types.SimpleNamespace(returncode=1, stdout="", stderr="login required")

    monkeypatch.setattr(aa.subprocess, "run", fake_run)
    monkeypatch.setattr(aa, "_resolve_anthropic_pool_token", lambda: "pool-token")

    assert aa.resolve_anthropic_token() == "pool-token"
    assert calls["run"] == 1


def test_cli_refresh_disabled_by_env(monkeypatch):
    calls = {"run": 0}
    monkeypatch.setenv("HERMES_ANTHROPIC_DISABLE_CLAUDE_CLI_REFRESH", "1")
    monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: None)
    monkeypatch.setattr(aa, "is_claude_code_token_valid", lambda creds: False)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(aa.subprocess, "run", lambda *a, **k: calls.__setitem__("run", calls["run"] + 1))
    monkeypatch.setattr(aa, "_resolve_anthropic_pool_token", lambda: "pool-token")

    assert aa.resolve_anthropic_token() == "pool-token"
    assert calls["run"] == 0


def test_cli_refresh_cooldown_prevents_repeated_subprocesses(monkeypatch):
    calls = {"run": 0}
    monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: None)
    monkeypatch.setattr(aa, "is_claude_code_token_valid", lambda creds: False)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/claude" if name == "claude" else None)

    def fake_run(*args, **kwargs):
        calls["run"] += 1
        return types.SimpleNamespace(returncode=1, stdout="", stderr="login required")

    monkeypatch.setattr(aa.subprocess, "run", fake_run)
    monkeypatch.setattr(aa, "_resolve_anthropic_pool_token", lambda: "pool-token")

    assert aa.resolve_anthropic_token() == "pool-token"
    assert aa.resolve_anthropic_token() == "pool-token"
    assert calls["run"] == 1


def test_env_tokens_do_not_trigger_cli_refresh(monkeypatch):
    calls = {"run": 0}
    monkeypatch.setenv("ANTHROPIC_TOKEN", "env-token")
    monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: None)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(aa.subprocess, "run", lambda *a, **k: calls.__setitem__("run", calls["run"] + 1))

    assert aa.resolve_anthropic_token() == "env-token"
    assert calls["run"] == 0


def test_resolve_anthropic_token_falls_through_when_claude_cli_unavailable(monkeypatch):
    monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: None)
    monkeypatch.setattr(aa, "is_claude_code_token_valid", lambda creds: False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(aa, "_resolve_anthropic_pool_token", lambda: "pool-token")

    assert aa.resolve_anthropic_token() == "pool-token"
