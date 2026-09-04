"""Tests for `hermes auth add openai-codex` importing an existing Codex CLI session.

Regression coverage for the device-code-only trap: when the OpenAI device-code
flow is rate-limited (429), `auth add` should offer to import the credentials the
Codex CLI already holds at ``~/.codex/auth.json`` — the same option `hermes model`
provides — instead of forcing another device-code login.
"""

from types import SimpleNamespace

import hermes_cli.auth as auth_mod
from agent.credential_pool import load_pool
from hermes_cli.auth_commands import auth_add_command


CLI_TOKENS = {
    "id_token": "id-xyz",
    "access_token": "cli-access-token",
    "refresh_token": "cli-refresh-token",
    "account_id": "acct-123",
}


def _args():
    return SimpleNamespace(provider="openai-codex", auth_type="", label=None, api_key=None)


class _Tty:
    def isatty(self):
        return True


def _boom(*a, **k):
    raise AssertionError("_codex_device_code_login must not run when import is chosen")


def test_import_accepted_skips_device_code(monkeypatch):
    monkeypatch.setattr(auth_mod, "_import_codex_cli_tokens", lambda: dict(CLI_TOKENS))
    monkeypatch.setattr(auth_mod, "_codex_device_code_login", _boom)
    monkeypatch.setattr("hermes_cli.auth_commands.sys.stdin", _Tty())
    monkeypatch.setattr("builtins.input", lambda *a, **k: "y")

    auth_add_command(_args())

    entries = load_pool("openai-codex").entries()
    assert len(entries) == 1
    assert entries[0].access_token == "cli-access-token"
    assert entries[0].refresh_token == "cli-refresh-token"


def test_import_declined_falls_back_to_device_code(monkeypatch):
    device_creds = {
        "tokens": {"access_token": "device-access", "refresh_token": "device-refresh"},
        "base_url": auth_mod.DEFAULT_CODEX_BASE_URL,
        "last_refresh": None,
    }
    monkeypatch.setattr(auth_mod, "_import_codex_cli_tokens", lambda: dict(CLI_TOKENS))
    monkeypatch.setattr(auth_mod, "_codex_device_code_login", lambda *a, **k: device_creds)
    monkeypatch.setattr("hermes_cli.auth_commands.sys.stdin", _Tty())
    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")

    auth_add_command(_args())

    entries = load_pool("openai-codex").entries()
    assert len(entries) == 1
    assert entries[0].access_token == "device-access"


def test_non_interactive_does_not_prompt(monkeypatch):
    """No TTY (piped/scripted) must not block on input(); falls back to device code."""
    device_creds = {
        "tokens": {"access_token": "device-access", "refresh_token": "device-refresh"},
        "base_url": auth_mod.DEFAULT_CODEX_BASE_URL,
        "last_refresh": None,
    }

    class _NoTty:
        def isatty(self):
            return False

    def _no_input(*a, **k):
        raise AssertionError("input() must not be called without a TTY")

    monkeypatch.setattr(auth_mod, "_import_codex_cli_tokens", lambda: dict(CLI_TOKENS))
    monkeypatch.setattr(auth_mod, "_codex_device_code_login", lambda *a, **k: device_creds)
    monkeypatch.setattr("hermes_cli.auth_commands.sys.stdin", _NoTty())
    monkeypatch.setattr("builtins.input", _no_input)

    auth_add_command(_args())

    entries = load_pool("openai-codex").entries()
    assert len(entries) == 1
    assert entries[0].access_token == "device-access"
