"""Behavioral tests for hermes_cli.bang_shell — the !<command> shell-mode helpers.

These cover the pure parser and gate functions; subprocess execution is not
tested here (that needs a live shell and is integration-level).
"""

import os

import pytest

from hermes_cli.bang_shell import (
    DEFAULT_TIMEOUT,
    USAGE_HINT,
    bang_shell_enabled,
    is_bang_command,
    parse_bang_command,
)


# ── is_bang_command ───────────────────────────────────────────────────────────

class TestIsBangCommand:
    def test_plain_bang_command(self):
        assert is_bang_command("!git status") is True

    def test_bang_with_leading_spaces(self):
        assert is_bang_command("  !ls -la") is True

    def test_bang_alone(self):
        assert is_bang_command("!") is True

    def test_bang_mid_text_is_not_bang_command(self):
        assert is_bang_command("fix the bug!") is False

    def test_plain_text(self):
        assert is_bang_command("hello world") is False

    def test_empty_string(self):
        assert is_bang_command("") is False

    def test_none_returns_false(self):
        assert is_bang_command(None) is False  # type: ignore[arg-type]

    def test_non_string_returns_false(self):
        assert is_bang_command(42) is False  # type: ignore[arg-type]

    def test_double_bang_is_bang_command(self):
        # !! is valid — the second ! becomes part of the shell command
        assert is_bang_command("!!") is True


# ── parse_bang_command ────────────────────────────────────────────────────────

class TestParseBangCommand:
    def test_strips_leading_bang(self):
        assert parse_bang_command("!git status") == "git status"

    def test_strips_spaces_after_bang(self):
        assert parse_bang_command("!  ls -la") == "ls -la"

    def test_bare_bang_returns_empty(self):
        assert parse_bang_command("!") == ""

    def test_double_bang_preserves_second(self):
        # !! → the literal "!" as the command (history expansion etc.)
        assert parse_bang_command("!!") == "!"

    def test_leading_spaces_before_bang_stripped(self):
        assert parse_bang_command("  !echo hi") == "echo hi"

    def test_non_bang_returns_empty(self):
        assert parse_bang_command("echo hi") == ""

    def test_none_returns_empty(self):
        assert parse_bang_command(None) == ""  # type: ignore[arg-type]

    def test_multiword_command_preserved(self):
        assert parse_bang_command("!docker ps -a --format json") == "docker ps -a --format json"


# ── bang_shell_enabled ────────────────────────────────────────────────────────

class TestBangShellEnabled:
    def test_enabled_in_plain_local_session(self, monkeypatch):
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        assert bang_shell_enabled() is True

    def test_disabled_in_gateway_session(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        assert bang_shell_enabled() is False

    def test_disabled_in_cron_session(self, monkeypatch):
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        assert bang_shell_enabled() is False

    def test_disabled_when_session_platform_set(self, monkeypatch):
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        assert bang_shell_enabled() is False

    def test_enabled_when_session_platform_empty_string(self, monkeypatch):
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "")
        assert bang_shell_enabled() is True


# ── constants sanity ──────────────────────────────────────────────────────────

def test_default_timeout_is_positive():
    assert DEFAULT_TIMEOUT > 0


def test_usage_hint_mentions_bang():
    assert "!" in USAGE_HINT
