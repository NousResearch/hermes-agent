"""Unit tests for ``agent.credential_resolver``.

The resolver lets users put references like ``env:VAR`` or
``keychain:service/account`` in config fields where Hermes previously
expected a literal API key. These tests cover the four supported forms
plus the failure modes.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agent.credential_resolver import (
    CredentialResolveError,
    resolve_credential_string,
)


# ─── plain pass-through ──────────────────────────────────────────────────


def test_literal_value_returned_unchanged():
    assert resolve_credential_string("sk-abc123") == "sk-abc123"


def test_value_is_stripped():
    assert resolve_credential_string("  sk-abc123  ") == "sk-abc123"


def test_empty_string_returns_empty():
    assert resolve_credential_string("") == ""
    assert resolve_credential_string("   ") == ""


def test_non_string_returned_unchanged():
    assert resolve_credential_string(None) is None  # type: ignore[arg-type]
    assert resolve_credential_string(123) == 123  # type: ignore[arg-type]


def test_unknown_prefix_treated_as_literal():
    # Anything not matching env:/keychain:/secret-tool: passes through.
    assert resolve_credential_string("https://example/key") == "https://example/key"


# ─── env: schema ─────────────────────────────────────────────────────────


def test_env_schema_resolves_from_environment(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_KEY", "value-from-env")
    assert resolve_credential_string("env:HERMES_TEST_KEY") == "value-from-env"


def test_env_schema_missing_var_raises(monkeypatch):
    monkeypatch.delenv("HERMES_TEST_KEY_MISSING", raising=False)
    with pytest.raises(CredentialResolveError, match="HERMES_TEST_KEY_MISSING"):
        resolve_credential_string("env:HERMES_TEST_KEY_MISSING")


def test_env_schema_empty_var_raises(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_KEY_EMPTY", "")
    with pytest.raises(CredentialResolveError, match="empty"):
        resolve_credential_string("env:HERMES_TEST_KEY_EMPTY")


def test_env_schema_empty_target_raises():
    with pytest.raises(CredentialResolveError, match="Empty reference"):
        resolve_credential_string("env:")


# ─── keychain: schema ────────────────────────────────────────────────────


def _make_subprocess_result(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


def test_keychain_resolves_via_security_cli():
    """Explicit account form 'service@account'."""
    with patch("agent.credential_resolver.shutil.which", return_value="/usr/bin/security"), \
         patch("agent.credential_resolver.subprocess.run") as run:
        run.return_value = _make_subprocess_result(0, stdout="secret-value\n")
        assert resolve_credential_string("keychain:my-service@my-account") == "secret-value"
        args = run.call_args.args[0]
        assert args[0:3] == ["security", "find-generic-password", "-w"]
        assert "-a" in args and args[args.index("-a") + 1] == "my-account"
        assert "-s" in args and args[args.index("-s") + 1] == "my-service"


def test_keychain_default_account_is_current_user():
    """Without '@account', current user is used (mirrors common security-cli usage)."""
    with patch("agent.credential_resolver.shutil.which", return_value="/usr/bin/security"), \
         patch("agent.credential_resolver.subprocess.run") as run, \
         patch("agent.credential_resolver.getpass.getuser", return_value="testuser"):
        run.return_value = _make_subprocess_result(0, stdout="secret-value\n")
        assert resolve_credential_string("keychain:ai-system/litellm-master-key") == "secret-value"
        args = run.call_args.args[0]
        # Service preserves slashes
        assert args[args.index("-s") + 1] == "ai-system/litellm-master-key"
        # Account defaults to current user
        assert args[args.index("-a") + 1] == "testuser"


def test_keychain_missing_security_cli_raises():
    with patch("agent.credential_resolver.shutil.which", return_value=None):
        with pytest.raises(CredentialResolveError, match="security"):
            resolve_credential_string("keychain:foo/bar")


def test_keychain_item_not_found_raises():
    with patch("agent.credential_resolver.shutil.which", return_value="/usr/bin/security"), \
         patch("agent.credential_resolver.subprocess.run") as run:
        run.return_value = _make_subprocess_result(44, stderr="not found")
        with pytest.raises(CredentialResolveError, match="not found"):
            resolve_credential_string("keychain:foo/bar")


def test_keychain_empty_value_raises():
    with patch("agent.credential_resolver.shutil.which", return_value="/usr/bin/security"), \
         patch("agent.credential_resolver.subprocess.run") as run:
        run.return_value = _make_subprocess_result(0, stdout="\n")
        with pytest.raises(CredentialResolveError, match="empty"):
            resolve_credential_string("keychain:foo/bar")


def test_keychain_timeout_raises():
    with patch("agent.credential_resolver.shutil.which", return_value="/usr/bin/security"), \
         patch(
             "agent.credential_resolver.subprocess.run",
             side_effect=subprocess.TimeoutExpired(cmd=["security"], timeout=5),
         ):
        with pytest.raises(CredentialResolveError, match="timed out"):
            resolve_credential_string("keychain:foo/bar")


def test_keychain_malformed_target_raises():
    # Empty account explicitly given via '@'
    with pytest.raises(CredentialResolveError, match="empty account"):
        resolve_credential_string("keychain:svc@")
    # Empty service before '@'
    with pytest.raises(CredentialResolveError, match="empty service"):
        resolve_credential_string("keychain:@account-only")


# ─── secret-tool: schema ─────────────────────────────────────────────────


def test_secret_tool_resolves_via_cli():
    with patch("agent.credential_resolver.shutil.which", return_value="/usr/bin/secret-tool"), \
         patch("agent.credential_resolver.subprocess.run") as run:
        run.return_value = _make_subprocess_result(0, stdout="linux-secret\n")
        assert resolve_credential_string("secret-tool:my-svc@my-acc") == "linux-secret"
        args = run.call_args.args[0]
        assert args[0:2] == ["secret-tool", "lookup"]
        assert "service" in args and args[args.index("service") + 1] == "my-svc"
        assert "account" in args and args[args.index("account") + 1] == "my-acc"


def test_secret_tool_missing_cli_raises():
    with patch("agent.credential_resolver.shutil.which", return_value=None):
        with pytest.raises(CredentialResolveError, match="secret-tool"):
            resolve_credential_string("secret-tool:foo/bar")


def test_secret_tool_not_found_raises():
    with patch("agent.credential_resolver.shutil.which", return_value="/usr/bin/secret-tool"), \
         patch("agent.credential_resolver.subprocess.run") as run:
        run.return_value = _make_subprocess_result(1, stdout="", stderr="No matching items")
        with pytest.raises(CredentialResolveError, match="not found"):
            resolve_credential_string("secret-tool:foo/bar")
