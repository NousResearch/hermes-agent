"""Regression tests: resolve_anthropic_token() must honour the profile secret scope.

BUG LOCATION
    agent/anthropic_adapter.py lines 1218, 1226, 1245

ROOT CAUSE
    resolve_anthropic_token() calls os.getenv() directly at three call sites,
    bypassing get_secret() from agent/secret_scope.py:

        line 1218: os.getenv("ANTHROPIC_TOKEN", "")
        line 1226: os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "")
        line 1245: os.getenv("ANTHROPIC_API_KEY", "")

    Every other credential reader in the resolution chain (hermes_cli/runtime_provider.py
    _getenv, secret_scope.get_secret) correctly uses the profile-scoped wrapper.

IMPACT
    In a multiplexed gateway (set_multiplex_active(True)), any Anthropic API call
    reads the process-level os.environ instead of the active profile's secret scope.
    This causes:
      - Cross-profile credential leakage (profile A's key used for profile B's turn)
      - Cron jobs in multiplex mode silently reading the wrong key
      - No fail-closed signal (UnscopedSecretError) when no scope is installed

STATUS
    All tests in this file are RED (failing) while the bug exists.
    They turn GREEN when os.getenv() at the three sites is replaced with
    get_secret() from agent.secret_scope (matching the pattern in runtime_provider.py).
"""

import pytest
from unittest.mock import patch

from agent import secret_scope as ss
from agent.anthropic_adapter import resolve_anthropic_token


@pytest.fixture(autouse=True)
def _reset_multiplex():
    """Isolate global multiplex flag between tests."""
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


@pytest.fixture(autouse=True)
def _pin_file_and_pool_sources():
    """Pin credential-file (source 3) and pool (source 4) to None.

    This isolates the three os.getenv() call sites (sources 1, 2, 5) so each
    test exercises exactly the env-var reading behaviour under scope control.
    """
    with patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None), \
         patch("agent.anthropic_adapter._resolve_anthropic_pool_token", return_value=None):
        yield


# ---------------------------------------------------------------------------
# Core isolation: scoped key must beat os.environ in multiplex mode
# ---------------------------------------------------------------------------

class TestApiKeyScopeIsolation:
    """ANTHROPIC_API_KEY (source 5, line 1245) must read from scope, not os.environ."""

    def test_scoped_api_key_used_over_environ(self, monkeypatch):
        """Profile scope's ANTHROPIC_API_KEY wins over os.environ value.

        Bug: os.getenv("ANTHROPIC_API_KEY") at line 1245 reads the wrong profile's
        key when a profile scope with a different value is active.
        Fix: replace with get_secret("ANTHROPIC_API_KEY").
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-WRONG-OTHER-PROFILE")
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-ant-api-CORRECT-PROFILE"})
        try:
            result = resolve_anthropic_token()
        finally:
            ss.reset_secret_scope(tok)

        assert result == "sk-ant-api-CORRECT-PROFILE", (
            f"resolve_anthropic_token() returned {result!r} (from os.environ) "
            f"instead of 'sk-ant-api-CORRECT-PROFILE' (from profile scope). "
            f"Bug confirmed at anthropic_adapter.py:1245 — os.getenv bypasses get_secret()."
        )

    def test_two_profiles_return_different_keys(self, monkeypatch):
        """Each profile scope must resolve to its own ANTHROPIC_API_KEY.

        This is the canonical credential-isolation invariant.
        With the bug, both calls return the same os.environ value.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-SHARED-ENVIRON-WRONG")
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

        ss.set_multiplex_active(True)

        tok_a = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-ant-api-PROFILE-A"})
        try:
            result_a = resolve_anthropic_token()
        finally:
            ss.reset_secret_scope(tok_a)

        tok_b = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-ant-api-PROFILE-B"})
        try:
            result_b = resolve_anthropic_token()
        finally:
            ss.reset_secret_scope(tok_b)

        assert result_a == "sk-ant-api-PROFILE-A", (
            f"Profile A got {result_a!r} — expected its own key."
        )
        assert result_b == "sk-ant-api-PROFILE-B", (
            f"Profile B got {result_b!r} — expected its own key."
        )
        assert result_a != result_b, (
            f"Both profiles resolved to {result_a!r}. "
            f"Credential isolation is broken — both read from os.environ."
        )


# ---------------------------------------------------------------------------
# OAuth token leakage: ANTHROPIC_TOKEN (source 1, line 1218)
# ---------------------------------------------------------------------------

class TestOAuthTokenLeakageFromEnviron:
    """ANTHROPIC_TOKEN in os.environ must not override the active profile scope."""

    def test_anthropic_token_env_does_not_shadow_scoped_api_key(self, monkeypatch):
        """An ANTHROPIC_TOKEN present in os.environ from another profile must not be used.

        Scenario: Profile B's OAuth token was set in os.environ (e.g., by a previous
        process or a different profile's session). Profile A's scope only has
        ANTHROPIC_API_KEY. resolve_anthropic_token() must use Profile A's API key.

        Bug: os.getenv("ANTHROPIC_TOKEN") at line 1218 reads the leaked env var
        and returns Profile B's OAuth token for Profile A's Anthropic calls.
        Fix: replace with get_secret("ANTHROPIC_TOKEN"), which returns None when
        ANTHROPIC_TOKEN is absent from the active scope.
        """
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat-LEAKED-PROFILE-B")
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-ant-api-PROFILE-A"})
        try:
            result = resolve_anthropic_token()
        finally:
            ss.reset_secret_scope(tok)

        assert result != "sk-ant-oat-LEAKED-PROFILE-B", (
            f"Profile B's OAuth token leaked from os.environ[ANTHROPIC_TOKEN] "
            f"(anthropic_adapter.py:1218). Profile A received the wrong credential."
        )
        assert result == "sk-ant-api-PROFILE-A", (
            f"Expected Profile A's API key but got {result!r}."
        )

    def test_anthropic_token_in_scope_is_used(self, monkeypatch):
        """When ANTHROPIC_TOKEN IS in the active profile scope, it must be used.

        This verifies the positive case: scoped OAuth tokens work after the fix.
        """
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"ANTHROPIC_TOKEN": "sk-ant-oat-PROFILE-A-OWN-OAUTH"})
        try:
            result = resolve_anthropic_token()
        finally:
            ss.reset_secret_scope(tok)

        assert result == "sk-ant-oat-PROFILE-A-OWN-OAUTH", (
            f"Profile A's own OAuth token (in scope) was not returned; got {result!r}."
        )


# ---------------------------------------------------------------------------
# Claude Code OAuth token leakage: CLAUDE_CODE_OAUTH_TOKEN (source 2, line 1226)
# ---------------------------------------------------------------------------

class TestClaudeCodeOAuthTokenLeakage:
    """CLAUDE_CODE_OAUTH_TOKEN in os.environ must not override the active profile scope."""

    def test_cc_oauth_env_does_not_shadow_scoped_api_key(self, monkeypatch):
        """CLAUDE_CODE_OAUTH_TOKEN from os.environ must not be used when it's not in scope.

        Bug: os.getenv("CLAUDE_CODE_OAUTH_TOKEN") at line 1226 reads the global env
        var even when the active profile scope doesn't include it.
        Fix: replace with get_secret("CLAUDE_CODE_OAUTH_TOKEN").
        """
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-CC-LEAKED-ENVIRON")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-ant-api-PROFILE-X"})
        try:
            result = resolve_anthropic_token()
        finally:
            ss.reset_secret_scope(tok)

        assert result != "sk-ant-oat-CC-LEAKED-ENVIRON", (
            f"CLAUDE_CODE_OAUTH_TOKEN leaked from os.environ (anthropic_adapter.py:1226). "
            f"Profile X's Anthropic call used the wrong credential."
        )
        assert result == "sk-ant-api-PROFILE-X", (
            f"Expected Profile X's API key but got {result!r}."
        )


# ---------------------------------------------------------------------------
# Cron scheduler scenario: unscoped call in multiplex mode
# ---------------------------------------------------------------------------

class TestCronSchedulerUnscopedCall:
    """Cron scheduler in multiplex mode: resolve_anthropic_token() must fail closed.

    The cron scheduler (cron/scheduler.py) loads a credential_pool for the job's
    provider, but does NOT call set_secret_scope() before running Anthropic calls.
    In multiplex mode, this means resolve_anthropic_token() runs with:
      - _MULTIPLEX_ACTIVE = True
      - No active profile scope

    With os.getenv() (current code): silently reads the process-level os.environ,
    which may be empty or hold a different profile's value. No error is raised.

    With get_secret() (after fix): raises UnscopedSecretError, signalling that the
    cron scheduler must be updated to call set_secret_scope() for each job's profile.
    """

    def test_unscoped_call_in_multiplex_mode_fails_closed(self, monkeypatch):
        """An unscoped resolve_anthropic_token() in multiplex mode must raise UnscopedSecretError.

        Bug: os.getenv() at lines 1218/1226/1245 never raises — silently returning
        the process-level key, masking the missing scope in cron scheduler context.
        Fix: use get_secret(), which raises UnscopedSecretError when multiplex is
        active and no scope is installed (matching secret_scope.py's fail-closed contract).
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-PROCESS-LEVEL-SHOULD-NOT-LEAK")
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

        ss.set_multiplex_active(True)
        # No set_secret_scope() call — simulates cron scheduler context

        with pytest.raises(ss.UnscopedSecretError, match="ANTHROPIC"):
            resolve_anthropic_token()
