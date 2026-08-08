"""Regression tests: the macOS Keychain lookup must disambiguate by account.

macOS permits several generic-password items to share one service name, and
Claude Code does exactly that. Alongside the login credential it stores its
MCP-server OAuth state under the same ``Claude Code-credentials`` service, as a
separate item whose account is ``unknown`` and whose payload contains only an
``mcpOAuth`` key — no ``claudeAiOauth`` at all.

``security find-generic-password`` returns the FIRST matching item. An unscoped
lookup therefore often wins the MCP item, parses fine, finds no
``claudeAiOauth``, and yields None — so Hermes reports "No Anthropic credentials
found" while the user is fully logged in to Claude Code. Nothing errors; the
lookup succeeds and returns the wrong item.

These tests pin the two properties that prevent that:

  1. the account-scoped read is attempted, and
  2. an item that parses but lacks ``claudeAiOauth`` does not abort the search.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from agent.anthropic_adapter import (
    _is_oauth_token,
    _read_claude_code_credentials_from_keychain,
    resolve_anthropic_token,
)

# Exercises the reader with explicit platform/subprocess mocks; never touches a
# real Keychain, so it opts out of the suite-wide guard like its sibling module.
pytestmark = pytest.mark.allow_macos_keychain


MCP_ONLY_PAYLOAD = json.dumps(
    {"mcpOAuth": {"posthog|abc123": {"serverName": "posthog", "accessToken": "mcp-tok"}}}
)
LOGIN_PAYLOAD = json.dumps(
    {
        "mcpOAuth": {"posthog|abc123": {"serverName": "posthog"}},
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat01-real-login-token",
            "refreshToken": "sk-ant-ort01-refresh",
            "expiresAt": 4102444800000,
            "scopes": ["user:inference", "user:profile"],
            "subscriptionType": "team",
        },
    }
)


def _result(stdout: str, code: int = 0) -> MagicMock:
    r = MagicMock()
    r.returncode = code
    r.stdout = stdout
    return r


class TestKeychainAccountScoping:
    def test_account_scoped_lookup_is_attempted(self):
        """The reader must pass ``-a <username>``, not only ``-s <service>``."""
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("getpass.getuser", return_value="alice"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   return_value=_result(LOGIN_PAYLOAD)) as run:
            creds = _read_claude_code_credentials_from_keychain()

        assert creds is not None
        assert creds["accessToken"] == "sk-ant-oat01-real-login-token"

        argv = run.call_args_list[0][0][0]
        assert "-a" in argv, f"account flag missing — got {argv}"
        assert argv[argv.index("-a") + 1] == "alice"
        assert "-s" in argv and argv[argv.index("-s") + 1] == "Claude Code-credentials"

    def test_mcp_only_first_item_does_not_shadow_the_login_credential(self):
        """THE BUG: an mcpOAuth-only item must not end the search.

        Models the real machine: the account-scoped read hits an item without
        ``claudeAiOauth``, and the unscoped fallback returns the login payload.
        A reader that gave up on the first parse-without-claudeAiOauth returns
        None here and reports "no credentials" to a logged-in user.
        """
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("getpass.getuser", return_value="alice"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   side_effect=[_result(MCP_ONLY_PAYLOAD), _result(LOGIN_PAYLOAD)]) as run:
            creds = _read_claude_code_credentials_from_keychain()

        assert run.call_count == 2, "must keep probing after an mcpOAuth-only item"
        assert creds is not None, "login credential was shadowed by the MCP item"
        assert creds["accessToken"] == "sk-ant-oat01-real-login-token"
        assert creds["source"] == "macos_keychain"

    def test_unscoped_fallback_preserved_when_username_unavailable(self):
        """If the username cannot be determined, the historical read still runs."""
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("getpass.getuser", side_effect=OSError("no login name")), \
             patch("agent.anthropic_adapter.subprocess.run",
                   return_value=_result(LOGIN_PAYLOAD)) as run:
            creds = _read_claude_code_credentials_from_keychain()

        assert creds is not None
        argv = run.call_args_list[0][0][0]
        assert "-a" not in argv, "must not pass an empty account"

    def test_returns_none_when_no_item_has_claude_oauth(self):
        """All candidates lacking claudeAiOauth still resolves to None."""
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("getpass.getuser", return_value="alice"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   return_value=_result(MCP_ONLY_PAYLOAD)):
            assert _read_claude_code_credentials_from_keychain() is None

    def test_nonzero_exit_on_scoped_read_falls_through_to_unscoped(self):
        """No item for this account is not a failure — try the unscoped read."""
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("getpass.getuser", return_value="alice"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   side_effect=[_result("", code=44), _result(LOGIN_PAYLOAD)]):
            creds = _read_claude_code_credentials_from_keychain()

        assert creds is not None
        assert creds["accessToken"] == "sk-ant-oat01-real-login-token"

    def test_malformed_json_does_not_abort_remaining_candidates(self):
        """A corrupt item must not mask a good one behind it."""
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("getpass.getuser", return_value="alice"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   side_effect=[_result("not json at all"), _result(LOGIN_PAYLOAD)]):
            creds = _read_claude_code_credentials_from_keychain()

        assert creds is not None
        assert creds["accessToken"] == "sk-ant-oat01-real-login-token"

    def test_non_darwin_short_circuits_without_subprocess(self):
        """Linux/Windows must not shell out to ``security`` at all."""
        with patch("agent.anthropic_adapter.platform.system", return_value="Linux"), \
             patch("agent.anthropic_adapter.subprocess.run") as run:
            assert _read_claude_code_credentials_from_keychain() is None
        run.assert_not_called()


# A syntactically valid Console key shape. Never sent anywhere — these tests only
# observe how the resolver ranks it and how the adapter classifies it.
CONSOLE_API_KEY = "sk-ant-api03-" + ("A" * 24)

KEYCHAIN_CREDS = {
    "accessToken": "sk-ant-oat01-real-login-token",
    "refreshToken": "sk-ant-ort01-refresh",
    "expiresAt": 4102444800000,  # year 2100 — never treated as expired
    "source": "macos_keychain",
}


class TestBillingLaneInvariant:
    """Why the lookup bug matters: it silently changes which account is billed.

    ``resolve_anthropic_token`` ranks the Claude Code credential (source 3) above
    ``ANTHROPIC_API_KEY`` (source 5). A subscription OAuth token ships as
    ``Authorization: Bearer`` and draws down the subscription plan allowance; a
    Console key ships as ``x-api-key`` and draws down that organisation's prepaid
    API credits. The two are different money.

    So a keychain read that wrongly returns None does not merely fail — when
    ``ANTHROPIC_API_KEY`` is also present it falls through to source 5 and
    reroutes the billing lane, with no warning at any log level. These tests pin
    the ranking that keeps subscription traffic on the subscription.
    """

    @staticmethod
    def _resolve(keychain_creds):
        """Resolve with sources 1, 2 and 4 silenced, so only 3 and 5 are live."""
        env = {
            "ANTHROPIC_TOKEN": "",
            "CLAUDE_CODE_OAUTH_TOKEN": "",
            "ANTHROPIC_API_KEY": CONSOLE_API_KEY,
        }
        with patch.dict(os.environ, env), \
             patch("agent.anthropic_adapter._read_claude_code_credentials_from_keychain",
                   return_value=keychain_creds), \
             patch("agent.anthropic_adapter._read_claude_code_credentials_from_file",
                   return_value=None), \
             patch("agent.anthropic_adapter._resolve_anthropic_pool_token",
                   return_value=None):
            return resolve_anthropic_token()

    def test_keychain_oauth_outranks_env_api_key(self):
        """With both present, the subscription credential must win."""
        token = self._resolve(KEYCHAIN_CREDS)

        assert token == "sk-ant-oat01-real-login-token", (
            "ANTHROPIC_API_KEY outranked the Claude Code login credential — "
            "subscription traffic would be billed to prepaid API credits"
        )
        assert _is_oauth_token(token), "resolved token must ship as Bearer, not x-api-key"

    def test_keychain_miss_falls_through_to_env_api_key(self):
        """Characterises the blast radius: a miss reroutes the billing lane.

        Not desired behaviour — documented so the ranking above is understood as
        load-bearing rather than incidental. Before the account-scoped read, an
        mcpOAuth-only item was enough to land here.
        """
        token = self._resolve(None)

        assert token == CONSOLE_API_KEY
        assert not _is_oauth_token(token), (
            "a Console API key must never be classified as OAuth"
        )
