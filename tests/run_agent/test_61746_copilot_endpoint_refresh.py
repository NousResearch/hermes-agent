"""
Regression test for issue #61746 - Copilot credential rotation reuses
stale public endpoint for Enterprise accounts.

The bug: when credential-pool recovery rotates to a Copilot entry whose
persisted base_url is the public default (api.githubcopilot.com) but the
account's token-exchange advertises an Enterprise endpoint
(api.enterprise.githubcopilot.com), the rotation trusts the stale
base_url. The request is sent to the wrong host, gets 403, and the
otherwise-valid credential is marked exhausted.

The fix: in _swap_credential, when the provider is copilot, perform a
live token exchange via get_copilot_api_token() to refresh the base_url
from a single token-exchange result before applying the swap.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_agent(provider="copilot", model="gpt-5.6-sol"):
    """Build a bare AIAgent suitable for exercising _swap_credential.

    ``object.__new__`` skips __init__ so we don't need to wire the
    full runtime. We attach the attributes _swap_credential reads.
    """
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    agent.api_mode = "chat_completions"
    agent.base_url = "https://api.enterprise.githubcopilot.com"
    agent.api_key = "OLD_KEY"
    agent._client_kwargs = {}
    agent._apply_client_headers_for_base_url = MagicMock()
    agent._replace_primary_openai_client = MagicMock()
    return agent


def _make_entry(*, raw_token="RAW_GH_TOKEN", stale_base="https://api.githubcopilot.com"):
    """Build a stale pool entry whose persisted base_url is the public
    default (what the PR review identified as the bug class).
    """
    entry = MagicMock()
    entry.runtime_api_key = raw_token
    entry.runtime_base_url = None
    entry.access_token = raw_token
    entry.base_url = stale_base
    entry.raw_token = raw_token
    return entry


class TestSwapCredentialCopilotExchange:
    """#61746: when swapping to a Copilot entry, perform a live token
    exchange so the base_url reflects the account's current authoritative
    endpoint, not a stale persisted default.
    """

    def test_copilot_swap_refreshes_base_url_via_exchange(self):
        """_swap_credential must call get_copilot_api_token and adopt
        the exchange's base_url, not trust entry.base_url.
        """
        agent = _make_agent()
        entry = _make_entry(stale_base="https://api.githubcopilot.com")

        exchange_result = {
            "token": "FRESH_TOKEN",
            "base_url": "https://api.enterprise.githubcopilot.com",
        }

        with patch("hermes_cli.copilot_auth.get_copilot_api_token", return_value=exchange_result):
            agent._swap_credential(entry)

        # The Enterprise base_url from the live exchange wins
        assert agent.base_url == "https://api.enterprise.githubcopilot.com", (
            f"agent.base_url must reflect the live exchange, got {agent.base_url!r}"
        )
        # The fresh token from the live exchange replaces the old
        assert agent.api_key == "FRESH_TOKEN", (
            f"agent.api_key must reflect the live exchange, got {agent.api_key!r}"
        )

    def test_copilot_swap_falls_back_to_entry_base_url_on_exchange_failure(self):
        """If the live exchange raises, _swap_credential must fall back
        to the entry's persisted base_url (degraded but not crashed).
        """
        agent = _make_agent()
        entry = _make_entry(stale_base="https://api.githubcopilot.com")

        with patch(
            "hermes_cli.copilot_auth.get_copilot_api_token",
            side_effect=Exception("exchange failed"),
        ):
            agent._swap_credential(entry)

        # Falls back to the entry's persisted base_url (stale, but better than crashing)
        assert agent.base_url == "https://api.githubcopilot.com", (
            f"on exchange failure, must fall back to entry.base_url, got {agent.base_url!r}"
        )
        # The persisted access_token is the best we have without a fresh exchange
        assert agent.api_key == "RAW_GH_TOKEN", (
            f"on exchange failure, must use entry.access_token, got {agent.api_key!r}"
        )

    def test_non_copilot_provider_skips_exchange(self):
        """For non-copilot providers, _swap_credential must NOT call
        get_copilot_api_token (the exchange is provider-specific).
        """
        agent = _make_agent(provider="anthropic", model="claude-opus-4-5")
        entry = MagicMock()
        entry.runtime_api_key = "AK"
        entry.runtime_base_url = "https://api.anthropic.com"
        entry.access_token = "AK"
        entry.base_url = "https://api.anthropic.com"

        with patch(
            "hermes_cli.copilot_auth.get_copilot_api_token"
        ) as mock_exchange:
            agent._swap_credential(entry)

        mock_exchange.assert_not_called()
        assert agent.base_url == "https://api.anthropic.com"
