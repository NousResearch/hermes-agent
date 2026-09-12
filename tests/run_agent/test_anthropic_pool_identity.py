"""A pool-selected Anthropic account must survive pre-dispatch refresh.

``_try_refresh_anthropic_client_credentials`` runs the singleton
``resolve_anthropic_token()`` before every native dispatch
(``_create_request_anthropic_client``, ``_anthropic_messages_create``) and on
the 401 retry path (``_refresh_credentials_after_401``). When the agent is
bound to a credential-pool entry, that singleton may hold a *different* Claude
account: adopting it swaps the native client while ``api_key`` and
``_credential_pool_entry_id`` keep claiming the pool-selected entry, so later
429s/401s are attributed to the wrong account. Pool-owned credentials must
refresh/rotate through the pool (which already adopts pairs rotated
out-of-band via its singleton stores).
"""
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _pool_bound_agent() -> Any:
    """Only the real runtime seam; no agent startup or live credentials."""
    agent: Any = object.__new__(AIAgent)
    agent.api_mode = "anthropic_messages"
    agent.provider = "anthropic"
    agent.model = "claude-fable-5"
    agent.api_key = "test-account-two"
    agent._anthropic_api_key = "test-account-two"
    agent._anthropic_base_url = "https://api.anthropic.com"
    agent._credential_pool = SimpleNamespace()
    agent._credential_pool_entry_id = "account-two"
    agent._anthropic_client = MagicMock()
    return agent


def test_pool_selected_account_is_not_replaced_by_singleton_before_dispatch():
    agent = _pool_bound_agent()
    original_client = agent._anthropic_client

    with (
        patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="test-account-one") as singleton,
        patch("agent.anthropic_adapter.build_anthropic_client") as rebuild,
    ):
        refreshed = agent._try_refresh_anthropic_client_credentials()

    assert agent._anthropic_api_key == "test-account-two", "pre-dispatch refresh silently selected another account"
    assert agent.api_key == agent._anthropic_api_key, "bookkeeping key and dispatched key disagree"
    assert agent._anthropic_client is original_client
    assert refreshed is False
    original_client.close.assert_not_called()
    singleton.assert_not_called()
    rebuild.assert_not_called()


def test_pool_bound_401_retry_defers_to_pool_not_singleton(capsys):
    """401s on a pool binding are recovered by the POOL (``try_refresh_matching``
    → rotate in ``_recover_auth_failure``), which runs *before* the per-provider
    singleton fallback in ``recover_after_classification``. By the time
    ``_refresh_credentials_after_401`` runs, the pool has already refreshed or
    rotated (and it adopts out-of-band-rotated pairs itself), so the singleton
    fallback must not swap in a possibly-different account; it reports the
    surviving 401 instead."""
    from agent.turn_recovery import _refresh_credentials_after_401
    from agent.turn_retry_state import TurnRetryState

    agent = _pool_bound_agent()
    agent.log_prefix = ""
    original_client = agent._anthropic_client
    retry = TurnRetryState()

    with (
        patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="test-account-one") as singleton,
        patch("agent.anthropic_adapter.build_anthropic_client") as rebuild,
    ):
        retried = _refresh_credentials_after_401(agent, Exception("401 unauthorized"), retry, 401)

    assert retried is False
    assert retry.anthropic_auth_retry_attempted is True
    assert agent._anthropic_api_key == "test-account-two"
    assert agent._anthropic_client is original_client
    singleton.assert_not_called()
    rebuild.assert_not_called()
    # The surviving 401 is still surfaced to the user, not silently swallowed.
    assert "Anthropic 401" in capsys.readouterr().out


def test_wire_receipt_uses_actual_request_header_without_logging_secret(caplog):
    import logging
    import httpx

    agent: Any = object.__new__(AIAgent)
    agent.provider = "anthropic"
    agent.model = "claude-fable-5"
    agent.api_key = "private-selected-token"
    agent._anthropic_api_key = "private-selected-token"
    agent._credential_pool_entry_id = "selected-entry"
    agent.log_prefix = "[test-session] "
    agent._capture_rate_limits = MagicMock()
    agent._capture_credits = MagicMock()
    for sent, expected in (("private-selected-token", "True"), ("private-other-token", "False")):
        caplog.clear()
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages", headers={"authorization": "Bearer " + sent})
        response = httpx.Response(200, request=request, headers={"request-id": "test-request"})
        with caplog.at_level(logging.INFO, logger="run_agent"):
            agent._capture_anthropic_response_headers(response)
        assert "Anthropic wire receipt" in caplog.text
        assert "pool_entry=selected-entry" in caplog.text
        assert "matches_selected=" + expected in caplog.text
        assert "request_id=test-request" in caplog.text
        assert "private-selected-token" not in caplog.text
        assert "private-other-token" not in caplog.text
