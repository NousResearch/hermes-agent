"""Regression tests for Copilot raw-token exchange during runtime resolution.

A raw GitHub OAuth token (``ghu_...``) sent as the Bearer to
``api.githubcopilot.com`` makes GitHub ignore ``Copilot-Integration-Id:
vscode-chat`` and pin the request to integrator ``copilot-language-server``,
whose model allow-list is tiny (only a handful of GPT models). Claude, Gemini
and most GPT models then fail with HTTP 400
``model_not_available_for_integrator``. The credential pool can end up holding
the raw token (the env seeder overwrites the exchanged singleton entry under
the same source key), so runtime resolution must defensively exchange a raw
token for a short-lived Copilot API token (``tid=...``) before use.

These tests pin that behaviour without any network access.
"""
from unittest.mock import patch

from hermes_cli import runtime_provider as rp


def test_looks_like_raw_github_token_recognizes_prefixes():
    for raw in (
        "ghu_abc123",
        "gho_abc123",
        "ghp_abc123",
        "ghs_abc123",
        "github_pat_abc123",
        "  ghu_leading_ws  ",
    ):
        assert rp._looks_like_raw_github_token(raw) is True

    # Already-exchanged Copilot API tokens and empty values are not "raw".
    for other in ("tid=abc;exp=123", "", "sk-proj-abc", "some-random-key"):
        assert rp._looks_like_raw_github_token(other) is False


def test_exchange_copilot_api_key_exchanges_raw_token_and_adopts_base_url():
    with patch(
        "hermes_cli.copilot_auth.get_copilot_api_token",
        return_value=("tid=exchanged;exp=999", "https://api.business.githubcopilot.com/"),
    ) as mock_exchange:
        api_key, base_url = rp._exchange_copilot_api_key(
            "ghu_rawtoken", "https://api.githubcopilot.com"
        )

    mock_exchange.assert_called_once_with("ghu_rawtoken")
    assert api_key == "tid=exchanged;exp=999"
    # Account-specific endpoint advertised by the exchange is adopted (trailing
    # slash stripped) so Business/Enterprise tenants hit the right host.
    assert base_url == "https://api.business.githubcopilot.com"


def test_exchange_copilot_api_key_passes_through_already_exchanged_token():
    with patch("hermes_cli.copilot_auth.get_copilot_api_token") as mock_exchange:
        api_key, base_url = rp._exchange_copilot_api_key(
            "tid=already;exp=1", "https://api.githubcopilot.com"
        )

    # No exchange attempted for a non-raw token; values pass through unchanged.
    mock_exchange.assert_not_called()
    assert api_key == "tid=already;exp=1"
    assert base_url == "https://api.githubcopilot.com"


def test_exchange_copilot_api_key_keeps_base_url_when_exchange_returns_none():
    # get_copilot_api_token returns (raw_token, None) on exchange failure; the
    # original base_url must be preserved rather than clobbered with None.
    with patch(
        "hermes_cli.copilot_auth.get_copilot_api_token",
        return_value=("ghu_rawtoken", None),
    ):
        api_key, base_url = rp._exchange_copilot_api_key(
            "ghu_rawtoken", "https://api.githubcopilot.com"
        )

    assert api_key == "ghu_rawtoken"
    assert base_url == "https://api.githubcopilot.com"
