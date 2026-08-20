"""Regression coverage for Slack API response compression negotiation."""

from unittest.mock import patch

from plugins.platforms.slack import adapter as slack_module


def test_slack_web_client_excludes_brotli_from_accept_encoding():
    """Slack API calls must avoid intermittent aiohttp Brotli decode failures."""
    sentinel = object()

    with patch.object(slack_module, "AsyncWebClient", return_value=sentinel) as factory:
        client = slack_module._new_slack_web_client("xoxb-test")

    assert client is sentinel
    kwargs = factory.call_args.kwargs
    assert kwargs["token"] == "xoxb-test"
    assert kwargs["user_agent_prefix"] == slack_module._HERMES_SLACK_USER_AGENT_PREFIX
    accept_encoding = kwargs["headers"]["Accept-Encoding"]
    assert [token.strip() for token in accept_encoding.split(",")] == ["gzip"]