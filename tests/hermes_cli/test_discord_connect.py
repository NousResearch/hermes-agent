"""Form checks for the desktop Discord connection. No live Discord calls."""

import json

import pytest

from hermes_cli.discord_connect import (
    DiscordConnectError,
    clean_message,
    clean_snowflake,
    clean_token,
    list_text_channels,
    outcome_error,
    resolve_token,
    text_channels_from_guilds,
)


def test_user_and_channel_ids_are_numeric_mentions():
    assert clean_snowflake("<@880636691499581512>") == "880636691499581512"
    assert clean_snowflake("<#1497111463976898620>") == "1497111463976898620"
    with pytest.raises(DiscordConnectError):
        clean_snowflake("go880636691499581512")


def test_token_rejects_an_application_id():
    with pytest.raises(DiscordConnectError):
        clean_token("123456789012345678")
    token = "aaa.bbb.ccc-token-value"
    assert clean_token(token) == token
    assert resolve_token("", "saved.bot.token-value") == "saved.bot.token-value"
    with pytest.raises(DiscordConnectError):
        resolve_token("", "")


def test_message_must_be_nonempty_and_within_discord_limit():
    assert clean_message("  hello  ") == "hello"
    with pytest.raises(DiscordConnectError):
        clean_message("   ")
    with pytest.raises(DiscordConnectError):
        clean_message("x" * 2001)


def test_channel_list_keeps_text_channels_only():
    listed = text_channels_from_guilds(
        [{"id": "1", "name": "Server"}],
        {"1": [
            {"id": "1497111463976898622", "name": "voice", "type": 2, "position": 0},
            {"id": "1497111463976898620", "name": "day-to-day-tasks", "type": 0, "position": 2},
            {"id": "1497111463976898621", "name": "announcements", "type": 5, "position": 1},
        ]},
    )
    assert listed == [{
        "guild": "Server",
        "channels": [
            {"id": "1497111463976898621", "name": "announcements"},
            {"id": "1497111463976898620", "name": "day-to-day-tasks"},
        ],
    }]


def test_channel_list_uses_the_opener_and_hides_the_token():
    class _Body:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def opener(request, timeout=20):
        assert timeout == 20
        assert request.get_header("Authorization", "").startswith("Bot ")
        url = request.full_url
        if url.endswith("/users/@me/guilds"):
            return _Body([{"id": "9", "name": "bhavesh"}])
        return _Body([{"id": "1497111463976898620", "name": "day-to-day-tasks", "type": 0, "position": 0}])

    listed = list_text_channels("secret.bot.token-value", opener)
    assert listed[0]["channels"][0]["name"] == "day-to-day-tasks"


def test_send_result_reports_failure_without_treating_success_as_an_error():
    assert outcome_error(json.dumps({"success": True})) is None
    assert "Missing Access" in outcome_error(json.dumps({"error": "Missing Access"}))
