"""Tests for the Slack adapter's ``display_name`` override in bot identity.

``auth.test`` pins the installed bot's username at install time; the override
makes the recorded identity follow an explicitly configured name instead.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter

_AUTH_RESPONSE = {
    "ok": True,
    "team_id": "T123",
    "team": "Test Workspace",
    "user_id": "U456",
    "user": "pinned-username",
    "scope": "bots",
    "is_bot": True,
}


def _make_adapter(extra):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake", extra=extra))
    client = MagicMock()
    client.auth_test = AsyncMock(return_value=dict(_AUTH_RESPONSE))
    adapter._new_web_client = lambda token, proxy_url: client
    for name in (
        "_warn_if_missing_group_dm_scopes",
        "_warn_if_not_bot_token",
        "_warn_if_inchannel_without_flat_reply",
    ):
        setattr(adapter, name, lambda *args, **kwargs: None)
    return adapter


@pytest.mark.asyncio
async def test_display_name_override_wins_over_pinned_username():
    adapter = _make_adapter({"display_name": "renamed-bot"})
    await adapter._authenticate_workspace("xoxb-fake", None)
    assert adapter._team_bot_names["T123"] == "renamed-bot"
    assert adapter._bot_display_name == "renamed-bot"


@pytest.mark.asyncio
async def test_no_override_keeps_auth_test_username():
    adapter = _make_adapter({})
    await adapter._authenticate_workspace("xoxb-fake", None)
    assert adapter._team_bot_names["T123"] == "pinned-username"
    assert adapter._bot_display_name == "pinned-username"


@pytest.mark.asyncio
async def test_blank_display_name_is_ignored():
    adapter = _make_adapter({"display_name": "   "})
    await adapter._authenticate_workspace("xoxb-fake", None)
    assert adapter._team_bot_names["T123"] == "pinned-username"
    assert adapter._bot_display_name == "pinned-username"
