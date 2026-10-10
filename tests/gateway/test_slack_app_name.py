"""Slack connect() must not let slack_bolt derive the app name via inspect.stack().

AsyncApp(name=None) calls ``inspect.stack()``, which resolves source for every frame on
the gateway event loop. Under load that blocked the loop past the liveness watchdog and
restarted the gateway on every Slack (re)connect attempt.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

from tests.gateway.test_slack_log_noise import _fake_create_task, _slack_mod  # noqa: F401  (installs slack mocks)
from plugins.platforms.slack.adapter import SlackAdapter
from gateway.config import PlatformConfig


def test_connect_passes_explicit_app_name():
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    mock_app = MagicMock()
    mock_app.event = lambda *_a, **_k: (lambda fn: fn)
    mock_app.action = lambda *_a, **_k: (lambda fn: fn)
    mock_app.command = lambda *_a, **_k: (lambda fn: fn)
    mock_app.client = AsyncMock()
    web_client = AsyncMock()
    web_client.auth_test = AsyncMock(return_value={
        "user_id": "U_BOT", "user": "bot", "team_id": "T_FAKE", "team": "Fake"})
    handler = MagicMock()
    handler.start_async = AsyncMock(return_value=None)

    with (
        patch.object(_slack_mod, "AsyncApp", return_value=mock_app) as app_cls,
        patch.object(_slack_mod, "AsyncWebClient", return_value=web_client),
        patch.object(_slack_mod, "AsyncSocketModeHandler", return_value=handler),
        patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}),
        patch("gateway.status.acquire_scoped_lock", return_value=(True, None)),
        patch("asyncio.create_task", side_effect=_fake_create_task),
    ):
        asyncio.run(adapter.connect())

    assert app_cls.call_count == 1
    assert app_cls.call_args.kwargs.get("name"), app_cls.call_args
