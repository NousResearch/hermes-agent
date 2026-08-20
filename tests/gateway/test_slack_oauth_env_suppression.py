"""Regression tests for issue #86228.

When SLACK_CLIENT_ID and SLACK_CLIENT_SECRET are both present in the
environment (e.g. because the user configured the Slack MCP server),
slack_bolt's ``AsyncApp.__init__`` auto-enables multi-team OAuth — even
though Hermes authenticates with a plain bot token and never runs an
OAuth install flow.  The auto-enabled ``FileInstallationStore`` is empty,
so ``authorize()`` returns ``None`` for every inbound event and the event
is silently dropped before any Hermes handler runs.

The adapter suppresses those two env vars for the duration of the
``AsyncApp`` constructor call so slack_bolt stays in single-team bot-token
mode.

These tests verify:
1. The real ``SlackAdapter.connect`` path constructs ``AsyncApp`` once with
   both OAuth env vars absent, then restores them.
2. Restoration also happens when the constructor raises.
3. A partial environment is left untouched because slack_bolt requires both
   names to activate its implicit OAuth mode.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure slack modules are available (real or mock)
# ---------------------------------------------------------------------------

_REAL_SLACK = False
try:
    import slack_bolt  # noqa: F401
    if hasattr(slack_bolt, "__file__"):
        _REAL_SLACK = True
except ImportError:
    pass

if not _REAL_SLACK:
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock
    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock
    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        (
            "slack_bolt.adapter.socket_mode.async_handler",
            slack_bolt.adapter.socket_mode.async_handler,
        ),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("aiohttp", MagicMock())

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


class _RecordingApp:
    """Small Slack Bolt stand-in that supports all registrations in connect()."""

    def __init__(self, token, client=None, **_kwargs):
        self.token = token
        self.client = client

    @staticmethod
    def _decorator(_matcher):
        def decorate(fn):
            return fn

        return decorate

    def event(self, matcher):
        return self._decorator(matcher)

    def command(self, matcher):
        return self._decorator(matcher)

    def action(self, matcher):
        return self._decorator(matcher)


class _FakeWebClient:
    """Deterministic Slack SDK client for the adapter's auth bootstrap."""

    def __init__(self, token, **_kwargs):
        self.token = token
        self.proxy = None

    async def auth_test(self):
        return {
            "team_id": "T_FAKE",
            "user_id": "U_BOT",
            "user": "testbot",
            "team": "FakeTeam",
        }


async def _run_connect(
    adapter,
    app_factory,
    env,
    tmp_path,
):
    """Run the real adapter connect path with only external Slack I/O stubbed."""
    app_constructor = MagicMock(side_effect=app_factory)
    plugin_manager = MagicMock()
    plugin_manager.get_slack_action_handlers.return_value = []

    with (
        patch.object(_slack_mod, "AsyncApp", app_constructor),
        patch.object(_slack_mod, "AsyncWebClient", _FakeWebClient),
        patch.object(_slack_mod, "_resolve_slack_proxy_url", return_value=None),
        patch.object(_slack_mod, "get_secret", return_value="xapp-fake"),
        patch("hermes_constants.get_hermes_home", return_value=tmp_path),
        patch("hermes_cli.commands.slack_native_slashes", return_value=[]),
        patch("hermes_cli.plugins.get_plugin_manager", return_value=plugin_manager),
        patch.object(adapter, "_acquire_platform_lock", return_value=True),
        patch.object(adapter, "_stop_socket_mode_handler", new=AsyncMock()),
        patch.object(adapter, "_close_workspace_clients", new=AsyncMock()),
        patch.object(adapter, "_start_socket_mode_handler"),
        patch.object(adapter, "_ensure_socket_watchdog"),
        patch.object(adapter, "_release_platform_lock"),
        patch.dict(os.environ, env, clear=True),
    ):
        result = await adapter.connect()
        restored_env = {
            key: os.environ.get(key, _MISSING)
            for key in ("SLACK_CLIENT_ID", "SLACK_CLIENT_SECRET")
        }

    return result, app_constructor, restored_env


_MISSING = object()


@pytest.mark.asyncio
async def test_connect_suppresses_oauth_env_and_restores_it(tmp_path, caplog):
    """connect() must protect the real AsyncApp construction boundary."""
    observed_env = []

    def build_app(*, token, client, **_kwargs):
        observed_env.append(
            {
                key: os.environ.get(key, _MISSING)
                for key in ("SLACK_CLIENT_ID", "SLACK_CLIENT_SECRET")
            }
        )
        return _RecordingApp(token, client)

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    with caplog.at_level("INFO", logger=_slack_mod.logger.name):
        result, app_constructor, restored_env = await _run_connect(
            adapter,
            build_app,
            {
                "SLACK_APP_TOKEN": "xapp-fake",
                "SLACK_CLIENT_ID": "client-id",
                "SLACK_CLIENT_SECRET": "client-secret",
            },
            tmp_path,
        )

    assert result is True
    app_constructor.assert_called_once()
    assert observed_env == [
        {"SLACK_CLIENT_ID": _MISSING, "SLACK_CLIENT_SECRET": _MISSING}
    ]
    assert restored_env == {
        "SLACK_CLIENT_ID": "client-id",
        "SLACK_CLIENT_SECRET": "client-secret",
    }
    assert "Suppressing SLACK_CLIENT_ID, SLACK_CLIENT_SECRET" in caplog.text


@pytest.mark.asyncio
async def test_connect_restores_oauth_env_when_app_construction_fails(tmp_path):
    """The process environment must survive an AsyncApp constructor error."""
    observed_env = []

    def fail_to_build_app(*, token, client, **_kwargs):
        observed_env.append(
            {
                key: os.environ.get(key, _MISSING)
                for key in ("SLACK_CLIENT_ID", "SLACK_CLIENT_SECRET")
            }
        )
        raise RuntimeError("constructor failed")

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    result, app_constructor, restored_env = await _run_connect(
        adapter,
        fail_to_build_app,
        {
            "SLACK_APP_TOKEN": "xapp-fake",
            "SLACK_CLIENT_ID": "client-id",
            "SLACK_CLIENT_SECRET": "client-secret",
        },
        tmp_path,
    )

    assert result is False
    app_constructor.assert_called_once()
    assert observed_env == [
        {"SLACK_CLIENT_ID": _MISSING, "SLACK_CLIENT_SECRET": _MISSING}
    ]
    assert restored_env == {
        "SLACK_CLIENT_ID": "client-id",
        "SLACK_CLIENT_SECRET": "client-secret",
    }


@pytest.mark.asyncio
async def test_connect_leaves_partial_oauth_env_untouched(tmp_path):
    """Only one OAuth variable must not trigger the suppression guard."""
    observed_env = []

    def build_app(*, token, client, **_kwargs):
        observed_env.append(
            {
                key: os.environ.get(key, _MISSING)
                for key in ("SLACK_CLIENT_ID", "SLACK_CLIENT_SECRET")
            }
        )
        return _RecordingApp(token, client)

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    result, app_constructor, restored_env = await _run_connect(
        adapter,
        build_app,
        {
            "SLACK_APP_TOKEN": "xapp-fake",
            "SLACK_CLIENT_ID": "only-client-id",
        },
        tmp_path,
    )

    assert result is True
    app_constructor.assert_called_once()
    assert observed_env == [
        {"SLACK_CLIENT_ID": "only-client-id", "SLACK_CLIENT_SECRET": _MISSING}
    ]
    assert restored_env == {
        "SLACK_CLIENT_ID": "only-client-id",
        "SLACK_CLIENT_SECRET": _MISSING,
    }
