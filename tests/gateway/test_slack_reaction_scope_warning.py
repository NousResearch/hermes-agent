"""Status reactions need ``reactions:write``; without it ``reactions.add`` fails on every message.

``_react`` logs that failure at debug only, so the connect-time ``auth.test`` scope header is
where the adapter tells the user (#157865), once per workspace and only while reactions are on.
"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod

_slack_mod.SLACK_AVAILABLE = True

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter

ADAPTER_LOGGER = "plugins.platforms.slack.adapter"
SCOPES_WITHOUT_WRITE = "app_mentions:read,channels:history,chat:write,im:history,mpim:history,reactions:read"


def _auth(scopes):
    return SimpleNamespace(headers={"x-oauth-scopes": scopes} if scopes is not None else {})


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "reactions:write" in r.getMessage()]


@pytest.fixture(autouse=True)
def _no_reactions_env(monkeypatch):
    monkeypatch.delenv("SLACK_REACTIONS", raising=False)


def test_missing_reactions_write_warns_once_per_workspace(caplog):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    with caplog.at_level(logging.DEBUG, logger=ADAPTER_LOGGER):
        adapter._warn_if_missing_reaction_scope(_auth(SCOPES_WITHOUT_WRITE), "Acme")
        adapter._warn_if_missing_reaction_scope(_auth(SCOPES_WITHOUT_WRITE), "Acme")  # reconnect

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "Acme" in warnings[0] and "SLACK_REACTIONS=false" in warnings[0]


def test_granted_reactions_write_is_silent(caplog):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    with caplog.at_level(logging.DEBUG, logger=ADAPTER_LOGGER):
        adapter._warn_if_missing_reaction_scope(_auth(SCOPES_WITHOUT_WRITE + ",reactions:write"), "Acme")

    assert _warnings(caplog) == []


def test_reactions_turned_off_is_silent(caplog, monkeypatch):
    monkeypatch.setenv("SLACK_REACTIONS", "false")
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    with caplog.at_level(logging.DEBUG, logger=ADAPTER_LOGGER):
        adapter._warn_if_missing_reaction_scope(_auth(SCOPES_WITHOUT_WRITE), "Acme")

    assert _warnings(caplog) == []


def test_absent_scope_header_is_silent(caplog):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    with caplog.at_level(logging.DEBUG, logger=ADAPTER_LOGGER):
        adapter._warn_if_missing_reaction_scope(_auth(None), "Acme")

    assert _warnings(caplog) == []
