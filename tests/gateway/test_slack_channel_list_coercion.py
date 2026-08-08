"""
Tests for JSON-array-string tolerance in Slack channel-list settings.

A channel list serialized as JSON inside a YAML string —

    allowed_channels: '["C0AAAAAAA", "C0BBBBBBB"]'

— used to be CSV-split verbatim, leaving brackets/quotes on every entry.
A non-empty whitelist of garbage IDs matches nothing, so the bot went
silent in every channel with no error anywhere. ``_coerce_channel_list``
decodes the JSON form back into a list at both boundaries (the YAML→env
translation in ``_apply_yaml_config`` and the ``config.extra`` readers),
and ``_warn_malformed_channel_entries`` surfaces leftover garbage at
startup instead of failing silently.

Follows the mock pattern of test_slack_mention.py.
"""

import sys
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Mock slack-bolt if not installed (same as test_slack.py)
# ---------------------------------------------------------------------------

def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

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
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod
_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import (  # noqa: E402
    SlackAdapter,
    _apply_yaml_config,
    _coerce_channel_list,
)


CHANNEL_A = "C0AAAAAAAAA"
CHANNEL_B = "C0BBBBBBBBB"

CHANNEL_ENV_VARS = [
    "SLACK_ALLOWED_CHANNELS",
    "SLACK_IGNORED_CHANNELS",
    "SLACK_FREE_RESPONSE_CHANNELS",
    "SLACK_REQUIRE_MENTION_CHANNELS",
]


def _make_adapter(**extra):
    adapter = object.__new__(SlackAdapter)
    adapter.platform = Platform.SLACK
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._bot_user_id = "U_BOT_123"
    adapter._team_bot_user_ids = {}
    return adapter


@pytest.fixture(autouse=True)
def _clean_channel_env(monkeypatch):
    for var in CHANNEL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# _coerce_channel_list
# ---------------------------------------------------------------------------

def test_coerce_decodes_json_array_string():
    raw = f'["{CHANNEL_A}", "{CHANNEL_B}"]'
    assert _coerce_channel_list(raw) == [CHANNEL_A, CHANNEL_B]


def test_coerce_leaves_csv_string_unchanged():
    raw = f"{CHANNEL_A},{CHANNEL_B}"
    assert _coerce_channel_list(raw) == raw


def test_coerce_leaves_plain_id_unchanged():
    assert _coerce_channel_list(CHANNEL_A) == CHANNEL_A


def test_coerce_leaves_invalid_json_unchanged():
    raw = '["unterminated'
    assert _coerce_channel_list(raw) == raw


def test_coerce_decodes_edge_json_arrays():
    assert _coerce_channel_list("[]") == []
    assert _coerce_channel_list('["only"]') == ["only"]
    assert _coerce_channel_list("[1, 2]") == [1, 2]


def test_coerce_passes_through_lists_and_none():
    assert _coerce_channel_list([CHANNEL_A]) == [CHANNEL_A]
    assert _coerce_channel_list(None) is None


# ---------------------------------------------------------------------------
# _apply_yaml_config: YAML→env translation tolerates JSON-array strings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "yaml_key, env_var",
    [
        ("allowed_channels", "SLACK_ALLOWED_CHANNELS"),
        ("ignored_channels", "SLACK_IGNORED_CHANNELS"),
        ("free_response_channels", "SLACK_FREE_RESPONSE_CHANNELS"),
        ("require_mention_channels", "SLACK_REQUIRE_MENTION_CHANNELS"),
    ],
)
def test_apply_yaml_config_decodes_json_array_string(yaml_key, env_var, monkeypatch):
    import os

    slack_cfg = {yaml_key: f'["{CHANNEL_A}", "{CHANNEL_B}"]'}
    _apply_yaml_config({}, slack_cfg)
    assert os.environ[env_var] == f"{CHANNEL_A},{CHANNEL_B}"


def test_apply_yaml_config_keeps_csv_and_list_forms(monkeypatch):
    import os

    _apply_yaml_config({}, {"allowed_channels": [CHANNEL_A, CHANNEL_B]})
    assert os.environ["SLACK_ALLOWED_CHANNELS"] == f"{CHANNEL_A},{CHANNEL_B}"

    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)
    _apply_yaml_config({}, {"allowed_channels": f"{CHANNEL_A},{CHANNEL_B}"})
    assert os.environ["SLACK_ALLOWED_CHANNELS"] == f"{CHANNEL_A},{CHANNEL_B}"


def test_apply_yaml_config_warns_on_garbage_entries(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        # Broken JSON falls through to the CSV path and keeps its brackets —
        # the startup warning must name the offending entries.
        _apply_yaml_config({}, {"allowed_channels": f'["{CHANNEL_A}", "{CHANNEL_B}'})
    assert any("allowed_channels" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Adapter readers: config.extra path tolerates JSON-array strings too
# ---------------------------------------------------------------------------

def test_allowed_channels_reader_decodes_json_string():
    adapter = _make_adapter(allowed_channels=f'["{CHANNEL_A}", "{CHANNEL_B}"]')
    assert adapter._slack_allowed_channels() == {CHANNEL_A, CHANNEL_B}


def test_ignored_channels_reader_decodes_json_string():
    adapter = _make_adapter(ignored_channels=f'["{CHANNEL_A}"]')
    assert adapter._slack_ignored_channels() == {CHANNEL_A}


def test_free_response_channels_reader_decodes_json_string():
    adapter = _make_adapter(free_response_channels=f'["{CHANNEL_A}"]')
    assert adapter._slack_free_response_channels() == {CHANNEL_A}


def test_require_mention_channels_reader_decodes_json_string():
    adapter = _make_adapter(require_mention_channels=f'["{CHANNEL_A}"]')
    assert adapter._slack_require_mention_channels() == {CHANNEL_A}


def test_allowed_channels_reader_env_json_string(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", f'["{CHANNEL_A}", "{CHANNEL_B}"]')
    adapter = _make_adapter()
    assert adapter._slack_allowed_channels() == {CHANNEL_A, CHANNEL_B}


def test_allowed_channels_reader_csv_unchanged():
    adapter = _make_adapter(allowed_channels=f"{CHANNEL_A},{CHANNEL_B}")
    assert adapter._slack_allowed_channels() == {CHANNEL_A, CHANNEL_B}
