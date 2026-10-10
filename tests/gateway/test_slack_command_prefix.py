"""Slack slash-command namespace prefix (``platforms.slack.extra.command_prefix``).

Lets several gateway apps share one Slack workspace: native slashes, the ``/hermes``
catch-all, ``!`` bang commands and the generated app manifest all carry the prefix.
"""

import importlib
import logging
import sys
from importlib.machinery import PathFinder
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType


def _load_installed_package(name):
    if PathFinder.find_spec(name) is None:
        return None
    prefix = f"{name}."
    displaced = {
        m: sys.modules.pop(m)
        for m in tuple(sys.modules)
        if (m == name or m.startswith(prefix)) and not isinstance(sys.modules[m], ModuleType)
    }
    try:
        return importlib.import_module(name)
    except ImportError:
        sys.modules.update(displaced)
        return None


_load_installed_package("slack_bolt")
_load_installed_package("slack_sdk")

SlackAdapter = importlib.import_module("plugins.platforms.slack.adapter").SlackAdapter


def _config_with_prefix(prefix):
    """Fake config.yaml payload for the manifest-side prefix lookup."""
    return {"platforms": {"slack": {"extra": {"command_prefix": prefix}}}}


class TestSlackCommandPrefix:
    """Namespace prefix so multiple gateway apps can share one workspace."""

    def _make(self, extra):
        config = PlatformConfig(enabled=True, token="xoxb-fake", extra=extra or {})
        a = SlackAdapter(config)
        a._app = MagicMock()
        a._app.client = AsyncMock()
        a._bot_user_id = "U_BOT"
        a._running = True
        a.handle_message = AsyncMock()
        return a

    @pytest.mark.asyncio
    async def test_prefixed_native_slash_is_stripped_and_routed(self):
        adapter = self._make({"command_prefix": "myorg-"})
        assert adapter._command_prefix == "myorg-"

        await adapter._handle_slash_command(
            {
                "command": "/myorg-model",
                "text": "opus",
                "user_id": "U1",
                "channel_id": "C1",
                "team_id": "T1",
            }
        )

        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/model opus"
        assert event.message_type == MessageType.COMMAND

    @pytest.mark.asyncio
    async def test_prefixed_hermes_routes_freeform_question(self):
        adapter = self._make({"command_prefix": "myorg-"})

        await adapter._handle_slash_command(
            {
                "command": "/myorg-hermes",
                "text": "what's up",
                "user_id": "U1",
                "channel_id": "C1",
                "team_id": "T1",
            }
        )

        event = adapter.handle_message.await_args.args[0]
        assert event.text == "what's up"
        assert event.message_type == MessageType.TEXT

    @pytest.mark.asyncio
    async def test_no_prefix_default_leaves_commands_unchanged(self):
        adapter = self._make({})
        assert adapter._command_prefix == ""

        await adapter._handle_slash_command(
            {
                "command": "/model",
                "text": "opus",
                "user_id": "U1",
                "channel_id": "C1",
                "team_id": "T1",
            }
        )

        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/model opus"

    def test_manifest_prepends_prefix(self, monkeypatch):
        from hermes_cli.commands_platforms import slack_app_manifest

        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config", lambda: _config_with_prefix("myorg-")
        )
        slashes = slack_app_manifest()["features"]["slash_commands"]

        assert slashes
        assert all(s["command"].startswith("/myorg-") for s in slashes)
        assert any(s["command"] == "/myorg-hermes" for s in slashes)

    def test_manifest_silent_when_all_prefixed_names_fit(self, monkeypatch, caplog):
        from hermes_cli import commands_platforms as cmds

        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config", lambda: _config_with_prefix("myorg-")
        )
        monkeypatch.setattr(
            cmds,
            "slack_native_slashes",
            lambda prefix="": [("hermes", "d", ""), ("model", "d", "")],
        )
        with caplog.at_level(logging.WARNING, logger="hermes_cli.commands"):
            slashes = cmds.slack_app_manifest()["features"]["slash_commands"]

        assert [s["command"] for s in slashes] == ["/myorg-hermes", "/myorg-model"]
        # Nothing was skipped, so the generator must stay silent.
        assert not caplog.records

    def test_manifest_default_has_no_prefix(self, monkeypatch):
        from hermes_cli.commands_platforms import slack_app_manifest

        # Empty config is authoritative — avoids reading a real config file.
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})
        slashes = slack_app_manifest()["features"]["slash_commands"]

        assert any(s["command"] == "/hermes" for s in slashes)
        assert all(not s["command"].startswith("/myorg-") for s in slashes)

    def test_manifest_skips_prefixed_name_over_slack_limit(self, monkeypatch, caplog):
        from hermes_cli import commands_platforms as cmds

        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config", lambda: _config_with_prefix("myorg-")
        )
        long_name = "x" * 30  # 30 + len("myorg-") = 35 > 32-char Slack limit
        monkeypatch.setattr(
            cmds,
            "slack_native_slashes",
            lambda prefix="": [("hermes", "d", ""), (long_name, "d", ""), ("model", "d", "")],
        )
        with caplog.at_level(logging.WARNING, logger="hermes_cli.commands"):
            names = [s["command"] for s in cmds.slack_app_manifest()["features"]["slash_commands"]]

        assert "/myorg-hermes" in names
        assert "/myorg-model" in names
        assert f"/myorg-{long_name}" not in names

        # The skip is warned, naming the dropped command and the fallback path.
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert f"myorg-{long_name}" in message
        assert "/myorg-hermes <subcommand>" in message

    def test_manifest_warns_when_catchall_itself_over_limit(self, monkeypatch, caplog):
        from hermes_cli import commands_platforms as cmds

        # 27-char prefix: "hermes" (6) no longer fits (33 > 32) but "model"
        # (5) still does — the warning must not advise the dead fallback.
        prefix = "x" * 26 + "-"
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config", lambda: _config_with_prefix(prefix)
        )
        monkeypatch.setattr(
            cmds,
            "slack_native_slashes",
            lambda prefix="": [("hermes", "d", ""), ("model", "d", "")],
        )
        with caplog.at_level(logging.WARNING, logger="hermes_cli.commands"):
            names = [s["command"] for s in cmds.slack_app_manifest()["features"]["slash_commands"]]

        assert names == [f"/{prefix}model"]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert f"{prefix}hermes" in message
        assert "<subcommand>" not in message
        assert "shorter command_prefix" in message

    def test_all_invalid_prefix_warns_and_disables_namespacing(self, caplog):
        from hermes_cli.commands_platforms import slack_command_prefix

        with caplog.at_level(logging.WARNING, logger="hermes_cli.commands"):
            prefix = slack_command_prefix({"command_prefix": "@@@"})

        assert prefix == ""
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "NOT namespaced" in warnings[0].getMessage()

    @pytest.mark.asyncio
    async def test_bang_prefixed_command_rewritten_in_thread(self):
        """``!myorg-model`` mirrors the prefixed native-slash surface."""
        adapter = self._make({"command_prefix": "myorg-"})

        await adapter._handle_slack_message(
            {
                "text": "!myorg-model opus",
                "user": "U_USER",
                "channel": "D123",
                "channel_type": "im",
                "ts": "1234567890.000001",
                "thread_ts": "1111111111.000001",
            }
        )

        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.text.startswith("/model opus")
        assert msg_event.message_type == MessageType.COMMAND

    @pytest.mark.asyncio
    async def test_bang_unprefixed_stays_text_when_prefix_set(self):
        """Two namespaced apps sharing a channel must not both run ``!model``."""
        adapter = self._make({"command_prefix": "myorg-"})

        await adapter._handle_slack_message(
            {
                "text": "!model opus",
                "user": "U_USER",
                "channel": "D123",
                "channel_type": "im",
                "ts": "1234567890.000001",
            }
        )

        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.text == "!model opus"
        assert msg_event.message_type != MessageType.COMMAND

    def test_manifest_prefix_unreserves_slack_builtin_names(self, monkeypatch):
        from hermes_cli.commands_platforms import slack_app_manifest

        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})
        bare = {s["command"] for s in slack_app_manifest()["features"]["slash_commands"]}
        # Bare /status collides with the Slack built-in, so it is skipped.
        assert "/status" not in bare

        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config", lambda: _config_with_prefix("myorg-")
        )
        prefixed = {s["command"] for s in slack_app_manifest()["features"]["slash_commands"]}
        # /myorg-status no longer collides, so it gets a native slot.
        assert "/myorg-status" in prefixed

    def test_manifest_alias_descriptions_use_prefixed_names(self, monkeypatch):
        from hermes_cli import commands_platforms as cmds

        # Canonical names alone fill the 50-slash cap, so the alias pass never gets a slot
        # against the real registry. Narrow the available set to one aliased command so an
        # alias entry is observable regardless of how large the registry grows.
        aliased = next(c for c in cmds.COMMAND_REGISTRY if c.aliases)
        monkeypatch.setattr(cmds, "_gateway_available_commands", lambda: [aliased])
        monkeypatch.setattr(cmds, "_iter_plugin_command_entries", lambda: [])
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config", lambda: _config_with_prefix("myorg-")
        )
        slashes = cmds.slack_app_manifest()["features"]["slash_commands"]

        alias_descs = [
            s["description"] for s in slashes if s["description"].startswith("Alias for /")
        ]
        assert alias_descs  # the single aliased command leaves room for its alias
        # Descriptions must reference the names actually registered
        # (/myorg-background), not the unregistered bare forms.
        assert all(d.startswith("Alias for /myorg-") for d in alias_descs)


@pytest.mark.parametrize(
    ("text", "prefix", "expected"),
    [
        ("!", "", "!"),
        ("!   ", "", "!   "),
        ("!nice work", "", "!nice work"),
        ("!model opus", "", "/model opus"),
        ("!model@hermes opus", "", "/model@hermes opus"),
        ("!myorg-model opus", "myorg-", "/model opus"),
        ("!model opus", "myorg-", "!model opus"),
        ("!myorg-", "myorg-", "!myorg-"),
        ("plain text", "myorg-", "plain text"),
    ],
)
def test_bang_rewrite_contract(text, prefix, expected):
    from plugins.platforms.slack.adapter_slash import _rewrite_known_bang_command

    assert _rewrite_known_bang_command(text, prefix) == expected

