"""Tests for Slack CLI helpers."""
import argparse
from unittest.mock import patch

from hermes_cli.slack_cli import _build_full_manifest


class TestSlackFullManifest:
    """Generated full Slack app manifest used by `hermes slack manifest`."""

    def test_app_home_messages_are_writable(self):
        manifest = _build_full_manifest("Hermes", "Your Hermes agent on Slack")

        assert manifest["features"]["app_home"] == {
            "home_tab_enabled": False,
            "messages_tab_enabled": True,
            "messages_tab_read_only_enabled": False,
        }

    def test_private_channel_directory_scope_is_included(self):
        manifest = _build_full_manifest("Hermes", "Your Hermes agent on Slack")

        bot_scopes = manifest["oauth_config"]["scopes"]["bot"]
        assert "groups:read" in bot_scopes

    def test_assistant_features_remain_enabled(self):
        manifest = _build_full_manifest("Hermes", "Your Hermes agent on Slack")

        assert "assistant_view" in manifest["features"]
        assert "assistant:write" in manifest["oauth_config"]["scopes"]["bot"]
        bot_events = manifest["settings"]["event_subscriptions"]["bot_events"]
        assert "assistant_thread_started" in bot_events


class TestSlackManifestCommandConfigFallback:
    """``hermes slack manifest`` reads bot identity from config.yaml."""

    def test_cli_name_flag_takes_precedence(self):
        from hermes_cli.slack_cli import slack_manifest_command

        args = argparse.Namespace(name="MyBot", description=None, write=None, slashes_only=False)
        captured = {}

        original_build = _build_full_manifest

        def capture_build(bot_name, bot_description):
            captured["name"] = bot_name
            return original_build(bot_name, bot_description)

        with patch("hermes_cli.slack_cli._build_full_manifest", side_effect=capture_build):
            with patch("hermes_cli.config.load_config", return_value={"gateway": {"bot_name": "ConfigBot"}}):
                with patch("hermes_cli.slack_cli.sys") as mock_sys:
                    mock_sys.stdout.write = lambda x: None
                    slack_manifest_command(args)

        assert captured["name"] == "MyBot"

    def test_config_bot_name_used_when_no_cli_flag(self):
        from hermes_cli.slack_cli import slack_manifest_command

        args = argparse.Namespace(name=None, description=None, write=None, slashes_only=False)
        captured = {}

        original_build = _build_full_manifest

        def capture_build(bot_name, bot_description):
            captured["name"] = bot_name
            captured["desc"] = bot_description
            return original_build(bot_name, bot_description)

        with patch("hermes_cli.slack_cli._build_full_manifest", side_effect=capture_build):
            with patch("hermes_cli.config.load_config", return_value={"gateway": {"bot_name": "Elysia"}}):
                with patch("hermes_cli.slack_cli.sys") as mock_sys:
                    mock_sys.stdout.write = lambda x: None
                    slack_manifest_command(args)

        assert captured["name"] == "Elysia"

    def test_default_hermes_when_no_config_no_flag(self):
        from hermes_cli.slack_cli import slack_manifest_command

        args = argparse.Namespace(name=None, description=None, write=None, slashes_only=False)
        captured = {}

        original_build = _build_full_manifest

        def capture_build(bot_name, bot_description):
            captured["name"] = bot_name
            return original_build(bot_name, bot_description)

        with patch("hermes_cli.slack_cli._build_full_manifest", side_effect=capture_build):
            with patch("hermes_cli.config.load_config", return_value={}):
                with patch("hermes_cli.slack_cli.sys") as mock_sys:
                    mock_sys.stdout.write = lambda x: None
                    slack_manifest_command(args)

        assert captured["name"] == "Hermes"
