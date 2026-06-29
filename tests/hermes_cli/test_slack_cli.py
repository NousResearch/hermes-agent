"""Tests for Slack CLI helpers."""

import argparse

from hermes_cli.slack_cli import _build_full_manifest
from hermes_cli.subcommands.slack import build_slack_parser


def _parse_slack_args(argv):
    """Build the real `hermes slack` parser and parse argv against it."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_slack_parser(subparsers, cmd_slack=lambda _args: 0)
    return parser.parse_args(argv)


class TestSlackManifestArgparse:
    """The `--no-assistant` flag wires through argparse to `no_assistant`."""

    def test_no_assistant_flag_defaults_false(self):
        args = _parse_slack_args(["slack", "manifest"])
        assert getattr(args, "no_assistant", False) is False

    def test_no_assistant_flag_sets_true(self):
        args = _parse_slack_args(["slack", "manifest", "--no-assistant"])
        assert args.no_assistant is True



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

    def test_no_assistant_omits_assistant_pieces(self):
        manifest = _build_full_manifest(
            "Hermes", "Your Hermes agent on Slack", include_assistant=False
        )

        # assistant_view feature is gone -> Slack renders a flat DM, not the
        # Assistant thread pane (where bare slash commands don't dispatch).
        assert "assistant_view" not in manifest["features"]
        assert "assistant:write" not in manifest["oauth_config"]["scopes"]["bot"]
        bot_events = manifest["settings"]["event_subscriptions"]["bot_events"]
        assert "assistant_thread_started" not in bot_events
        assert "assistant_thread_context_changed" not in bot_events

    def test_no_assistant_preserves_core_surface(self):
        """Dropping assistant mode must NOT strip the regular messaging surface."""
        manifest = _build_full_manifest(
            "Hermes", "Your Hermes agent on Slack", include_assistant=False
        )

        # Flat DM still needs the Messages tab writable.
        assert manifest["features"]["app_home"]["messages_tab_enabled"] is True
        # Slash commands and Socket Mode are independent of assistant mode.
        assert manifest["features"]["slash_commands"]
        assert manifest["settings"]["socket_mode_enabled"] is True
        # Channel + DM scopes/events survive so the bot still works everywhere.
        bot_scopes = manifest["oauth_config"]["scopes"]["bot"]
        for scope in ("commands", "channels:history", "groups:read", "im:history"):
            assert scope in bot_scopes
        bot_events = manifest["settings"]["event_subscriptions"]["bot_events"]
        for event in ("message.im", "message.channels", "message.groups", "app_mention"):
            assert event in bot_events
