"""Tests for Slack CLI helpers."""

import argparse

from hermes_cli.main import cmd_slack
from hermes_cli.slack_cli import _build_full_manifest, _slack_ingress_settings
from hermes_cli.subcommands.slack import build_slack_parser


def _parse_slack_args(argv):
    """Build the real `hermes slack` parser and parse argv against it."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_slack_parser(subparsers, cmd_slack=lambda _args: 0)
    return parser.parse_args(argv)


class TestSlackIngressArgparse:
    def test_ingress_defaults_to_loopback(self):
        args = _parse_slack_args(["slack", "ingress"])
        assert args.slack_command == "ingress"
        assert args.host == "127.0.0.1"
        assert args.port == 8791
        assert args.state is None

    def test_ingress_accepts_operational_overrides(self):
        args = _parse_slack_args(
            [
                "slack",
                "ingress",
                "--host",
                "localhost",
                "--port",
                "9001",
                "--state",
                "/tmp/slack.db",
            ]
        )
        assert args.host == "localhost"
        assert args.port == 9001
        assert args.state == "/tmp/slack.db"

    def test_ingress_policy_settings_come_from_config(self):
        settings = _slack_ingress_settings(
            {
                "slack": {
                    "ingress": {
                        "follow_ttl_days": 7,
                        "max_followed_threads": 321,
                        "reaction_user_ids": ["U_OWNER", "U_OWNER", ""],
                        "reaction_names": [":eyes:", "bookmark"],
                    }
                }
            }
        )

        assert settings == {
            "ttl_seconds": 7 * 24 * 60 * 60,
            "max_threads": 321,
            "reaction_user_ids": {"U_OWNER"},
            "reaction_names": {"eyes", "bookmark"},
        }

    def test_ingress_policy_settings_have_bounded_defaults(self):
        settings = _slack_ingress_settings({})
        assert settings["ttl_seconds"] == 30 * 24 * 60 * 60
        assert settings["max_threads"] == 10_000
        assert settings["reaction_user_ids"] == set()
        assert settings["reaction_names"] == set()

    def test_ingress_dispatches_to_runtime_command(self, monkeypatch):
        import hermes_cli.slack_cli as slack_cli

        called = []
        monkeypatch.setattr(
            slack_cli,
            "slack_ingress_command",
            lambda args: called.append(args) or 17,
        )
        args = _parse_slack_args(["slack", "ingress"])

        assert cmd_slack(args) == 17
        assert called == [args]

    def test_ingress_refuses_direct_slack_gateway_ownership(
        self, monkeypatch, capsys
    ):
        import gateway.config as gateway_config_module
        import hermes_cli.config as cli_config
        import hermes_cli.slack_cli as slack_cli
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        monkeypatch.setattr(cli_config, "load_config", lambda: {})
        monkeypatch.setattr(
            gateway_config_module,
            "load_gateway_config",
            lambda: GatewayConfig(
                platforms={
                    Platform.SLACK: PlatformConfig(
                        enabled=True,
                        token="xoxb-test",
                    )
                }
            ),
        )
        monkeypatch.setattr(
            slack_cli.asyncio,
            "run",
            lambda coroutine: coroutine.close(),
        )

        assert slack_cli.slack_ingress_command(
            _parse_slack_args(["slack", "ingress"])
        ) == 1
        assert "slack.enabled: false" in capsys.readouterr().err


class TestSlackManifestArgparse:
    """Slack manifest messaging-experience flags wire through argparse."""

    def test_no_assistant_flag_defaults_false(self):
        args = _parse_slack_args(["slack", "manifest"])
        assert getattr(args, "no_assistant", False) is False

    def test_no_assistant_flag_sets_true(self):
        args = _parse_slack_args(["slack", "manifest", "--no-assistant"])
        assert args.no_assistant is True

    def test_agent_view_flag_defaults_false(self):
        args = _parse_slack_args(["slack", "manifest"])
        assert getattr(args, "agent_view", False) is False

    def test_agent_view_flag_sets_true(self):
        args = _parse_slack_args(["slack", "manifest", "--agent-view"])
        assert args.agent_view is True


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

    def test_group_dm_scopes_and_event_are_included(self):
        """Group DMs (mpim) need message.mpim + mpim:history or Slack never
        delivers them — the adapter classifies mpim as a DM and replies
        ambiently, but only if the event reaches the bot at all."""
        manifest = _build_full_manifest("Hermes", "Your Hermes agent on Slack")

        bot_scopes = manifest["oauth_config"]["scopes"]["bot"]
        bot_events = manifest["settings"]["event_subscriptions"]["bot_events"]

        # The event is the load-bearing piece: without message.mpim Slack
        # drops group-DM messages before the adapter sees them.
        assert "message.mpim" in bot_events
        # mpim:history is the scope message.mpim requires (per Slack docs);
        # mpim:read mirrors im:read for conversations.info classification.
        assert "mpim:history" in bot_scopes
        assert "mpim:read" in bot_scopes

    def test_reaction_trigger_subscription_is_included(self):
        manifest = _build_full_manifest("Hermes", "Your Hermes agent on Slack")

        bot_scopes = manifest["oauth_config"]["scopes"]["bot"]
        bot_events = manifest["settings"]["event_subscriptions"]["bot_events"]
        assert "reactions:read" in bot_scopes
        assert "reaction_added" in bot_events

    def test_group_dm_surface_present_without_assistant_mode(self):
        """Dropping assistant mode must not strip the group-DM surface."""
        manifest = _build_full_manifest(
            "Hermes", "Your Hermes agent on Slack", include_assistant=False
        )

        bot_scopes = manifest["oauth_config"]["scopes"]["bot"]
        bot_events = manifest["settings"]["event_subscriptions"]["bot_events"]
        assert "message.mpim" in bot_events
        assert "mpim:history" in bot_scopes

    def test_assistant_features_remain_enabled(self):
        manifest = _build_full_manifest("Hermes", "Your Hermes agent on Slack")

        assert "assistant_view" in manifest["features"]
        assert "agent_view" not in manifest["features"]
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
        assert "agent_view" not in manifest["features"]
        assert "assistant:write" not in manifest["oauth_config"]["scopes"]["bot"]
        bot_events = manifest["settings"]["event_subscriptions"]["bot_events"]
        assert "assistant_thread_started" not in bot_events
        assert "assistant_thread_context_changed" not in bot_events

    def test_agent_view_uses_agent_manifest_surface(self):
        manifest = _build_full_manifest(
            "Hermes",
            "Your Hermes agent on Slack",
            messaging_experience="agent",
        )

        assert manifest["features"]["agent_view"] == {
            "agent_description": "Chat with Hermes in Slack Messages.",
        }
        assert "assistant_view" not in manifest["features"]
        assert "assistant:write" in manifest["oauth_config"]["scopes"]["bot"]

    def test_agent_view_uses_agent_event_subscriptions(self):
        manifest = _build_full_manifest(
            "Hermes",
            "Your Hermes agent on Slack",
            messaging_experience="agent",
        )

        bot_events = manifest["settings"]["event_subscriptions"]["bot_events"]
        assert "app_home_opened" in bot_events
        assert "app_context_changed" in bot_events
        assert "message.im" in bot_events
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
