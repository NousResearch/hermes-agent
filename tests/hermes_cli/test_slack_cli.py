"""Tests for Slack CLI helpers."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli.slack_cli import _build_full_manifest, slack_manifest_command
from hermes_cli.subcommands.slack import build_slack_parser


def _parse_slack_args(argv):
    """Build the real `hermes slack` parser and parse argv against it."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_slack_parser(subparsers, cmd_slack=lambda _args: 0)
    return parser.parse_args(argv)


def _run_console_entrypoint(*argv: str) -> subprocess.CompletedProcess[str]:
    """Run the packaged console-script contract in a fresh interpreter."""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from hermes_cli.main import main; raise SystemExit(main())",
            *argv,
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_slack_dispatcher_propagates_manifest_failure(monkeypatch):
    from hermes_cli import main as main_module
    from hermes_cli import slack_cli

    monkeypatch.setattr(slack_cli, "slack_manifest_command", lambda _args: 2)

    with pytest.raises(SystemExit) as exc_info:
        main_module.cmd_slack(argparse.Namespace(slack_command="manifest"))

    assert exc_info.value.code == 2


class TestSlackManifestConsoleExitStatus:
    """The packaged CLI must expose manifest validation failures to shells."""

    def test_too_short_long_description_exits_two(self):
        result = _run_console_entrypoint(
            "slack", "manifest", "--long-description", "x" * 174
        )

        assert result.returncode == 2
        assert result.stdout == ""
        assert "at least 175 characters" in result.stderr

    def test_missing_long_description_file_exits_two(self, tmp_path):
        missing = tmp_path / "missing.md"
        result = _run_console_entrypoint(
            "slack", "manifest", "--long-description-file", str(missing)
        )

        assert result.returncode == 2
        assert result.stdout == ""
        assert "cannot read long description" in result.stderr

    def test_slashes_only_conflict_exits_two(self):
        result = _run_console_entrypoint(
            "slack",
            "manifest",
            "--slashes-only",
            "--long-description",
            "x" * 175,
        )

        assert result.returncode == 2
        assert result.stdout == ""
        assert "cannot be used with --slashes-only" in result.stderr


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

    def test_long_description_file_preserves_newlines(self, tmp_path, capsys):
        content = ("x" * 175) + "\r\n" + ("y" * 175) + "\r"
        source = tmp_path / "AGENTS.md"
        source.write_bytes(content.encode("utf-8"))
        args = _parse_slack_args(
            ["slack", "manifest", "--long-description-file", str(source)]
        )

        assert slack_manifest_command(args) == 0

        manifest = json.loads(capsys.readouterr().out)
        assert manifest["display_information"]["long_description"] == content

    def test_long_description_accepts_inline_text(self, capsys):
        content = "x" * 175
        args = _parse_slack_args(
            ["slack", "manifest", "--long-description", content]
        )

        assert slack_manifest_command(args) == 0

        manifest = json.loads(capsys.readouterr().out)
        assert manifest["display_information"]["long_description"] == content

    def test_long_description_rejects_fewer_than_175_characters(self, capsys):
        args = _parse_slack_args(
            ["slack", "manifest", "--long-description", "x" * 174]
        )

        assert slack_manifest_command(args) == 2

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "at least 175 characters" in captured.err

    def test_long_description_rejects_more_than_4000_characters(self, capsys):
        args = _parse_slack_args(
            ["slack", "manifest", "--long-description", "x" * 4001]
        )

        assert slack_manifest_command(args) == 2

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "4000 characters" in captured.err

    def test_long_description_options_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exc_info:
            _parse_slack_args(
                [
                    "slack",
                    "manifest",
                    "--long-description",
                    "inline",
                    "--long-description-file",
                    "AGENTS.md",
                ]
            )

        assert exc_info.value.code == 2

    @pytest.mark.parametrize(
        ("option", "value"),
        [
            ("--long-description", "x" * 175),
            ("--long-description-file", "missing.md"),
        ],
    )
    def test_long_description_options_reject_slashes_only(
        self, option, value, capsys
    ):
        args = _parse_slack_args(
            ["slack", "manifest", "--slashes-only", option, value]
        )

        assert slack_manifest_command(args) == 2

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "cannot be used with --slashes-only" in captured.err

    def test_long_description_file_reports_read_errors(self, tmp_path, capsys):
        missing = tmp_path / "missing.md"
        args = _parse_slack_args(
            ["slack", "manifest", "--long-description-file", str(missing)]
        )

        assert slack_manifest_command(args) == 2

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "cannot read long description" in captured.err

    @pytest.mark.parametrize(
        ("length", "expected_status"),
        [(174, 2), (175, 0), (4000, 0), (4001, 2)],
    )
    def test_long_description_file_enforces_slack_length_boundaries(
        self, tmp_path, capsys, length, expected_status
    ):
        content = "x" * length
        source = tmp_path / "AGENTS.md"
        source.write_text(content, encoding="utf-8")
        args = _parse_slack_args(
            ["slack", "manifest", "--long-description-file", str(source)]
        )

        assert slack_manifest_command(args) == expected_status

        captured = capsys.readouterr()
        if expected_status == 0:
            manifest = json.loads(captured.out)
            assert manifest["display_information"]["long_description"] == content
            assert captured.err == ""
        else:
            assert captured.out == ""
            assert "long description must be" in captured.err

    def test_long_description_file_reports_invalid_utf8(self, tmp_path, capsys):
        source = tmp_path / "AGENTS.md"
        source.write_bytes(b"\xff" * 175)
        args = _parse_slack_args(
            ["slack", "manifest", "--long-description-file", str(source)]
        )

        assert slack_manifest_command(args) == 2

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "cannot read long description" in captured.err

    def test_long_description_file_reports_tilde_expansion_errors(
        self, monkeypatch, capsys
    ):
        source = "~hermes-user-that-does-not-exist-20260716/AGENTS.md"

        def fail_expanduser(_path):
            raise RuntimeError("home directory unavailable")

        monkeypatch.setattr(Path, "expanduser", fail_expanduser)
        args = _parse_slack_args(
            ["slack", "manifest", "--long-description-file", source]
        )

        assert slack_manifest_command(args) == 2

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "cannot read long description" in captured.err
        assert source in captured.err


class TestSlackFullManifest:
    """Generated full Slack app manifest used by `hermes slack manifest`."""

    def test_long_description_is_included_without_truncation(self):
        long_description = "# Agent policy\n\n" + ("x" * 3984)

        manifest = _build_full_manifest(
            "Hermes",
            "Your Hermes agent on Slack",
            long_description=long_description,
        )

        assert manifest["display_information"]["long_description"] == long_description
        assert len(long_description) == 4000

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

    def test_reaction_scope_and_event_included(self):
        """reaction_added/removed events + reactions:read scope must be in the
        manifest so the adapter can forward reactions into the message
        pipeline and gateway hooks."""
        manifest = _build_full_manifest("Hermes", "Your Hermes agent on Slack")

        bot_scopes = manifest["oauth_config"]["scopes"]["bot"]
        bot_events = manifest["settings"]["event_subscriptions"]["bot_events"]
        assert "reactions:read" in bot_scopes
        assert "reaction_added" in bot_events
        assert "reaction_removed" in bot_events

    def test_reaction_scope_survives_no_assistant(self):
        manifest = _build_full_manifest(
            "Hermes", "Your Hermes agent on Slack", include_assistant=False
        )
        bot_scopes = manifest["oauth_config"]["scopes"]["bot"]
        bot_events = manifest["settings"]["event_subscriptions"]["bot_events"]
        assert "reactions:read" in bot_scopes
        assert "reaction_added" in bot_events
