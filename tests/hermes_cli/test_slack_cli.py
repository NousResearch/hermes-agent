"""Tests for Slack CLI helpers."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli.slack_cli import _build_full_manifest, slack_manifest_command
from hermes_cli.subcommands.slack import build_slack_parser
from hermes_cli import main_platform_setup


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


class TestSlackManifestArgparse:
    """Slack manifest messaging-experience flags wire through argparse."""




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






    def test_assistant_features_remain_enabled(self):
        manifest = _build_full_manifest("Hermes", "Your Hermes agent on Slack")

        assert "assistant_view" in manifest["features"]
        assert "agent_view" not in manifest["features"]
        assert "assistant:write" in manifest["oauth_config"]["scopes"]["bot"]
        bot_events = manifest["settings"]["event_subscriptions"]["bot_events"]
        assert "assistant_thread_started" in bot_events




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


