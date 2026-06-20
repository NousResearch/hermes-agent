"""Multi-tenant Slack namespacing (sadee-and-co ADR-0037): a per-instance command
name + a single-command mode, so multiple Hermes agents can share one Slack
workspace without colliding on slash commands (which are a workspace-global,
single-owner namespace — unlike Telegram's per-bot or Discord's per-app)."""
from __future__ import annotations

from hermes_cli.commands import (
    slack_app_manifest,
    slack_command_name,
    slack_native_slashes,
)


def test_env_overrides_and_sanitizes_command_name(monkeypatch):
    monkeypatch.setenv("HERMES_SLACK_COMMAND", "Welchman!")  # uppercase + invalid char
    assert slack_command_name() == "welchman"


def test_default_mode_exposes_the_full_command_set(monkeypatch):
    monkeypatch.delenv("HERMES_SLACK_SINGLE_COMMAND", raising=False)
    monkeypatch.setenv("HERMES_SLACK_COMMAND", "hermes")
    slashes = slack_native_slashes()
    assert slashes[0][0] == "hermes"  # catch-all reserved first
    assert len(slashes) > 1  # plus individual commands (single-tenant behaviour preserved)


def test_single_command_mode_exposes_only_the_handle(monkeypatch):
    monkeypatch.setenv("HERMES_SLACK_COMMAND", "welchman")
    monkeypatch.setenv("HERMES_SLACK_SINGLE_COMMAND", "1")
    slashes = slack_native_slashes()
    assert slashes == [("welchman", "Talk to Hermes or run a subcommand", "[subcommand] [args]")]
    # the generated manifest then declares exactly one command — nothing to collide
    manifest = slack_app_manifest()["features"]["slash_commands"]
    assert [c["command"] for c in manifest] == ["/welchman"]
