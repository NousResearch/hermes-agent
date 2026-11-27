"""Slack bolt slash matcher must accept every slash the gateway can dispatch.

Regression for #101871: a skill slash (/<skill-name>) declared in the Slack manifest reached
slack_bolt with no matching listener, so bolt logged "Unhandled request", never acked, and Slack
showed "app did not respond" even though gateway dispatch resolves /<skill>.
"""

from unittest.mock import patch

from hermes_cli.commands_platforms import slack_slash_command_pattern


def test_skill_slugs_match_alongside_registry_slashes():
    with patch("agent.skill_commands.get_skill_commands",
               return_value={"/my-skill": {"name": "my-skill"}}):
        pattern = slack_slash_command_pattern()

    assert pattern.match("/my-skill")
    assert pattern.match("/hermes"), "registry slashes must still match"
    assert not pattern.match("/my-skill-extra"), "matcher must be anchored"
    assert not pattern.match("/unknown")


def test_skill_scan_failure_keeps_registry_slashes():
    """A broken skills dir must not take the whole Slack slash surface down."""
    with patch("agent.skill_commands.get_skill_commands", side_effect=RuntimeError("boom")):
        pattern = slack_slash_command_pattern()

    assert pattern.match("/hermes")
    assert not pattern.match("/my-skill")
