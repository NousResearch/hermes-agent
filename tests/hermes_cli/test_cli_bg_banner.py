"""The /bg success banner must not read as a reply channel for the task it just started.

The task id names a fresh session nothing can address; "continue chatting" sitting under
a ``Task ID:`` line invited users to keep talking *to* the task. The banner now says the
session is historyless, labels the id as a log reference, and names /btw as the tool that
actually shares this conversation's context.
"""

from __future__ import annotations

import pytest

from hermes_cli import cli_commands_mixin as ccm


@pytest.fixture()
def banner(monkeypatch):
    """Capture what _handle_background_command prints on the success path."""
    printed = []

    def capture(*lines):
        printed.extend(lines)

    monkeypatch.setattr(ccm, "_cp", capture)
    cli = type(
        "Cli",
        (),
        {
            "_ensure_runtime_credentials": lambda self: True,
            "_resolve_turn_agent_config": lambda self, prompt: {
                "model": "m",
                "runtime": {},
                "request_overrides": None,
            },
            "_side_worker": lambda self, produce, **kwargs: type(
                "T", (), {"start": staticmethod(lambda: None)}
            )(),
            "_handle_background_command": ccm.CLICommandsMixin._handle_background_command,
        },
    )()
    cli._background_task_counter = 0
    cli._background_tasks = {}
    cli._handle_background_command("/bg summarize the HN top stories")
    return printed


class TestBgBannerMentionsFreshSession:
    def test_banner_names_the_historyless_session(self, banner):
        assert any(
            "fresh session" in line and "no conversation history" in line
            for line in banner
        )

    def test_task_id_is_labelled_for_logs_not_as_a_handle(self, banner):
        assert any(
            "Task ID" in line and "logs" in line and "not an addressable handle" in line
            for line in banner
        )

    def test_banner_points_to_btw_for_follow_ups(self, banner):
        assert any("/btw" in line for line in banner)

    def test_banner_no_longer_reads_as_inviting_chat_with_the_task(self, banner):
        assert not any("You can continue chatting" in line for line in banner)
