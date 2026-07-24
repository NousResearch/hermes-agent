"""Tests for stable external user-message ingress on the classic CLI."""

import queue
from types import SimpleNamespace

from cli import HermesCLI


def test_enqueue_external_user_message_puts_clean_text_on_pending_input():
    cli = SimpleNamespace(_pending_input=queue.Queue())

    accepted = HermesCLI.enqueue_external_user_message(
        cli,
        "  wake up and check the finished job  ",
        source="agent_wake",
    )

    assert accepted is True
    assert cli._pending_input.get_nowait() == "wake up and check the finished job"


def test_enqueue_external_user_message_rejects_empty_text():
    cli = SimpleNamespace(_pending_input=queue.Queue())

    accepted = HermesCLI.enqueue_external_user_message(cli, "   ", source="agent_wake")

    assert accepted is False
    assert cli._pending_input.empty()


def test_enqueue_external_user_message_initializes_missing_queue():
    cli = SimpleNamespace()

    accepted = HermesCLI.enqueue_external_user_message(cli, "hello", source="agent_wake")

    assert accepted is True
    assert cli._pending_input.get_nowait() == "hello"
