"""An approval panel that just appeared ignores keys that would approve it."""

import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from cli import HermesCLI


def _cli_with_fresh_prompt(shown_at):
    cli = HermesCLI.__new__(HermesCLI)
    cli._approval_lock = threading.Lock()
    cli._invalidate = MagicMock()
    cli._approval_state = {
        "command": "rm -rf build/", "description": "recursive delete",
        "choices": ["once", "session", "always", "deny"], "selected": 0,
        "response_queue": queue.Queue(), "shown_at": shown_at,
    }
    return cli


def test_early_keys_cannot_approve_but_deny_and_later_keys_act():
    event = SimpleNamespace(app=SimpleNamespace(invalidate=MagicMock()))
    cli = _cli_with_fresh_prompt(time.monotonic())
    answers = cli._approval_state["response_queue"]

    # A composer Enter / digit landing with the paint: no answer, and the highlight does not move.
    cli._handle_approval_selection()
    cli._tui_make_approval_number_handler(2)(event)
    assert answers.empty() and cli._approval_state["selected"] == 0

    # Deny is never delayed.
    cli._tui_make_approval_number_handler(3)(event)
    assert answers.get_nowait() == "deny"

    # Once the prompt has been on screen past the grace window, Enter approves as before.
    cli = _cli_with_fresh_prompt(time.monotonic() - 1.0)
    answers = cli._approval_state["response_queue"]
    cli._handle_approval_selection()
    assert answers.get_nowait() == "once"
