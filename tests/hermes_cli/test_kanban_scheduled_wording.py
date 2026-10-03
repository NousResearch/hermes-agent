"""The ``scheduled`` kanban status has no timer or due-date release path —
only a human ``unblock_task`` call re-gates it. Regression for #124395:
the docstring and help text used to claim a time-based release that does
not exist anywhere in the dispatcher or schema.
"""

from __future__ import annotations

import argparse

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_parser import build_parser


def _schedule_help_text() -> str:
    wrap = argparse.ArgumentParser(prog="test-wrap", add_help=False)
    kanban_parser = build_parser(wrap.add_subparsers(dest="_top"))
    subparsers_action = next(
        a for a in kanban_parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    pseudo_action = next(
        a for a in subparsers_action._choices_actions if a.dest == "schedule"
    )
    return pseudo_action.help


def test_schedule_task_docstring_does_not_claim_a_timer():
    doc = kb.schedule_task.__doc__ or ""
    assert "waiting on time" not in doc
    assert "unblock_task" in doc


def test_schedule_cli_help_does_not_claim_a_timer():
    assert "waiting on time" not in _schedule_help_text()


def test_slash_kanban_help_does_not_claim_time_delay_work():
    assert "time-delay work" not in kc._SLASH_KANBAN_HELP
