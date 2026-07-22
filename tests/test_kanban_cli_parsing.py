"""Tests for kanban CLI argument parsing — multi-word title support.

Regression tests for https://github.com/NousResearch/hermes-agent/issues/52696.
``/kanban create`` previously accepted only a single positional token as the
title, which broke quick-command aliases that append free-form user text
without shell-style quoting.
"""

import argparse
import pytest


@pytest.fixture()
def kanban_subparser():
    """Return a fresh ``create`` subparser with the real argument spec."""
    top = argparse.ArgumentParser(prog="hermes")
    subs = top.add_subparsers(dest="command")
    p_create = subs.add_parser("create", help="Create a new task")
    p_create.add_argument("title", nargs="+", help="Task title (multi-word OK)")
    p_create.add_argument("--body", default=None, help="Optional opening post")
    p_create.add_argument("--assignee", default=None, help="Profile name to assign")
    p_create.add_argument("--parent", action="append", default=[])
    p_create.add_argument("--workspace", default="scratch")
    p_create.add_argument("--branch", default=None)
    p_create.add_argument("--tenant", default=None)
    p_create.add_argument("--priority", type=int, default=0)
    p_create.add_argument("--triage", action="store_true")
    return top


def _join_title(args):
    """Mirror the handler logic that joins the title list."""
    return " ".join(args.title).strip() if isinstance(args.title, list) else (args.title or "").strip()


class TestKanbanCreateMultiWordTitle:
    """``kanban create`` must accept multi-word titles without quoting."""

    def test_single_word_title(self, kanban_subparser):
        args = kanban_subparser.parse_args(["create", "Draft"])
        assert _join_title(args) == "Draft"

    def test_multi_word_title(self, kanban_subparser):
        args = kanban_subparser.parse_args(["create", "Draft", "onboarding", "checklist"])
        assert _join_title(args) == "Draft onboarding checklist"

    def test_title_with_body_flag(self, kanban_subparser):
        args = kanban_subparser.parse_args(
            ["create", "Draft", "onboarding", "checklist", "--body", "Details here"]
        )
        assert _join_title(args) == "Draft onboarding checklist"
        assert args.body == "Details here"

    def test_flags_before_title(self, kanban_subparser):
        """Quick-command alias pattern: flags first, then free-form user text."""
        args = kanban_subparser.parse_args(
            ["create", "--assignee", "default", "Draft", "onboarding", "checklist"]
        )
        assert _join_title(args) == "Draft onboarding checklist"
        assert args.assignee == "default"

    def test_flags_mixed_with_title(self, kanban_subparser):
        args = kanban_subparser.parse_args(
            ["create", "--priority", "3", "Fix", "login", "bug", "--triage"]
        )
        assert _join_title(args) == "Fix login bug"
        assert args.priority == 3
        assert args.triage is True

    def test_title_with_hyphens(self, kanban_subparser):
        args = kanban_subparser.parse_args(["create", "fix-unicode-decode-error"])
        assert _join_title(args) == "fix-unicode-decode-error"
