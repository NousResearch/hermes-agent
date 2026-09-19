"""Unit tests for the shared reserved-item classifier.

``hermes_cli.kanban_reserved`` is the single source of truth for "is this
block reason a founder-reserved item" — imported by both the backlog drain
(``~/.hermes/scripts/drain-backlog.py``) and the kanban loop breaker
(``hermes_cli.kanban_db._route_block``) so the signal list cannot drift
between the two call sites.
"""

from __future__ import annotations

import pytest

from hermes_cli.kanban_reserved import RESERVED_SIGNAL_PATTERNS, reserved_item_signal


@pytest.mark.parametrize("reason", [
    "Blocked on ADR-0042 pending review",
    "This needs a HARD STOP until finance signs off",
    "This needs a HARD-STOP until finance signs off",
    "That field is reserved for the founder",
    "Flip strict mode on ruleset 20759088",
    "Decision already sent via telegram",
    "Awaiting reply, message_id 13409 still unanswered",
    "See docs/runbooks/release.md for the restore body",
    "This is Burak's call, not mine",
    "This is Baraks call, not mine".replace("Baraks", "Burak's"),
])
def test_reserved_item_signal_detects_each_pattern(reason: str) -> None:
    assert reserved_item_signal(reason) is not None


@pytest.mark.parametrize("reason", [
    None,
    "",
    "waiting on CI to go green",
    "the API returned a 500, retrying",
    "missing credentials for the staging database",
])
def test_reserved_item_signal_returns_none_for_non_reserved(reason) -> None:
    assert reserved_item_signal(reason) is None


def test_reserved_item_signal_is_case_insensitive() -> None:
    assert reserved_item_signal("this is RESERVED for later") is not None
    assert reserved_item_signal("blocked on Ruleset 999") is not None


def test_reserved_item_signal_returns_the_matched_text() -> None:
    signal = reserved_item_signal("blocked on ruleset 20759088 pending review")
    assert signal is not None
    assert "20759088" in signal


def test_pattern_list_has_no_duplicates() -> None:
    patterns = [p.pattern for p in RESERVED_SIGNAL_PATTERNS]
    assert len(patterns) == len(set(patterns))
