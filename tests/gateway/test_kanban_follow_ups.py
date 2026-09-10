"""Follow-up handoffs survive from a worker's terminal call to the human's notification.

A worker's most perishable output is the work it did NOT do: a decision left to a
human, a manual step still required, a gap it deliberately skipped. Before this,
``summary`` was rendered first-line-only and ``metadata`` was never read by the
notifier at all, so that content reached a person only if they independently went
and read the board -- which is what the notification exists to make unnecessary.

These are behaviour contracts between the DB writer and the notifier renderer, not
snapshots of either one's current text.
"""

from __future__ import annotations

import pytest

from gateway.kanban_watchers_notifier import _follow_ups_block
from hermes_cli.kanban_db import FOLLOW_UPS_MAX
from hermes_cli.kanban_db import extract_follow_ups


class _Ev:
    """Minimal stand-in for a task event row (the notifier reads ``.payload``)."""

    def __init__(self, payload):
        self.payload = payload


# --- the writer half: what lands in the event payload ---


def test_follow_ups_survive_extraction():
    assert extract_follow_ups({"follow_ups": ["Merge PR #174", "Resync deploy"]}) == [
        "Merge PR #174",
        "Resync deploy",
    ]


def test_a_bare_string_is_accepted_as_one_item():
    """Models emit a scalar where a list is documented; degrade, don't drop."""
    assert extract_follow_ups({"follow_ups": "Merge PR #174"}) == ["Merge PR #174"]


def test_absent_or_malformed_metadata_yields_no_follow_ups():
    for meta in (None, {}, {"follow_ups": None}, {"follow_ups": 42}, "not a dict"):
        assert extract_follow_ups(meta) == []


def test_non_string_items_are_dropped_not_coerced():
    """Rendering ``None`` or ``{}`` at a human is worse than showing nothing."""
    assert extract_follow_ups({"follow_ups": ["real", None, 7, {"a": 1}, "also real"]}) == [
        "real",
        "also real",
    ]


def test_follow_ups_are_bounded():
    """A notification is a pointer to the board, not a replacement for it."""
    many = [f"item {i}" for i in range(FOLLOW_UPS_MAX + 10)]
    assert len(extract_follow_ups({"follow_ups": many})) == FOLLOW_UPS_MAX


def test_multiline_items_are_flattened_to_one_line():
    """Each entry occupies one bullet; an embedded newline must not forge more."""
    out = extract_follow_ups({"follow_ups": ["first line\nsmuggled second"]})
    assert len(out) == 1
    assert "\n" not in out[0]


def test_blank_items_do_not_become_empty_bullets():
    assert extract_follow_ups({"follow_ups": ["", "   ", "\n", "real"]}) == ["real"]


# --- the reader half: what the notifier renders ---


def test_the_notifier_renders_what_the_writer_stored():
    """The contract that matters: writer output is readable by the renderer."""
    stored = extract_follow_ups({"follow_ups": ["Merge PR #174", "Resync deploy"]})
    block = _follow_ups_block(_Ev({"follow_ups": stored}))
    assert "Merge PR #174" in block
    assert "Resync deploy" in block


def test_no_follow_ups_renders_nothing_at_all():
    """An empty section header would be noise on every routine completion."""
    for payload in ({}, {"follow_ups": []}, {"follow_ups": None}, {"summary": "x"}):
        assert _follow_ups_block(_Ev(payload)) == ""


def test_a_legacy_event_without_the_key_renders_nothing():
    """Events written before this feature must not break the notifier."""
    assert _follow_ups_block(_Ev({"result_len": 0, "summary": "old row"})) == ""


def test_the_block_is_labelled_so_a_human_can_see_it_is_for_them():
    block = _follow_ups_block(_Ev({"follow_ups": ["Merge PR #174"]}))
    assert "Needs you" in block


@pytest.mark.parametrize("bad_payload", [None, {"follow_ups": "a string"}, {"follow_ups": 3}])
def test_the_renderer_tolerates_payloads_the_writer_would_not_produce(bad_payload):
    """Hand-written rows and other producers must not crash a notifier tick."""
    assert _follow_ups_block(_Ev(bad_payload)) == ""
