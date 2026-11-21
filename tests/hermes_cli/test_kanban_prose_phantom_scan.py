"""Regression for the advisory prose scan reading code examples as card refs.

The completion-time scan (:func:`hermes_cli.kanban_db._scan_prose_for_phantom_ids`)
looks for ``t_<hex>`` ids in a completion's summary/result and flags the ones
that do not resolve. Probe write-ups and review notes quote example commands
(``hermes kanban complete t_dead0001``) and branch names (``fix/t_66a9d5c0``);
those tokens name a placeholder or a ref, never a card, and on a real board they
produced 26 of 28 ``prose_phantom_refs`` warnings for ids that were never meant
to exist.

Both directions are load-bearing: code spans, fenced blocks and path/branch
tails must not be flagged, and a plain-prose reference to an unknown id — the
thing the rule exists to catch — still must be.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _phantom_refs(text: str) -> list[str]:
    conn = kbc.connect()
    try:
        return kb._scan_prose_for_phantom_ids(conn, text)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Code examples are quoted material, not references
# ---------------------------------------------------------------------------


def test_inline_code_span_id_is_not_flagged(kanban_home):
    text = "Verified with `hermes kanban complete t_dead0001 --summary done`."
    assert _phantom_refs(text) == []


def test_double_backtick_span_id_is_not_flagged(kanban_home):
    text = "The form was ``hermes kanban show t_dead0009`` in the probe."
    assert _phantom_refs(text) == []


def test_fenced_block_ids_are_not_flagged(kanban_home):
    text = (
        "Replayed the sequence:\n"
        "\n"
        "```sh\n"
        "hermes kanban complete t_dead0001\n"
        "sh -c 'hermes kanban show t_dead0009'\n"
        "```\n"
        "\n"
        "Both were placeholders.\n"
    )
    assert _phantom_refs(text) == []


def test_unclosed_fence_swallows_the_rest(kanban_home):
    """A truncated fence is still code; nothing after it is prose."""
    text = "Command used:\n~~~bash\nhermes kanban complete t_dead0002\n"
    assert _phantom_refs(text) == []


# ---------------------------------------------------------------------------
# Path and branch tokens: a ``t_<hex>`` tail names a file or a ref
# ---------------------------------------------------------------------------


def test_branch_and_worktree_tails_are_not_flagged(kanban_home):
    text = (
        "Landed on branch fix/t_66a9d5c0 in worktree wt/t_ffd2508c, "
        r"log at C:\tmp\t_dead0002\out.txt and /var/log/t_dead0003.log."
    )
    assert _phantom_refs(text) == []


def test_url_tail_is_not_flagged(kanban_home):
    text = "See https://github.com/NousResearch/hermes-agent/pull/t_66a9d5c0 for the diff."
    assert _phantom_refs(text) == []


# ---------------------------------------------------------------------------
# The rule's real job: plain-prose ids that do not resolve
# ---------------------------------------------------------------------------


def test_plain_prose_unknown_id_is_still_flagged(kanban_home):
    text = "Filed as t_deadbeef in the board, and the follow-up is t_deadbeef again."
    assert _phantom_refs(text) == ["t_deadbeef"]


def test_prose_id_next_to_a_code_span_is_still_flagged(kanban_home):
    """Stripping code must not swallow the prose around it."""
    text = "Ran `hermes kanban show t_dead0001`, then filed t_deadbeef for the fallout."
    assert _phantom_refs(text) == ["t_deadbeef"]


def test_resolvable_prose_id_is_not_flagged(kanban_home):
    conn = kbc.connect()
    try:
        real = kb.create_task(conn, title="real", assignee="w")
    finally:
        conn.close()
    assert _phantom_refs(f"Cross-reference: {real} carries the detail.") == []


def test_completion_with_only_code_example_ids_flags_nothing(kanban_home):
    """The end-to-end contract: no ``suspected_hallucinated_references`` event
    for a summary whose only ``t_<hex>`` tokens are quoted examples."""
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="probe replay", assignee="w")
        kb.complete_task(
            conn, task_id,
            summary="Replayed `hermes kanban complete t_dead0001`; branch fix/t_66a9d5c0.",
        )
        flagged = conn.execute(
            "SELECT COUNT(*) FROM task_events"
            " WHERE task_id = ? AND kind = 'suspected_hallucinated_references'",
            (task_id,),
        ).fetchone()[0]
    finally:
        conn.close()
    assert flagged == 0


def test_completion_with_a_prose_phantom_still_flags(kanban_home):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="prose phantom", assignee="w")
        kb.complete_task(conn, task_id, summary="Follow-up lives on t_deadbeef.")
        row = conn.execute(
            "SELECT payload FROM task_events"
            " WHERE task_id = ? AND kind = 'suspected_hallucinated_references'",
            (task_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert "t_deadbeef" in row[0]
