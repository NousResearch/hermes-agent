"""Charter §4 estimation at the mint path — est-mintpath-20260911.

Every card must leave a mint path carrying a points estimate. Measured
2026-09-11: 21/241 cards (8.7%). The 2026-09-04 fix edited SOUL prose and moved
nothing, because ~70% of cards are minted by the machine path, which never
reads a SOUL. These tests pin the mechanism at the mint path itself.

The number travels as a ``points-estimate: N`` COMMENT, not a column: that is
what ``scripts/cost-ledger.py`` (``PTS_RE``) parses and therefore what the
charter §2 ``points_coverage`` metric counts. A test that only asserted a
column would pass while the metric stayed flat.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_graph import decompose_triage_task

# The ledger's parser, copied verbatim from scripts/cost-ledger.py. If this
# stops matching, the metric stops moving — that is the failure this guards.
PTS_RE = re.compile(r"points-estimate[^0-9]*([0-9]+)", re.I)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _ledger_points(conn, tid: str):
    """Exactly what cost-ledger._card_meta computes for ``points``."""
    value = None
    for (body,) in conn.execute(
        "SELECT body FROM task_comments WHERE task_id=? ORDER BY id", (tid,)
    ).fetchall():
        m = PTS_RE.search(body or "")
        if m:
            value = int(m.group(1))  # last one wins
    return value


def test_create_task_writes_a_placeholder_when_no_estimate_given(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no estimate supplied", assignee="bob")
        assert _ledger_points(conn, tid) == kb.AUTO_POINTS_PLACEHOLDER

        bodies = [c.body for c in kb.list_comments(conn, tid)]
        assert any("auto-points" in b for b in bodies), (
            "the placeholder must say it is a placeholder, or the next reader "
            "mistakes it for a real estimate"
        )


def test_explicit_points_are_recorded_and_write_no_placeholder(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="estimated", assignee="bob", points=5)
        assert _ledger_points(conn, tid) == 5
        assert not any("auto-points" in c.body for c in kb.list_comments(conn, tid))


def test_points_below_one_is_rejected(kanban_home):
    """0 is falsy, so a 0 would read as 'no estimate' in the coverage metric."""
    with kbc.connect() as conn:
        for bad in (0, -1):
            with pytest.raises(ValueError):
                kb.create_task(conn, title="bad", assignee="bob", points=bad)


def test_a_later_estimate_comment_wins_over_the_placeholder(kanban_home):
    """The specifier replaces the placeholder by posting the real estimate."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="specified later", assignee="bob")
        assert _ledger_points(conn, tid) == kb.AUTO_POINTS_PLACEHOLDER
        kb.add_comment(conn, tid, "jobsy", "points-estimate: 8 — scoped")
        assert _ledger_points(conn, tid) == 8


def test_every_mint_path_covers_including_holds_and_triage(kanban_home):
    """Held cards and triage cards are cards; the machine path is the point."""
    with kbc.connect() as conn:
        held = kb.create_task(
            conn, title="deploy", assignee="bob", initial_status="blocked",
            block_kind="operator_hold",
        )
        triaged = kb.create_task(conn, title="rough idea", assignee=None, triage=True)
        for tid in (held, triaged):
            assert _ledger_points(conn, tid) == kb.AUTO_POINTS_PLACEHOLDER


def test_decomposed_children_get_the_placeholder_too(kanban_home):
    """The auto-decomposer is the ~70% path the 2026-09-04 SOUL edit missed."""
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="big brief", assignee="jobsy", triage=True)
        kids = decompose_triage_task(
            conn, root, root_assignee="jobsy",
            children=[
                {"title": "build it", "assignee": "bob"},
                {"title": "review it", "assignee": "rodge"},
            ],
        )
        assert kids, "fan-out produced no children"
        for kid in kids:
            assert _ledger_points(conn, kid) == kb.AUTO_POINTS_PLACEHOLDER


def test_no_card_is_minted_without_an_estimate(kanban_home):
    """The coverage number itself: every card, whatever minted it."""
    with kbc.connect() as conn:
        made = [
            kb.create_task(conn, title="a", assignee="bob"),
            kb.create_task(conn, title="b", assignee="karl", triage=True),
            kb.create_task(conn, title="c", assignee="bob", initial_status="blocked",
                           block_kind="operator_hold"),
            kb.create_task(conn, title="d", assignee="steve-o", points=3),
        ]
        root = kb.create_task(conn, title="e", assignee="jobsy", triage=True)
        made += decompose_triage_task(
            conn, root, root_assignee="jobsy",
            children=[{"title": "f", "assignee": "bob"}],
        ) or []
        covered = [t for t in made if _ledger_points(conn, t)]
        assert len(covered) == len(made), f"{len(covered)}/{len(made)} covered"
