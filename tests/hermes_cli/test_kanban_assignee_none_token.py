"""The shell spellings for "unassigned" must store NULL, not a literal name.

``hermes kanban assign <id> none`` has always mapped ``none``/``-``/``null`` to
NULL (``hermes_cli/kanban.py::_none_profile``), but ``create`` passed the string
straight through, so ``--assignee none`` (and the ``kanban_create`` tool, and any
script) stored the **literal** ``'none'``.

That literal is a silent dead end rather than a visible error, because it is
truthy on every path that looks for "no assignee":

  * the dispatcher's adoption branch skips it (``if not row_assignee``), so
    ``kanban.default_assignee`` never applies;
  * ``_apply_default_assignee`` only matches ``assignee IS NULL OR assignee = ''``,
    so it cannot rescue the row later either;
  * ``_dispatch_lane_task`` then asks ``profile_exists('none')``, gets False, and
    skips the row.

A ``ready`` card carrying it therefore waits forever with no worker, no error and
no event. Canonicalizing in the shared write seam keeps NULL as the one
"unassigned" value for every creator, so one code path is the whole contract.

Regression for kanban card t_074201fb.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB.

    ``Path.home`` is redirected too, so ``profile_exists`` sees only the
    special-cased ``default`` profile and no named profile dir on the real
    machine can make an unresolvable name look routable.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.mark.parametrize("raw", ["none", "NONE", "-", "null", "", "   ", "@none", " none "])
def test_unassigned_spellings_canonicalize_to_none(raw, kanban_home):
    """Every spelling of "no assignee" collapses to the one canonical value."""
    assert kb._canonical_assignee(raw) is None


@pytest.mark.parametrize("raw", ["default", "Default", "some-profile"])
def test_real_profile_names_are_still_normalized_not_dropped(raw, kanban_home):
    """The token mapping must not swallow a genuine profile name."""
    normalized = kb._canonical_assignee(raw)
    assert normalized is not None
    assert normalized != ""
    assert normalized == raw.strip().lower()


def test_create_stores_null_for_assignee_none(kanban_home):
    """The write path: ``create`` and ``assign`` must agree on NULL."""
    with kbc.connect() as conn:
        created = kb.create_task(
            conn, title="none assignee", assignee="none",
        )
        stored_create = kb.get_task(conn, created).assignee
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'created'",
            (created,),
        ).fetchone()

        assigned = kb.create_task(conn, title="assign none", assignee="default")
        kb.assign_task(conn, assigned, "none")
        stored_assign = kb.get_task(conn, assigned).assignee

    assert stored_create is None, "create must store NULL, not the literal 'none'"
    assert stored_assign is None, "assign must keep storing NULL (no regression)"
    assert '"assignee": null' in (event["payload"] or "")


def test_null_assignee_ready_row_is_adopted_by_default_assignee(kanban_home):
    """A ``none`` card must be reachable by the dispatcher's adoption path.

    With the literal stored this fails at both ends: the adoption branch reads a
    truthy assignee and never consults ``default_assignee``, and the row is then
    bucketed ``skipped_nonspawnable`` — a healthy-looking card that no worker
    can ever claim.
    """
    spawned: list[str] = []

    def spawn_stub(task, workspace, board=None):
        spawned.append(task.id)
        return None

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="none assignee dispatch", assignee="none",
        )
        res = kbd.dispatch_once(
            conn, dry_run=False, default_assignee="default", spawn_fn=spawn_stub,
        )
        after = kb.get_task(conn, tid).assignee

    assert res.skipped_nonspawnable == [], (
        "a 'none' card must not be bucketed as an unroutable lane"
    )
    assert tid in res.auto_assigned_default, "default_assignee must adopt the card"
    assert after == "default", "adoption must persist the owner, not leave it unassigned"
    assert tid in spawned, "the adopted card must actually be spawned"
