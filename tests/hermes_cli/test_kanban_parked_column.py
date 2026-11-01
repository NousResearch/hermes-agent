"""The ``parked`` column — a card nobody can run must not sit in ``ready``.

A task in ``ready`` with no assignee is invisible to the dispatcher: the
lane skips it, the board counts it as queued work, and nothing fails. These
tests pin the parked-column contract:

  * an unassigned card is parked, not left in ``ready``
  * parking happens even when the tick had no room to spawn (the card sat
    BEHIND assigned ones in lane order, so a budget break never reached it)
  * assigning a profile returns the card to the lane it came from
  * ``kanban.default_assignee`` still routes instead of parking
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def isolated_kanban_home(tmp_path, monkeypatch):
    """Fresh HERMES_HOME with a clean kanban DB."""
    test_home = tmp_path / ".hermes"
    test_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(test_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db
    yield kanban_db, test_home


def _fake_spawn(*args, **kwargs):
    """Stand-in for the real worker spawn — returns a fake PID."""
    return 12345


def test_unassigned_task_is_parked_not_left_in_ready(isolated_kanban_home):
    """No assignee and no default_assignee: the card must not sit in ``ready``.

    Before the fix it stayed in ``ready`` forever, and the board reported
    ``ready=N`` as if that work could run.
    """
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
        # Force the legacy shape: a card that is already IN ready with no
        # assignee (the state the ticket measured on the board).
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
        status = kb.get_task(conn, task_id).status
    assert status == "parked", f"unassigned card must be parked, got {status}"
    assert res.parked_unassigned == [task_id]
    assert not res.skipped_unassigned
    assert not res.spawned


def test_parked_card_returns_to_ready_when_assigned(isolated_kanban_home):
    """Assigning a profile is what routes the card — it must come back."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kbc.connect_closing() as conn:
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
        assert kb.get_task(conn, task_id).status == "parked"
        kb.assign_task(conn, task_id, "default")
        assert kb.get_task(conn, task_id).status == "ready"


def test_parked_card_is_dispatched_after_assignment(isolated_kanban_home):
    """A parked card that gets assigned actually runs."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kbc.connect_closing() as conn:
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
        kb.assign_task(conn, task_id, "default")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    assert any(s[0] == task_id for s in res.spawned)


def test_default_assignee_routes_instead_of_parking(isolated_kanban_home):
    """The fallback keeps working: a routable card is never parked."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False, default_assignee="default",
        )
        status = kb.get_task(conn, task_id).status
    assert res.auto_assigned_default == [task_id]
    assert not res.parked_unassigned
    assert status == "running"


def test_unassigned_card_is_parked_even_when_the_budget_is_full(isolated_kanban_home):
    """The lane is ordered by priority: an unassigned card can sit BEHIND
    assigned ones and never be reached when the budget breaks out early.

    The parking sweep must not be gated on the spawn budget — otherwise a
    busy board keeps advertising unrunnable work as ``ready`` forever.
    """
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        # Higher priority + assigned: this one fills the budget.
        busy_id = kb.create_task(conn, title="busy", assignee="default", priority=100)
        # Lower priority + unassigned: it sits BEHIND the busy card in lane
        # order, so a budget break would never reach it.
        parked_id = kb.create_task(conn, title="parked", assignee=None, priority=1)
        # Force the legacy shape (already in ready, no assignee).
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (parked_id,))
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False, max_spawn=1,
        )
        parked_status = kb.get_task(conn, parked_id).status
        busy_status = kb.get_task(conn, busy_id).status
    assert busy_status == "running"
    assert parked_status == "parked", (
        f"unassigned card behind the budget must still be parked, got {parked_status}"
    )
    assert res.parked_unassigned == [parked_id]


def test_unassigned_review_card_is_parked(isolated_kanban_home):
    """The review lane had the same defect: an unassigned review card was
    skipped in silence, forever."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
        # Move it to review by hand (no assignee on it).
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
        status = kb.get_task(conn, task_id).status
    assert status == "parked", f"unassigned review card must be parked, got {status}"
    assert res.parked_unassigned == [task_id]


def test_parked_review_card_returns_to_review_when_assigned(isolated_kanban_home):
    """A card parked out of ``review`` still needs a reviewer, not a worker."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
    with kbc.connect_closing() as conn:
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
        assert kb.get_task(conn, task_id).status == "parked"
        kb.assign_task(conn, task_id, "default")
        assert kb.get_task(conn, task_id).status == "review"


def test_created_unassigned_card_is_parked_by_the_dispatcher(isolated_kanban_home):
    """Creating a card with no assignee must not leave it advertised as runnable.

    Creation still lands in ``ready`` (creation is not the defect), but the
    dispatcher's very next tick parks it: the card becomes visible as
    "queued, nobody can run it" instead of inflating the ready count.
    """
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        unassigned = kb.create_task(conn, title="no owner", assignee=None)
        assigned = kb.create_task(conn, title="owned", assignee="default")
        assert kb.get_task(conn, unassigned).status == "ready"
        assert kb.get_task(conn, assigned).status == "ready"
    with kbc.connect_closing() as conn:
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
        assert kb.get_task(conn, unassigned).status == "parked"
        assert kb.get_task(conn, assigned).status == "running"
