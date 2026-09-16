"""Assignee ingress: the display spelling must round-trip to a routable value, and
a name the dispatcher can never spawn for must be refused at the writer.

Regression for kanban card t_9080ae81. Notifications render the assignee as
``@name`` (``gateway/kanban_watchers_notifier`` attribution tag); an agent read
its own notification, passed ``@default`` back into card creation, and the value
was stored verbatim. The dispatcher resolves assignees with
``profiles.profile_exists``, which can never match ``@default``, so seven cards
sat in ``ready`` with no worker, no error and no event — while the gateway's
"dispatcher stuck" warning sent the operator to profile health, which was fine.

The contract these tests hold:
  1. a mention-style assignee is normalized to the profile name it names;
  2. a name that resolves to no profile is rejected where it is written;
  3. a row that still carries one (pre-guard, or written by an external tool) is
     not silent — it records exactly one ``skipped_nonspawnable`` event and the
     dispatcher's health notice names it.
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

    ``default`` is the only profile that resolves here: ``profile_exists``
    special-cases it, and ``Path.home`` is redirected so no named profile
    directory exists on the real machine.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _event_kinds(conn, task_id: str) -> list[str]:
    return [
        row["kind"] for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
        ).fetchall()
    ]


def test_mention_style_assignee_is_stored_as_the_profile_name(kanban_home):
    """``@default`` is the display spelling; storing it verbatim makes the card
    unspawnable. It must land as ``default``, and the dispatcher must then see a
    real profile rather than a non-spawnable lane."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="mention-style assignee", assignee="@ Default ")
        stored = kb.get_task(conn, tid).assignee
        res = kbd.dispatch_once(conn, dry_run=True)

    assert stored == "default"
    assert res.skipped_nonspawnable == []
    assert tid in [entry[0] for entry in res.spawned]


def test_unresolvable_assignee_is_reported_at_creation(kanban_home):
    """Creation must not be silent, and must not be a hard error either: a name
    that resolves to no local profile is still a legal write (an external worker
    lane pulls its own tasks), so the guard is an explicit advisory the creator
    can act on while it is still in the loop."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ghost assignee", assignee="ghost-profile")
        assert kb.get_task(conn, tid).assignee == "ghost-profile"
        advisory = kbd.assignee_advisory("ghost-profile", task_id=tid)
        assert kbd.assignee_advisory("default", task_id=tid) is None

    assert advisory is not None
    assert "ghost-profile" in advisory and tid in advisory
    assert "hermes kanban assign" in advisory


def test_unresolvable_assignee_is_recorded_and_reported_once(kanban_home):
    """A pre-guard row must not be silent, and must not spam an event per tick."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="legacy bad row")
        conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", ("ghost-profile", tid))
        conn.commit()

        kbd.dispatch_once(conn)
        kbd.dispatch_once(conn)

        kinds = _event_kinds(conn, tid)
        assert kinds.count(kbd.SKIPPED_NONSPAWNABLE) == 1
        assert kb.get_task(conn, tid).status == "ready"
        assert kbd.unresolved_assignee_ready(conn) == [(tid, "ghost-profile")]

        seen: set = set()
        first = kbd.unresolved_assignee_notice([(tid, "ghost-profile")], seen=seen)
        second = kbd.unresolved_assignee_notice([(tid, "ghost-profile")], seen=seen)

    assert first is not None
    assert "ghost-profile" in first and tid in first
    # Not a profile-health problem, and says so — the misleading hint is the bug.
    assert "profile-health" in first
    # Steady state stays quiet: the same card is not re-reported every tick.
    assert second is None
