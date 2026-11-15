"""Dispatcher must leave per-task diagnostics for unknown assignees (#122422).

A card assigned to a profile that does not exist lands in the aggregate
``skipped_nonspawnable`` bucket with no per-task event, so ``show``/``tail``
never explain why the card sits in ``ready`` forever.
"""
from __future__ import annotations

from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_unknown_assignee_skip_writes_per_task_event(tmp_path, monkeypatch):
    """RED (#122422): the skip must append a per-task board event naming the
    missing profile, so ``tail``/``show`` reveal why the card never spawns."""
    _isolated_home(tmp_path, monkeypatch)
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="no-such-profile")
        res = kbd.dispatch_once(conn, dry_run=False)
        kinds = [(e.kind, e.payload) for e in kb.list_events(conn, tid)]
        task = kb.get_task(conn, tid)
    assert res.skipped_nonspawnable == [tid]
    assert task is not None and task.status == "ready"
    matches = [p for (k, p) in kinds if k == "skipped_nonspawnable"]
    assert matches, f"no per-task skip event, kinds={[k for k, _ in kinds]}"
    assert isinstance(matches[0], dict) and matches[0].get("assignee") == "no-such-profile"


def test_unknown_assignee_skip_event_is_written_once(tmp_path, monkeypatch):
    """The condition never expires on its own: repeated ticks must not append a
    row each (one per minute forever; one per foreign home on a shared board),
    and a dry-run tick writes nothing."""
    _isolated_home(tmp_path, monkeypatch)
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="no-such-profile")
        kbd.dispatch_once(conn, dry_run=True)
        assert "skipped_nonspawnable" not in [e.kind for e in kb.list_events(conn, tid)]
        for _ in range(3):
            kbd.dispatch_once(conn, dry_run=False)
        kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert kinds.count("skipped_nonspawnable") == 1
