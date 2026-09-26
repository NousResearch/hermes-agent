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


def test_unknown_assignee_dry_run_writes_no_event(tmp_path, monkeypatch):
    """Dry-run dispatch must not write rows — bucket only, no event."""
    _isolated_home(tmp_path, monkeypatch)
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="no-such-profile")
        res = kbd.dispatch_once(conn, dry_run=True)
        kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert res.skipped_nonspawnable == [tid]
    assert "skipped_nonspawnable" not in kinds
