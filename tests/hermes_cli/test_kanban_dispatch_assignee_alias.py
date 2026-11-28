"""``kanban.assignee_aliases`` reroutes a non-spawnable assignee in-tick.

A card addressed to a board persona (``chief``, mapped onto the
``default`` profile) used to sit in ``skipped_nonspawnable`` forever: with the mapping set,
the dispatcher mutates ownership and spawns the mapped profile in the same
tick, instead of silently archiving the work.
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


def _write_aliases(home, mapping):
    (home / "config.yaml").write_text(
        "kanban:\n  review_dispatch: true\n  assignee_aliases:\n"
        + "".join(f"    {k}: {v}\n" for k, v in mapping.items()),
        encoding="utf-8",
    )


def test_alias_reroutes_unknown_assignee_and_spawns(tmp_path, monkeypatch):
    """The mapped card changes owner, records the reroute event, and spawns
    under the mapped profile in the same tick."""
    _isolated_home(tmp_path, monkeypatch)
    _write_aliases(tmp_path / ".hermes", {"chief": "default"})
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "default")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="chief")
        res = kbd.dispatch_once(
            conn, dry_run=False, spawn_fn=lambda task, workspace, board=None: 4242)
        task = kb.get_task(conn, tid)
        kinds = [(e.kind, e.payload) for e in kb.list_events(conn, tid)]
    assert task is not None and task.assignee == "default"
    assert res.skipped_nonspawnable == []
    assert (tid, "chief", "default") in res.alias_rerouted
    assigned = [p for (k, p) in kinds if k == "assigned"]
    assert isinstance(assigned[0], dict)
    assert assigned[0].get("from") == "chief"
    assert assigned[0].get("assignee") == "default"
    assert assigned[0].get("source") == "kanban.assignee_aliases"
    assert res.spawned and res.spawned[-1][1] == "default"


def test_dry_run_reports_reroute_without_writing(tmp_path, monkeypatch):
    """The tick projects the reroute (and spawn) without touching the board."""
    _isolated_home(tmp_path, monkeypatch)
    _write_aliases(tmp_path / ".hermes", {"chief": "default"})
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "default")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="chief")
        res = kbd.dispatch_once(
            conn, dry_run=True, spawn_fn=lambda task, workspace, board=None: 4242)
        task = kb.get_task(conn, tid)
        kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert task is not None and task.assignee == "chief"
    assert "assigned" not in kinds
    assert (tid, "chief", "default") in res.alias_rerouted
    assert res.spawned and res.spawned[-1][1] == "default"


def test_without_alias_key_skip_unchanged(tmp_path, monkeypatch):
    """No mapping key: the pre-existing ``skipped_nonspawnable`` behavior."""
    _isolated_home(tmp_path, monkeypatch)
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="no-such-profile")
        res = kbd.dispatch_once(conn, dry_run=False)
        task = kb.get_task(conn, tid)
    assert task is not None and task.assignee == "no-such-profile"
    assert res.skipped_nonspawnable == [tid]
    assert res.alias_rerouted == []


def test_alias_to_missing_target_does_not_reroute(tmp_path, monkeypatch):
    """A mapping that names another non-spawnable profile is ignored
    (fail-open): the card stays in the skip bucket, ownership untouched."""
    _isolated_home(tmp_path, monkeypatch)
    _write_aliases(tmp_path / ".hermes", {"chief": "ghost"})
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "default")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="chief")
        res = kbd.dispatch_once(conn, dry_run=False)
        task = kb.get_task(conn, tid)
    assert task is not None and task.assignee == "chief"
    assert res.skipped_nonspawnable == [tid]
    assert res.alias_rerouted == []
