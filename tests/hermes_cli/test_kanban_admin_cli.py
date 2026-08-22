"""CLI tests for the guarded admin archive integration.

Covers the two new ``hermes kanban`` verbs -- ``archive-graph`` and
``unarchive`` -- end-to-end through ``kanban_command`` on scratch boards
created under pytest ``tmp_path``. All DB access in these tests resolves
inside the temporary HERMES_HOME (the fixture deletes any ambient
HERMES_KANBAN_DB / HERMES_KANBAN_HOME pin), so live board data is never
touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban as kc


# ---------------------------------------------------------------------------
# Harness: scratch board under tmp_path + CLI runner
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_scratch(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a scratch 'recovery' board + an 'alt' board.

    Deletes ambient HERMES_KANBAN_DB / HERMES_KANBAN_HOME pins so DB
    resolution always stays under tmp_path/.hermes/kanban/boards/*.
    HERMES_KANBAN_BOARD defaults to the 'recovery' scratch board.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "admin-tester")
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "recovery")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db(board="recovery")
    kb.init_db(board="alt")
    return home


def run_cli(*argv: str):
    """Invoke ``hermes kanban <argv...>`` via the real argparse entry."""
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args(["kanban", *argv])
    return kc.kanban_command(args)


def _id(db, title):
    return kb.create_task(db, title=title, initial_status="blocked")


def _link(db, parent, child):
    kb.link_tasks(db, parent, child)


def _statuses(db):
    return {
        row["id"]: row["status"]
        for row in db.execute("SELECT id, status FROM tasks").fetchall()
    }


def build_dm(db):
    """A->B, A->C, B->D, C->D, all todo (closed dominated closure)."""
    a = _id(db, "dm A")
    b = kb.create_task(db, title="dm B", parents=[a])
    c = kb.create_task(db, title="dm C", parents=[a])
    d = kb.create_task(db, title="dm D", parents=[b, c])
    db.execute("UPDATE tasks SET status='todo' WHERE id=?", (a,))
    return {"a": a, "b": b, "c": c, "d": d}


def build_h1(db):
    """A->B; A->C1; D(done external)->C1. Archiving A would promote C1."""
    a = _id(db, "h1 A")
    b = kb.create_task(db, title="h1 B", parents=[a])
    c1 = kb.create_task(db, title="h1 C1", parents=[a])
    d = _id(db, "h1 D")
    _link(db, d, c1)
    db.execute("UPDATE tasks SET status='todo' WHERE id=?", (a,))
    db.execute("UPDATE tasks SET status='done' WHERE id=?", (d,))
    return {"a": a, "b": b, "c1": c1, "d": d}


def build_i1(db):
    """P(done external)->A(root) only. I1 live_external_parent warning."""
    p = _id(db, "i1 P")
    a = kb.create_task(db, title="i1 A")
    _link(db, p, a)
    db.execute("UPDATE tasks SET status='todo' WHERE id=?", (a,))
    db.execute("UPDATE tasks SET status='done' WHERE id=?", (p,))
    return {"p": p, "a": a}


def add_running(db, task_id):
    """Put task_id into a claimed/running state with a dead worker pid."""
    db.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
    claimed = kb.claim_task(db, task_id)
    assert claimed is not None and claimed.status == "running"
    db.execute(
        "UPDATE tasks SET worker_pid=? WHERE id=?", (999999, task_id)
    )


def _single_json(stdout: str) -> dict:
    """Exactly one JSON object on stdout; anything else is a failure."""
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected exactly one stdout line, got: {stdout!r}"
    return json.loads(lines[0])


# ---------------------------------------------------------------------------
# Parser / required-args / help text
# ---------------------------------------------------------------------------

def test_build_parser_has_archive_graph_and_unarchive(kanban_scratch):
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args(["kanban"])
    kanban_sub = args._kanban_parser
    choices = set()
    for act in kanban_sub._actions:
        if isinstance(act, argparse._SubParsersAction):
            choices.update(act.choices)
    assert "archive-graph" in choices
    assert "unarchive" in choices


def test_archive_graph_help_warns_secrets_and_promotions(kanban_scratch, capsys):
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["kanban", "archive-graph", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    # --reason help warns the text is PERSISTED and must hold no secrets.
    assert "PERSISTED" in out
    assert "secrets" in out.lower()
    # --allow-promotions help warns recompute_ready is global / may promote
    # unrelated tasks.
    assert "global" in out.lower()


# ---------------------------------------------------------------------------
# archive-graph
# ---------------------------------------------------------------------------

def test_archive_graph_dry_run_is_deterministic_and_non_mutating(
    kanban_scratch, capsys
):
    with kb.connect_closing(board="recovery") as db:
        built = build_dm(db)
    rc1 = run_cli("archive-graph", built["a"], "--reason", "cleanup",
                  "--dry-run")
    captured1 = capsys.readouterr()
    rc2 = run_cli("archive-graph", built["a"], "--reason", "cleanup",
                  "--dry-run")
    captured2 = capsys.readouterr()
    assert rc1 == 0 and rc2 == 0
    # Byte-identical, exactly one deterministic JSON object on stdout.
    assert captured1.out == captured2.out
    plan = _single_json(captured1.out)
    assert plan["dry_run"] is True
    assert plan["archive_group_id"] is None
    assert sorted(plan["root_ids"]) == [built["a"]]
    # Human summary goes to stderr, not stdout.
    assert captured1.err  # non-empty stderr summary
    assert not captured1.out.splitlines()[0].startswith("dry run")
    # No mutation, no group.
    with kb.connect_closing(board="recovery") as db:
        st = _statuses(db)
        for tid in (built["a"], built["b"], built["c"], built["d"]):
            assert st[tid] == "todo"
        assert db.execute(
            "SELECT COUNT(*) AS n FROM task_events "
            "WHERE kind='admin_archived'"
        ).fetchone()["n"] == 0


def test_archive_graph_h1_refusal_then_override(kanban_scratch, capsys):
    with kb.connect_closing(board="recovery") as db:
        built = build_h1(db)
    # H1: without --allow-promotions the command refuses (names id + flag).
    rc = run_cli("archive-graph", built["a"], "--reason", "cleanup")
    err = capsys.readouterr().err
    assert rc == 1
    assert "refused" in err
    assert "--allow-promotions" in err
    assert built["c1"] in err
    # Nothing was archived by the refusal.
    with kb.connect_closing(board="recovery") as db:
        assert _statuses(db)[built["a"]] == "todo"
        assert db.execute(
            "SELECT COUNT(*) AS n FROM task_events "
            "WHERE kind='admin_archived'"
        ).fetchone()["n"] == 0
    # Override with --allow-promotions succeeds; C1 is promoted to ready.
    rc = run_cli("archive-graph", built["a"], "--reason", "cleanup",
                 "--allow-promotions")
    out = capsys.readouterr().out
    assert rc == 0
    assert "Archive group" in out
    with kb.connect_closing(board="recovery") as db:
        st = _statuses(db)
        assert st[built["a"]] == "archived"
        assert st[built["b"]] == "archived"
        assert st[built["c1"]] == "ready"


def test_archive_graph_running_refusal_then_force(kanban_scratch, capsys,
                                                  monkeypatch):
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker",
                        lambda *a, **k: {})
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    with kb.connect_closing(board="recovery") as db:
        a = _id(db, "run A")
        b = kb.create_task(db, title="run B", parents=[a])
        add_running(db, a)
    rc = run_cli("archive-graph", a, "--reason", "cleanup")
    err = capsys.readouterr().err
    assert rc == 1
    assert "refused" in err
    assert "--force-running" in err
    assert a in err
    with kb.connect_closing(board="recovery") as db:
        assert _statuses(db)[a] == "running"
    rc = run_cli("archive-graph", a, "--reason", "cleanup", "--force-running")
    out = capsys.readouterr().out
    assert rc == 0
    assert "Archive group" in out
    with kb.connect_closing(board="recovery") as db:
        st = _statuses(db)
        assert st[a] == "archived"
        assert st[b] == "archived"


def test_archive_graph_execution_names_group_and_counts(kanban_scratch, capsys):
    with kb.connect_closing(board="recovery") as db:
        built = build_dm(db)
    rc = run_cli("archive-graph", built["a"], "--reason", "cleanup")
    out = capsys.readouterr().out
    assert rc == 0
    assert "Archive group ag_" in out
    assert "4 transitioned, 0 skipped" in out


def test_archive_graph_unknown_id_all_or_nothing(kanban_scratch, capsys):
    with kb.connect_closing(board="recovery") as db:
        built = build_dm(db)
    bogus = "t_doesnotexist"
    rc = run_cli("archive-graph", bogus, "--reason", "cleanup")
    err = capsys.readouterr().err
    assert rc == 1
    assert "refused" in err
    assert bogus in err
    # All-or-nothing: nothing was archived, even with a valid sibling id.
    rc2 = run_cli("archive-graph", built["a"], bogus, "--reason", "cleanup")
    err2 = capsys.readouterr().err
    assert rc2 == 1
    assert bogus in err2
    with kb.connect_closing(board="recovery") as db:
        assert _statuses(db)[built["a"]] == "todo"


def test_archive_graph_i1_warning_reports_not_refuses(kanban_scratch, capsys):
    with kb.connect_closing(board="recovery") as db:
        built = build_i1(db)
    # I1 (live external parent p) warns on stderr but does NOT refuse.
    rc = run_cli("archive-graph", built["a"], "--reason", "cleanup")
    captured = capsys.readouterr()
    assert rc == 0
    assert "Archive group" in captured.out
    assert "live_external_parent" in captured.err
    assert built["p"] in captured.err
    with kb.connect_closing(board="recovery") as db:
        assert _statuses(db)[built["a"]] == "archived"


# ---------------------------------------------------------------------------
# unarchive
# ---------------------------------------------------------------------------

def test_unarchive_modes_mutual_exclusion(kanban_scratch, capsys, monkeypatch):
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker",
                        lambda *a, **k: {})
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    with kb.connect_closing(board="recovery") as db:
        built = build_dm(db)
    # Seed a group to unarchive against.
    rc = run_cli("archive-graph", built["a"], "--reason", "cleanup")
    assert rc == 0
    capsys.readouterr()
    # Both task_ids and --group -> refused with "exactly one mode".
    rc = run_cli("unarchive", built["a"], "--group", "ag_whatever")
    err = capsys.readouterr().err
    assert rc == 2
    assert "exactly one mode" in err
    # Neither mode -> refused.
    rc = run_cli("unarchive")
    err = capsys.readouterr().err
    assert rc == 2
    assert "requires task_ids or --group" in err


def test_unarchive_direct_restores(kanban_scratch, capsys):
    with kb.connect_closing(board="recovery") as db:
        built = build_dm(db)
    rc = run_cli("archive-graph", built["a"], "--reason", "cleanup")
    assert rc == 0
    capsys.readouterr()
    rc = run_cli("unarchive", built["a"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Unarchive group direct: 1 restored, 0 skipped" in out
    with kb.connect_closing(board="recovery") as db:
        st = _statuses(db)
        # Restored out of 'archived'; recompute_ready may promote a root
        # task to 'ready'. It must not be 'archived'.
        assert st[built["a"]] != "archived"
        assert st[built["a"]] in ("todo", "ready")


def test_unarchive_group_restores_all_members(kanban_scratch, capsys):
    with kb.connect_closing(board="recovery") as db:
        built = build_dm(db)
    run_cli("archive-graph", built["a"], "--reason", "cleanup")
    out = capsys.readouterr().out
    import re
    m = re.search(r"Archive group (ag_[a-f0-9]+)", out)
    assert m
    group = m.group(1)
    capsys.readouterr()
    rc = run_cli("unarchive", "--group", group)
    out = capsys.readouterr().out
    assert rc == 0
    assert f"Unarchive group {group}: 4 restored, 0 skipped" in out
    with kb.connect_closing(board="recovery") as db:
        for tid in (built["a"], built["b"], built["c"], built["d"]):
            # All restored out of 'archived' (recompute may promote some).
            assert _statuses(db)[tid] != "archived"


def test_unarchive_unknown_group_refused(kanban_scratch, capsys):
    with kb.connect_closing(board="recovery") as db:
        _id(db, "unrelated")
    rc = run_cli("unarchive", "--group", "ag_missing")
    err = capsys.readouterr().err
    assert rc == 1
    assert "refused" in err
    assert "ag_missing" in err


def test_unarchive_missing_workspace_reported_without_recreation(
    kanban_scratch, capsys
):
    missing_ws = str(Path(kanban_scratch) / "gone-workspace")
    assert not Path(missing_ws).exists()
    with kb.connect_closing(board="recovery") as db:
        a = _id(db, "ws A")
        db.execute("UPDATE tasks SET workspace_path=? WHERE id=?",
                   (missing_ws, a))
    run_cli("archive-graph", a, "--reason", "cleanup")
    capsys.readouterr()
    rc = run_cli("unarchive", a)
    captured = capsys.readouterr()
    assert rc == 0
    assert "1 restored, 0 skipped" in captured.out
    assert "missing workspace" in captured.err.lower()
    # Reported without recreation: the directory still does not exist.
    assert not Path(missing_ws).exists()


# ---------------------------------------------------------------------------
# --board scoping end-to-end
# ---------------------------------------------------------------------------

def test_end_to_end_board_scoping(kanban_scratch, capsys):
    with kb.connect_closing(board="recovery") as db:
        r_a = _id(db, "recovery A")
        db.execute("UPDATE tasks SET status='todo' WHERE id=?", (r_a,))
    with kb.connect_closing(board="alt") as db:
        alt = _id(db, "alt A")
        db.execute("UPDATE tasks SET status='todo' WHERE id=?", (alt,))
    # Archive on 'recovery' with an explicit --board; the alt board's task
    # must remain untouched.
    rc = run_cli("--board", "recovery", "archive-graph", r_a, "--reason",
                 "cleanup", "--dry-run")
    assert rc == 0
    with kb.connect_closing(board="recovery") as db:
        assert _statuses(db)[r_a] == "todo"
    with kb.connect_closing(board="alt") as db:
        assert _statuses(db)[alt] == "todo"
    # Real execution scoped to recovery.
    rc = run_cli("--board", "recovery", "archive-graph", r_a, "--reason",
                 "cleanup")
    assert rc == 0
    capsys.readouterr()
    with kb.connect_closing(board="recovery") as db:
        assert _statuses(db)[r_a] == "archived"
    with kb.connect_closing(board="alt") as db:
        assert _statuses(db)[alt] == "todo"
