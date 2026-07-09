from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def parser():
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    return parser


def _run_repair(parser, *args):
    ns = parser.parse_args(["kanban", "repair", "zero-budget-failures", *args])
    return kc.kanban_command(ns)


def _seed_zero_budget_failure(conn, *, assignee="blocked-worker", title="extract clip"):
    tid = kb.create_task(
        conn,
        title=title,
        body="preserve this body",
        assignee=assignee,
        goal_mode=True,
        idempotency_key=f"source-key-{title}",
    )
    preflight = kb.WorkerBudgetPreflight(
        allowed=False,
        budget=0,
        budget_key="goals.max_turns",
        detail="effective Kanban worker execution budget goals.max_turns resolved to 0",
    )
    kb._block_spawn_zero_budget(conn, tid, preflight)
    return tid


def test_zero_budget_repair_dry_run_reports_candidates_without_mutating(
    kanban_home, parser, capsys
):
    with kb.connect() as conn:
        candidate = _seed_zero_budget_failure(conn)
        content_failure = kb.create_task(conn, title="content failure", assignee="worker")
        kb.block_task(conn, content_failure, reason="source missing", kind="capability")
        before = {
            row.id: row.status
            for row in kb.list_tasks(conn, limit=100)
        }

    rc = _run_repair(parser, "--dry-run", "--json")

    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["dry_run"] is True
    assert [item["task_id"] for item in out["candidates"]] == [candidate]
    assert out["actions"][0]["action"] == "unblock"
    assert out["actions"][0]["mutated"] is False

    with kb.connect() as conn:
        after = {
            row.id: row.status
            for row in kb.list_tasks(conn, limit=100)
        }
    assert after == before


def test_zero_budget_repair_unblocks_original_when_config_was_fixed(
    kanban_home, parser, capsys
):
    with kb.connect() as conn:
        tid = _seed_zero_budget_failure(conn)

    rc = _run_repair(parser, "--json")

    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["actions"] == [{"task_id": tid, "action": "unblock", "mutated": True}]
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        comments = kb.list_comments(conn, tid)
        assert any("zero-budget repair" in c.body for c in comments)


def test_zero_budget_reroute_is_idempotent_and_links_downstream(
    kanban_home, parser, capsys
):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="source parent", assignee="planner")
        kb.complete_task(conn, parent, summary="parent done")
        original = _seed_zero_budget_failure(conn)
        kb.link_tasks(conn, parent, original)
        downstream = kb.create_task(
            conn, title="synthesize", assignee="synth", parents=[original]
        )

    rc1 = _run_repair(parser, "--reroute-profile", "healthy-worker", "--json")
    first = json.loads(capsys.readouterr().out)
    rc2 = _run_repair(parser, "--reroute-profile", "healthy-worker", "--json")
    second = json.loads(capsys.readouterr().out)

    assert rc1 == 0
    assert rc2 == 0
    first_replacement = first["actions"][0]["replacement_task_id"]
    assert second["actions"][0]["replacement_task_id"] == first_replacement
    assert second["actions"][0]["created"] is False

    with kb.connect() as conn:
        orig = kb.get_task(conn, original)
        repl = kb.get_task(conn, first_replacement)
        assert orig.status == "done"
        assert "superseded/routing-only" in (orig.result or "")
        assert repl.assignee == "healthy-worker"
        assert repl.title == "extract clip"
        assert repl.body == "preserve this body"
        assert repl.idempotency_key == f"zero-budget-repair:{original}:healthy-worker"
        assert kb.parent_ids(conn, repl.id) == [parent]
        assert set(kb.parent_ids(conn, downstream)) == {original, repl.id}
        replacements = [
            row.id for row in kb.list_tasks(conn, limit=100)
            if row.idempotency_key == f"zero-budget-repair:{original}:healthy-worker"
        ]
        assert replacements == [repl.id]


def test_zero_budget_reroute_profiles_distribute_candidates(kanban_home, parser, capsys):
    with kb.connect() as conn:
        first = _seed_zero_budget_failure(conn, title="first")
        second = _seed_zero_budget_failure(conn, title="second")

    rc = _run_repair(parser, "--reroute-profiles", "worker-a,worker-b", "--json")

    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    profiles = [action["reroute_profile"] for action in out["actions"]]
    assert sorted(action["task_id"] for action in out["actions"]) == sorted([first, second])
    assert sorted(profiles) == ["worker-a", "worker-b"]
