"""Exact dispatch selects only the requested task and shares normal dispatch gates."""

import argparse
import contextlib
import json
import os

import pytest

from hermes_cli import (
    kanban_db as kb,
    kanban_db_connect as kbc,
    kanban_db_dispatch as kbd,
)
from hermes_cli.kanban_ops import _cmd_dispatch
from hermes_cli.kanban_parser import build_parser


@pytest.mark.parametrize(
    "gate",
    [
        "none",
        "missing",
        "dependency",
        "human_review",
        "capacity",
        "profile_capacity",
        "paused",
        "guard",
        "memory",
        "review_disabled",
        "workspace",
        "unassigned",
        "locked",
        "other_board",
    ],
)
def test_exact_dispatch_never_substitutes_another_task(tmp_path, monkeypatch, gate):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda name: True)
    monkeypatch.setattr(
        kbd,
        "_memory_pressure_level",
        lambda: "critical" if gate == "memory" else "normal",
    )
    kb.init_db()
    spawned = []
    with kbc.connect_closing() as conn:
        other = kb.create_task(
            conn, title="Unrelated higher priority", assignee="other", priority=100
        )
        parents = [other] if gate == "dependency" else []
        target = kb.create_task(
            conn, title="Requested", assignee="builder", parents=parents
        )
        if gate == "human_review":
            kb.block_task(
                conn,
                target,
                reason="awaiting review from the approving owner",
                kind="needs_input",
            )
        if gate == "profile_capacity":
            occupied = kb.create_task(conn, title="Occupied", assignee="builder")
            claimed = kb.claim_task(conn, occupied, claimer="test")
            kbd._set_worker_pid(conn, occupied, os.getpid())
            assert claimed
        if gate == "paused":
            (tmp_path / "ESTOP").touch()
        if gate == "guard":
            monkeypatch.setattr(
                kbd, "check_respawn_guard", lambda *a, **kw: "active_pr"
            )
        if gate == "review_disabled":
            kb.request_review(conn, target, summary="Ready for independent review")
            monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: False)
        if gate == "workspace":
            (tmp_path / "not-a-directory").write_text("file", encoding="utf-8")
            conn.execute(
                "UPDATE tasks SET workspace_kind='dir', workspace_path=? WHERE id=?",
                (str(tmp_path / "not-a-directory"), target),
            )
        if gate == "unassigned":
            kb.assign_task(conn, target, None)
        if gate == "other_board":
            kb.create_board("another")
            with kbc.connect_closing(board="another") as other_conn:
                target = kb.create_task(
                    other_conn, title="Outside board", assignee="builder"
                )
        requested = "t_missing" if gate == "missing" else target

        with (
            kbc._dispatch_tick_lock(kb.kanban_db_path())
            if gate == "locked"
            else contextlib.nullcontext()
        ):
            result = kbd.dispatch_once(
                conn,
                task_id=requested,
                spawn_fn=lambda task, workspace, **kw: (
                    spawned.append(task.id) or os.getpid()
                ),
                max_in_progress=0 if gate == "capacity" else None,
                max_in_progress_per_profile=1,
            )

        assert spawned == ([target] if gate == "none" else [])
        assert kb.get_task(conn, other).status == "ready"
        assert bool(result.requested_task_reason) == (gate != "none")
        if gate == "human_review":
            assert kb.get_task(conn, target).status == "blocked"


def test_cli_exact_dry_run_reports_target_without_claiming(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda name: True)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "normal")
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: {"kanban": {"max_in_progress": 2}}
    )
    kb.init_db()
    with kbc.connect_closing() as conn:
        target = kb.create_task(conn, title="Requested", assignee="builder")
        other = kb.create_task(conn, title="Other", assignee="builder", priority=100)
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["kanban", "dispatch", target, "--dry-run", "--json"])

    assert _cmd_dispatch(args) == 0

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["requested_task_id"] == target
    assert [row["task_id"] for row in receipt["spawned"]] == [target]
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, target).status == "ready"
        assert kb.get_task(conn, other).status == "ready"
