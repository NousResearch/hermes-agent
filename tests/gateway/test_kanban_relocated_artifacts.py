"""Owner admission must hand the worker the dispatcher's relocated scratch root.

With ``HERMES_KANBAN_WORKSPACES_ROOT`` relocating scratch workspaces, the dispatcher resolves
board paths pin-first (``kanban_db.pin_first_board_resolution``) when it creates a card's
scratch directory. The gateway's ``kanban.run`` admission computed the worker's
``workspaces_root`` outside that context, so the worker was told a different root than the one
its scratch directory lives under. Artifact staging then treated the scratch directory as
unmanaged and the files a worker attached to ``kanban_complete`` / ``kanban_request_review``
never reached the card.
"""

import json
import os
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(("action", "count"), [("complete", 1), ("review", 7)])
def test_relocated_scratch_artifacts_survive_handoff(tmp_path, monkeypatch, action, count):
    from gateway.session_contract import SessionActor
    from gateway.session_kanban import build_kanban_policy
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_workspace as kbw
    from hermes_cli.kanban_db_connect import connect

    for key in list(os.environ):
        if key.startswith("HERMES_KANBAN_") or key == "HERMES_DELEGATED_CHILD_CONTEXT":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "state"))
    relocated = tmp_path / "build" / "workspaces"
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(relocated))
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda name: str(tmp_path / "state"))
    # The dispatcher's side: the scratch workspace is created pin-first.
    with kb.pin_first_board_resolution(), closing(connect(board="default")) as conn:
        tid = kb.create_task(conn, title="Artifact handoff", assignee="default")
        ws = kbw.resolve_workspace(kb.get_task(conn, tid), board="default")
        kbw.set_workspace_path(conn, tid, ws)
        task = kb.claim_task(conn, tid)
        db = kb.kanban_db_path(board="default").resolve()
    files = []
    for n in range(count):
        p = ws / f"artifact-{n}.png"
        p.write_bytes(f"file {n}".encode())
        files.append(str(p))

    # The admitting gateway's side: it builds the worker's policy outside any pin-first context.
    actor = SessionActor("owner", str(tmp_path / "state"), frozenset({"session:create"}), "transport")
    connection = SimpleNamespace(
        authority=SimpleNamespace(profile_id=str(tmp_path / "state")), actor=actor, native_owner=True,
    )
    params = dict(board="default", task_id=tid, run_id=task.current_run_id, claim_lock=task.claim_lock, db=str(db))
    config = {"model": {"default": "loop-model", "provider": "custom"}, "platform_toolsets": {"cli": ["terminal", "file"]}}
    policy, _ = build_kanban_policy(connection, params, config)
    context = json.loads(policy.kanban_json)

    # The worker hands its files off with the root it was given.
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", context["workspaces_root"])
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    from tools import kanban_tools as kt

    args = {"summary": "Artifact handoff verified successfully", "artifacts": files,
            "metadata": {"acceptance_check": {"check": "artifact readback", "result": "pass", "evidence": "files written"}}}
    handler = kt._handle_complete if action == "complete" else kt._handle_request_review
    result = json.loads(handler(args))
    assert result.get("ok") is True, result
    rows = json.loads(kt._handle_attachments({"task_id": tid}))["attachments"]
    assert len(rows) == count
    assert {Path(row["stored_path"]).read_bytes() for row in rows} == {f"file {n}".encode() for n in range(count)}
    assert context["workspaces_root"] == str(relocated.resolve())
