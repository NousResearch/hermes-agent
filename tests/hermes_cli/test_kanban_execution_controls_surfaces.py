"""Execution-control race, rollback, CLI and dashboard route contracts."""
import argparse

from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread, main_thread

import pytest

from hermes_cli import kanban_db as kb
from tests.hermes_cli import test_kanban_execution_controls as controls_tests

operator_db = controls_tests.operator_db


def test_positive_sql_failure_rolls_back_all_controls(operator_db, monkeypatch):
    conn = operator_db
    tid = kb.create_task(conn, title="rollback", workspace_kind="scratch", initial_status="blocked")
    before = list(conn.iterdump())
    def broken_event(*args, **kwargs):
        raise RuntimeError("event insertion failed")
    monkeypatch.setattr(kb, "_append_event", broken_event)
    with pytest.raises(RuntimeError, match="event insertion failed"):
        kb.edit_task(conn, tid, goal_mode=True, goal_max_turns=2, max_retries=1,
                     max_runtime_seconds=1800)
    assert list(conn.iterdump()) == before


def test_positive_claim_wins_race(operator_db, monkeypatch):
    from hermes_cli import kanban_db_connect as kbc
    conn = operator_db
    tid = kb.create_task(conn, title="race", workspace_kind="scratch", initial_status="blocked")
    about_to_lock, release_editor = Event(), Event()
    original = kbc._execute_boundary_with_retry
    def boundary(c, sql):
        if sql == "BEGIN IMMEDIATE" and current_thread() is not main_thread():
            about_to_lock.set()
            assert release_editor.wait(10)
        return original(c, sql)
    monkeypatch.setattr(kbc, "_execute_boundary_with_retry", boundary)
    def edit():
        other = kbc.connect()
        try:
            return kb.edit_task(other, tid, goal_mode=True, max_retries=1)
        finally:
            other.close()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(edit)
        try:
            assert about_to_lock.wait(10)
            assert kb.unblock_task(conn, tid)
            assert kb.claim_task(conn, tid, claimer="isolated-test") is not None
            before = list(conn.iterdump())
        finally:
            release_editor.set()
        with pytest.raises(ValueError, match="stopped"):
            future.result(timeout=10)
    assert list(conn.iterdump()) == before


def test_positive_cli_and_dashboard_contract(operator_db):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from plugins.kanban.dashboard.plugin_api import router
    from hermes_cli.kanban import _cmd_edit
    from hermes_cli.kanban_parser import build_parser
    conn = operator_db
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post("/tasks", json={"title": "parity", "triage": True,
                               "workspace_kind": "scratch", "max_retries": 3})
        assert response.status_code == 200, response.text
        tid = response.json()["task"]["id"]
        assert kb.get_task(conn, tid).max_retries == 3
        values = dict(goal_mode=True, goal_max_turns=2, max_retries=1, max_runtime_seconds=1800)
        response = client.patch(f"/tasks/{tid}", json=values)
        assert response.status_code == 200, response.text
        assert all(response.json()["task"][key] == value for key, value in values.items())
        before = list(conn.iterdump())
        assert client.patch(f"/tasks/{tid}", json={"max_retries": False}).status_code == 422
        assert client.patch(f"/tasks/{tid}", json={"max_retries": 1, "status": "ready"}).status_code == 400
        assert list(conn.iterdump()) == before
        response = client.patch(f"/tasks/{tid}", json={"max_retries": None})
        assert response.status_code == 200
        assert kb.get_task(conn, tid).max_retries is None
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers())
    args = parser.parse_args(["kanban", "edit", tid, "--goal-mode", "false",
                             "--goal-max-turns", "clear", "--max-runtime-seconds", "60"])
    assert _cmd_edit(args) == 0
    task = kb.get_task(conn, tid)
    assert task.goal_mode is False and task.goal_max_turns is None and task.max_runtime_seconds == 60


def test_pure_create_rejects_raw_types_before_access():
    for values in ({"goal_mode": None}, {"max_retries": False}, {"max_runtime_seconds": -1}):
        with pytest.raises(ValueError):
            kb.create_task(None, title="invalid", **values)


def test_refusal_dashboard_no_board_access():
    from fastapi import HTTPException
    from agent.delegation_context import delegated_child_context
    from plugins.kanban.dashboard.plugin_api import update_task, UpdateTaskBody
    with delegated_child_context(), pytest.raises(HTTPException) as error:
        update_task("t_example", UpdateTaskBody(goal_mode=True), board=None)
    assert error.value.status_code == 403


def test_pure_dashboard_mixed_refused_before_board_access():
    from fastapi import HTTPException
    from plugins.kanban.dashboard.plugin_api import update_task, UpdateTaskBody
    with pytest.raises(HTTPException) as error:
        update_task("t_example", UpdateTaskBody(goal_mode=True, status="ready"), board=None)
    assert error.value.status_code == 400
