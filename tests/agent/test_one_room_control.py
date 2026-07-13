"""Unit tests for one-room control tower → Kanban mapping."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Isolate HERMES_HOME before kanban_db touches disk
@pytest.fixture(autouse=True)
def _hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


def test_disabled_by_default_returns_none():
    from agent.one_room_control import handle_one_room_control

    assert handle_one_room_control("멈춰") is None
    assert handle_one_room_control("지금 뭐 하고 있어?") is None


def test_stop_consumed_when_enabled():
    from agent.one_room_control import handle_one_room_control

    session = {"frontdesk_live_enabled": True}
    cancelled = []
    r = handle_one_room_control(
        "멈춰",
        session=session,
        cancel_callback=lambda t: cancelled.append(t),
        main_in_flight=True,
    )
    assert r is not None
    assert r.action == "stop"
    assert cancelled == ["멈춰"]
    assert "not queued" in r.message.lower() or "Stopped" in r.message


def test_status_no_model_when_enabled():
    from agent.one_room_control import handle_one_room_control

    session = {"frontdesk_live_enabled": True}
    r = handle_one_room_control("지금 뭐 하고 있어?", session=session)
    assert r is not None
    assert r.action == "status"
    assert "Kanban" in r.message or "No active" in r.message


def test_worker_creates_kanban_task():
    from agent.one_room_control import handle_one_room_control
    from hermes_cli import kanban_db

    session = {"frontdesk_live_enabled": True}
    r = handle_one_room_control(
        "워커 레인에 배당해서 이 회귀를 조사해줘",
        session=session,
    )
    assert r is not None
    assert r.action == "new_task"
    assert r.task_id
    conn = kanban_db.connect()
    try:
        tasks = kanban_db.list_tasks(conn, limit=10)
    finally:
        conn.close()
    ids = {t.id for t in tasks}
    assert r.task_id in ids


def test_plain_question_falls_through_to_main():
    from agent.one_room_control import handle_one_room_control

    session = {"frontdesk_live_enabled": True}
    # Avoid status anchors (뭐/어때/어디까지) — use a clear MAIN shape
    assert handle_one_room_control("서울 파니니 맛집 추천해줘", session=session) is None
