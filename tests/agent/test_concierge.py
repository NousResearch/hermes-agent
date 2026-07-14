"""Unit tests for Concierge mode → Kanban mapping."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


def test_disabled_by_default_returns_none():
    from agent.concierge import handle_concierge

    assert handle_concierge("멈춰") is None
    assert handle_concierge("지금 뭐 하고 있어?") is None


def test_stop_consumed_when_enabled():
    from agent.concierge import handle_concierge

    session = {"concierge_live_enabled": True}
    cancelled = []
    r = handle_concierge(
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
    from agent.concierge import handle_concierge

    session = {"concierge_live_enabled": True}
    r = handle_concierge("지금 뭐 하고 있어?", session=session)
    assert r is not None
    assert r.action == "status"
    assert "Kanban" in r.message or "No active" in r.message
    assert "concierge" in r.message.lower() or "Kanban" in r.message


def test_worker_creates_kanban_task():
    from agent.concierge import handle_concierge
    from hermes_cli import kanban_db

    session = {"concierge_live_enabled": True}
    r = handle_concierge(
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
    from agent.concierge import handle_concierge

    session = {"concierge_live_enabled": True}
    assert handle_concierge("서울 파니니 맛집 추천해줘", session=session) is None


def test_shim_still_imports():
    from agent.concierge import handle_concierge, concierge_enabled

    assert concierge_enabled(session={"concierge_live_enabled": True}) is True
    assert handle_concierge("멈춰", session={"concierge_live_enabled": True}) is not None
