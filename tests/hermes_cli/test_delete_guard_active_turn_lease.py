"""Delete surfaces must refuse a session row that a turn still owns (#123583).

Deleting the row under a live agent makes every later flush of that session fail its FK, and the
turn's transcript is dropped with one WARNING and no other trace. The agent-side self-heal covers
the idle case; these pin the mid-turn refusal on each user-facing delete surface, plus the control
that a session with no lease still deletes.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_state import SessionDB


def _holder(tag: str = "turn") -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


@pytest.fixture
def session_db(_isolate_hermes_home):
    from hermes_constants import get_hermes_home

    db = SessionDB(get_hermes_home() / "state.db")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def web_client(_isolate_hermes_home):
    """The real ``manage_router`` on a single-profile host, so ``destructive_profile`` passes
    an omitted profile straight through to this process's own state.db."""
    from hermes_cli.web_routers.sessions import manage_router

    app = FastAPI()
    app.include_router(manage_router)
    with TestClient(app) as client:
        yield client


def test_web_delete_refuses_while_a_turn_holds_the_lease(session_db, web_client):
    session_db.create_session("live", source="desktop")
    holder = _holder("web")
    assert session_db.try_acquire_session_turn_lease("live", holder, ttl_seconds=300)

    resp = web_client.delete("/api/sessions/live")

    assert resp.status_code == 409, resp.text
    assert "active turn" in resp.json()["detail"]
    assert holder in resp.json()["detail"]
    assert session_db.get_session("live") is not None


def test_web_delete_still_removes_a_session_with_no_lease(session_db, web_client):
    session_db.create_session("idle", source="desktop")

    resp = web_client.delete("/api/sessions/idle")

    assert resp.status_code == 200, resp.text
    assert resp.json()["ok"] is True
    assert session_db.get_session("idle") is None


def test_web_bulk_delete_skips_leased_rows_and_reports_them(session_db, web_client):
    session_db.create_session("live", source="desktop")
    session_db.create_session("idle", source="desktop")
    holder = _holder("bulk")
    assert session_db.try_acquire_session_turn_lease("live", holder, ttl_seconds=300)

    resp = web_client.post("/api/sessions/bulk-delete", json={"ids": ["live", "idle"]})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["deleted"] == 1
    assert body["skipped_active"] == {"live": holder}
    assert session_db.get_session("live") is not None
    assert session_db.get_session("idle") is None


def test_cli_delete_refuses_while_a_turn_holds_the_lease(session_db, capsys):
    from hermes_cli.sessions_cmd import _cmd_delete

    session_db.create_session("live", source="cli")
    holder = _holder("cli")
    assert session_db.try_acquire_session_turn_lease("live", holder, ttl_seconds=300)

    rc = _cmd_delete(session_db, SimpleNamespace(session_id="live", yes=True))

    assert rc == 1
    out = capsys.readouterr().out
    assert "active turn" in out and holder in out
    assert session_db.get_session("live") is not None


def test_cli_delete_still_removes_a_session_with_no_lease(session_db, capsys):
    from hermes_cli.sessions_cmd import _cmd_delete

    session_db.create_session("idle", source="cli")

    rc = _cmd_delete(session_db, SimpleNamespace(session_id="idle", yes=True))

    assert rc is None
    assert "Deleted session" in capsys.readouterr().out
    assert session_db.get_session("idle") is None
