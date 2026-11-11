"""The session list ships reset provenance so clients stop nesting ``/new`` as a fork (#142064).

A ``/new`` reset stamps ``model_config._reset_from`` and keeps ``parent_session_id`` for durable
lineage; the Desktop reads the marker itself and treats the row as a new conversation.  Any REST
client that nests on ``parent_session_id`` got no such bit, so one long-lived DM rendered as a root
plus one fork per reset.  ``model_config`` is stripped from list rows (it can carry provider
secrets), so the route exposes the derived boolean next to the row instead.
"""

from __future__ import annotations

from contextlib import closing

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "reset-provenance-token")
    from hermes_cli import web_server
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    # Resolved at import: another module in the same session may have imported
    # web_server first, so pin the token the app actually compares against.
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "reset-provenance-token")
    with closing(hermes_state.SessionDB()) as db:
        db.create_session("root-conversation", source="weixin")
        db.create_session(
            "reset-conversation", source="weixin",
            parent_session_id="root-conversation",
            model_config={"_reset_from": "root-conversation", "api_key": "must-not-leak"})
        db.create_session(
            "branch-conversation", source="weixin",
            parent_session_id="root-conversation",
            model_config={"_branched_from": "root-conversation"})

    with closing(TestClient(web_server.app, raise_server_exceptions=False)) as c:
        c.headers["Authorization"] = "Bearer reset-provenance-token"
        yield c


def _rows(client):
    r = client.get("/api/sessions", params={"order": "recent", "limit": 50})
    assert r.status_code == 200
    return {row["id"]: row for row in r.json()["sessions"]}


def test_reset_child_is_flagged_without_leaking_model_config(client):
    rows = _rows(client)
    assert rows["reset-conversation"]["is_reset_child"] is True
    assert rows["reset-conversation"]["parent_session_id"] == "root-conversation"
    assert "model_config" not in rows["reset-conversation"]
    assert "must-not-leak" not in str(rows["reset-conversation"])


def test_list_rows_ship_the_raw_markers_without_the_snapshot(client):
    """The dashboard list is the third consumer: it must ship the same pair the
    TUI gateway's ``_project_tree_row`` sends, and still no ``model_config``."""
    rows = _rows(client)
    assert rows["reset-conversation"]["_reset_from"] == "root-conversation"
    assert rows["branch-conversation"]["_branched_from"] == "root-conversation"
    assert "_branched_from" not in rows["reset-conversation"]
    assert "_reset_from" not in rows["branch-conversation"]
    for row in rows.values():
        assert "model_config" not in row
        assert "must-not-leak" not in str(row)


def test_roots_and_branches_keep_nesting_as_forks(client):
    rows = _rows(client)
    assert rows["root-conversation"]["is_reset_child"] is False
    # A /branch IS a fork: the client must keep nesting on its parent edge.
    assert rows["branch-conversation"]["is_reset_child"] is False
