"""Real read-only timeline requests against temporary profile stores."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_state import SessionDB


@pytest.fixture
def timeline_store(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id="timeline-root", source="desktop")
    from hermes_cli.web_routers.sessions import manage_router

    app = FastAPI()
    app.include_router(manage_router)
    with TestClient(app) as client:
        yield db, client, home
    db.close()


def test_timeline_pages_project_human_prompts_without_tool_payloads(timeline_store):
    db, client, _ = timeline_store
    db.append_messages_batch("timeline-root", [
        {"role": "user", "content": " First\n  prompt ", "timestamp": 200},
        {"role": "assistant", "content": "answer", "tool_calls": [
            {"id": "tool-1", "function": {"name": "terminal", "arguments": "x" * 10000}}]},
        {"role": "tool", "content": "secret tool payload" * 1000, "tool_call_id": "tool-1"},
        {"role": "user", "content": "é" * 150, "timestamp": 100},
    ])
    expected_ids = [row["id"] for row in db.get_messages("timeline-root") if row["role"] == "user"]
    response = client.get("/api/sessions/timeline-root/timeline?limit=1")
    assert response.status_code == 200
    first = response.json()
    assert first["session_id"] == "timeline-root"
    assert first["profile"] == "default"
    assert first["entries"] == [{"row_id": expected_ids[0], "preview": "First prompt", "timestamp": 200}]
    assert first["pagination"]["total"] == 2
    assert first["pagination"]["returned"] == 1
    assert first["pagination"]["has_more"] is True
    second = client.get("/api/sessions/timeline-root/timeline", params={
        "after_row_id": first["pagination"]["next_cursor"], "limit": 1}).json()
    assert second["entries"][0]["row_id"] == expected_ids[1]
    assert len(second["entries"][0]["preview"]) == 120
    assert second["pagination"]["has_more"] is False
    assert second["pagination"]["next_cursor"] is None
    assert b"secret tool payload" not in response.content


@pytest.mark.parametrize("legacy", [False, True])
def test_compacted_timeline_and_jump_share_display_order_and_visibility(timeline_store, legacy):
    from agent.context_compressor import COMPRESSION_CONTINUATION_USER_CONTENT, SUMMARY_PREFIX, _SUMMARY_END_MARKER

    db, client, _ = timeline_store
    sid = "timeline-root"
    db.append_messages_batch(sid, [
        {"role": "user", "content": "old ask", "timestamp": 10},
        {"role": "assistant", "content": "old answer", "timestamp": 11},
        {"role": "user", "content": "retained ask", "timestamp": 12},
        {"role": "assistant", "content": "retained answer", "timestamp": 13},
    ])
    retained = db.get_messages(sid)[2:]
    db.archive_and_compact(sid, [
        {"role": "user", "content": SUMMARY_PREFIX + "summary", "_compressed_summary": True},
        *retained,
    ])
    db.append_messages_batch(sid, [
        {"role": "user", "content": "hidden ask", "display_kind": "hidden"},
        {"role": "user", "content": COMPRESSION_CONTINUATION_USER_CONTENT},
        {"role": "user", "content": "[IMPORTANT: Background process 12 completed]"},
        {"role": "user", "content": "[ASYNC DELEGATION BATCH COMPLETE] completed"},
        {"role": "user", "content": "A background subagent you dispatched earlier has finished."},
        {"role": "user", "content": SUMMARY_PREFIX + "handoff\n" + _SUMMARY_END_MARKER + "\nlive ask",
         "_compressed_summary": True, "display_kind": "hidden"},
        {"role": "assistant", "content": "live answer"},
        {"role": "user", "content": "rewound ask"},
    ])
    rewound_id = db.get_messages(sid)[-1]["id"]
    db._write_sql("UPDATE messages SET active = 0, compacted = 0 WHERE id = ?", (rewound_id,))
    if legacy:
        db._write_sql("UPDATE messages SET display_identity = NULL, display_order = NULL WHERE session_id = ?", (sid,))
    timeline = client.get(f"/api/sessions/{sid}/timeline").json()
    assert [entry["preview"] for entry in timeline["entries"]] == ["old ask", "retained ask", "live ask"]
    anchor = timeline["entries"][1]["row_id"]
    response = client.get(f"/api/sessions/{sid}/messages/around?row_id={anchor}&limit=2")
    assert response.status_code == 200
    jump = response.json()
    assert [row["content"] for row in jump["messages"]] == ["retained ask", "retained answer"]
    assert jump["pagination"]["has_older"] is True
    assert jump["pagination"]["has_newer"] is True
    live_id = timeline["entries"][-1]["row_id"]
    last = client.get(f"/api/sessions/{sid}/messages/around?row_id={live_id}").json()
    assert last["messages"][0]["display_content"] == "live ask"
    assert last["messages"][1]["content"] == "live answer"
    assert last["pagination"]["has_newer"] is False
    assert client.get(f"/api/sessions/{sid}/messages/around?row_id={rewound_id}").status_code == 404


def test_cursor_survives_compaction_between_pages(timeline_store):
    db, client, _ = timeline_store
    sid = "timeline-root"
    db.append_messages_batch(sid, [
        {"role": "user", "content": f"ask {i}", "timestamp": 100 - i} for i in range(4)
    ])
    first = client.get(f"/api/sessions/{sid}/timeline?limit=2").json()
    old_row_id = first["entries"][-1]["row_id"]
    retained = db.get_messages(sid)[1:]
    db.archive_and_compact(sid, retained)
    second = client.get(f"/api/sessions/{sid}/timeline", params={
        "limit": 2, "after_row_id": first["pagination"]["next_cursor"]}).json()
    assert [r["preview"] for r in first["entries"] + second["entries"]] == [f"ask {i}" for i in range(4)]
    assert second["pagination"]["total"] == 4
    assert second["pagination"]["has_more"] is False
    current = client.get(f"/api/sessions/{sid}/timeline").json()
    assert current["entries"][1]["row_id"] > old_row_id
    empty = client.get(f"/api/sessions/{sid}/timeline?after_row_id=99999").json()
    assert empty["entries"] == []
    assert empty["pagination"]["total"] == 4
    assert empty["pagination"]["has_more"] is False


def test_exact_owner_lineage_validation_and_bounded_jump(timeline_store):
    db, client, home = timeline_store
    sid = "timeline-root"
    db.end_session(sid, end_reason="compression")
    db.create_session(session_id="timeline-tip", source="desktop", parent_session_id=sid)
    db.append_messages_batch("timeline-tip", [
        {"role": "user", "content": "tip ask"},
        *[{"role": "assistant", "content": f"step {i}"} for i in range(130)],
        {"role": "user", "content": "last ask"},
        {"role": "assistant", "content": "last answer"},
    ])
    db.create_session(session_id="delegate", source="tool", parent_session_id="timeline-tip")
    db.append_message("delegate", role="user", content="delegate ask")
    db.create_session(session_id="foreign", source="desktop")
    foreign = db.append_message("foreign", role="user", content="foreign ask")
    expected_sid = client.get(f"/api/sessions/{sid}/messages").json()["session_id"]
    first = client.get(f"/api/sessions/{sid}/timeline?limit=1").json()
    assert first["session_id"] == expected_sid == "timeline-tip"
    row_id = first["entries"][0]["row_id"]
    jump = client.get(f"/api/sessions/{sid}/messages/around?row_id={row_id}").json()
    assert jump["messages"][0]["id"] == row_id
    assert len(jump["messages"]) == jump["pagination"]["returned"] == 120
    assert jump["pagination"]["has_older"] is False
    assert jump["pagination"]["has_newer"] is True
    assert client.get(f"/api/sessions/{sid}/messages/around?row_id={foreign}").status_code == 404
    assert client.get(f"/api/sessions/{sid}/messages/around?row_id={row_id + 1}").status_code == 404
    assert client.get(f"/api/sessions/{sid}/messages/around?row_id={row_id}&limit=121").status_code == 422
    assert client.get(f"/api/sessions/{sid}/timeline?limit=501").status_code == 422
    assert client.get("/api/sessions/timeline-ro/timeline").status_code == 404
    assert client.get("/api/sessions/absent/timeline").status_code == 404
    work = home / "profiles" / "work"
    work.mkdir(parents=True)
    with SessionDB(db_path=work / "state.db") as other:
        other.create_session(session_id=sid, source="desktop")
        other.append_message(sid, role="user", content="work ask")
    for profile, expected in (("default", "tip ask"), ("work", "work ask"), ("default", "tip ask")):
        page = client.get(f"/api/sessions/{sid}/timeline?profile={profile}").json()
        assert page["profile"] == profile
        assert page["entries"][0]["preview"] == expected
    db._write_sql("UPDATE sessions SET profile_name = 'wrong-owner' WHERE id = 'timeline-tip'")
    assert client.get(f"/api/sessions/{sid}/timeline").status_code == 404
    assert client.get(f"/api/sessions/{sid}/messages/around?row_id={row_id}").status_code == 404


def test_timeline_sql_never_reads_tool_columns_or_writes(timeline_store, monkeypatch):
    import sqlite3
    from contextlib import contextmanager
    from hermes_state_timeline import get_session_timeline

    db, _, home = timeline_store
    db.append_messages_batch("timeline-root", [
        {"role": "user", "content": [{"type": "text", "text": "multimodal ask"},
                                      {"type": "image_url", "image_url": {"url": "data:unused"}}]},
        {"role": "tool", "content": "unread tool result", "tool_calls": [{"unused": "payload"}]},
    ])
    reader = SessionDB(db_path=home / "state.db", read_only=True)
    original = reader._read_ctx
    accesses = []

    def authorize(action, table, column, *_):
        if action == sqlite3.SQLITE_READ:
            accesses.append((table, column))
            if table == "messages" and column in {"tool_calls", "reasoning", "api_content", "codex_reasoning_items"}:
                return sqlite3.SQLITE_DENY
        if action in {sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_INSERT, sqlite3.SQLITE_DELETE}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    @contextmanager
    def guarded():
        with original() as conn:
            conn.set_authorizer(authorize)
            try:
                yield conn
            finally:
                conn.set_authorizer(None)

    monkeypatch.setattr(reader, "_read_ctx", guarded)
    try:
        page = get_session_timeline(reader, "timeline-root")
        assert page["entries"][0]["preview"] == "multimodal ask"
        assert page["pagination"]["total"] == 1
        assert accesses
    finally:
        reader.close()


@pytest.mark.parametrize("legacy", [False, True])
def test_adjacent_pages_cross_long_tool_turn_without_gaps_and_survive_compaction(timeline_store, legacy, monkeypatch):
    db, client, _ = timeline_store
    sid = "timeline-root"
    rows = [{"role": "user", "content": "long turn", "timestamp": 1}]
    for i in range(150):
        rows.extend([
            {"role": "assistant", "content": f"step {i}", "timestamp": i + 2,
             "tool_calls": [{"id": f"call-{i}", "type": "function",
                             "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "content": f"result {i}", "timestamp": i + 2,
             "tool_name": "terminal", "tool_call_id": f"call-{i}"},
        ])
    rows.extend([{"role": "user", "content": "last ask", "timestamp": 999},
                 {"role": "assistant", "content": "last answer", "timestamp": 1000}])
    db.append_messages_batch(sid, rows)
    original = db.get_messages(sid)
    # Every request hydrates at most its bounded selected payloads.
    hydrated = []
    decode = SessionDB._row_to_message_dict
    def counted(self, row, *args, **kwargs):
        hydrated.append(row["id"])
        return decode(self, row, *args, **kwargs)
    monkeypatch.setattr(SessionDB, "_row_to_message_dict", counted)
    if legacy:
        db._write_sql("UPDATE messages SET display_identity = NULL, display_order = NULL WHERE session_id = ?", (sid,))
    url = f"/api/sessions/{sid}/messages/around"
    def read(**params):
        hydrated.clear()
        response = client.get(url, params=params)
        assert response.status_code == 200, response.text
        page = response.json()
        assert len(hydrated) == len(page["messages"]) <= 120
        return page
    first = read(row_id=original[0]["id"])
    # Replace the active tail while a client still holds the old logical cursor.
    db.archive_and_compact(sid, original[100:])
    if legacy:
        db._write_sql("UPDATE messages SET display_identity = NULL, display_order = NULL WHERE session_id = ?", (sid,))
    second = read(after_cursor=first["pagination"]["last_cursor"])
    third = read(after_cursor=second["pagination"]["last_cursor"])
    combined = first["messages"] + second["messages"] + third["messages"]
    assert [m["content"] for m in combined] == [m["content"] for m in rows]
    assert first["pagination"]["has_older"] is False
    assert first["pagination"]["leading_prompt_row_id"] == original[0]["id"]
    assert third["pagination"]["has_newer"] is False
    assert second["pagination"]["leading_prompt_row_id"] == original[0]["id"]
    backward = read(before_cursor=third["pagination"]["first_cursor"])
    assert [m["content"] for m in backward["messages"]] == [m["content"] for m in second["messages"]]
    backward = read(before_cursor=backward["pagination"]["first_cursor"])
    assert [m["content"] for m in backward["messages"]] == [m["content"] for m in first["messages"]]
    assert backward["pagination"]["has_older"] is False


@pytest.mark.parametrize("params", [{}, {"row_id": 1, "after_cursor": 1},
                                     {"before_cursor": 1, "after_cursor": 1},
                                     {"before_cursor": 0}, {"after_cursor": 1, "limit": 121}])
def test_adjacent_page_addresses_are_exclusive_and_bounded(timeline_store, params):
    _, client, _ = timeline_store
    assert client.get("/api/sessions/timeline-root/messages/around", params=params).status_code == 422


def test_adjacent_page_keeps_exact_profile_ownership(timeline_store):
    db, client, home = timeline_store
    sid = "timeline-root"
    db.append_messages_batch(sid, [{"role": "user", "content": "default ask"},
                                   {"role": "assistant", "content": "default reply"}])
    work = home / "profiles" / "work"
    work.mkdir(parents=True)
    with SessionDB(db_path=work / "state.db") as other:
        other.create_session(session_id=sid, source="desktop")
        other.append_messages_batch(sid, [{"role": "user", "content": "work ask"},
                                          {"role": "assistant", "content": "work reply"}])
    for profile in ("default", "work", "default"):
        url = f"/api/sessions/{sid}/messages/around"
        first = client.get(url, params={"profile": profile, "row_id": 1, "limit": 1}).json()
        page = client.get(url, params={"profile": profile,
                         "after_cursor": first["pagination"]["last_cursor"]}).json()
        assert [m["content"] for m in page["messages"]] == [f"{profile} reply"]
    db._write_sql("UPDATE sessions SET profile_name = 'wrong-owner' WHERE id = ?", (sid,))
    assert client.get(url, params={"after_cursor": 1}).status_code == 404
    assert client.get("/api/sessions/timeline-ro/messages/around?after_cursor=1").status_code == 404
