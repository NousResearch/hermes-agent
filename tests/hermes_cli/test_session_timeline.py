"""Real read-only timeline requests against temporary profile stores."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from persistence.core.identity import SessionId
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session


def _append_batch(store, session_id, messages):
    store.messages.append_batch(session_id, [NewMessage.from_live(message) for message in messages])


def _compact(store, session_id, messages):
    store.messages.compact(session_id, [NewMessage.from_live(message) for message in messages])


def _write_sql(store, sql, params=()):
    raw_holder(store)._execute_write(lambda conn: conn.execute(sql, params))


@pytest.fixture
def timeline_store(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.web_routers.sessions import manage_router

    app = FastAPI()
    app.include_router(manage_router)
    with open_persistence(profile_scope(home=home)) as db:
        seed_session(db, "timeline-root", "desktop")
        with TestClient(app) as client:
            yield db, client, home


@pytest.mark.parametrize("source", ["gui", "local"])
def test_desktop_history_includes_both_sides_of_a_local_reset(timeline_store, source):
    from tests.persistence_sqlite.authority.test_binding_local import SID, _child_entry, make_binding

    db, client, _ = timeline_store
    epoch = db.runtime.begin_epoch(instance_id="timeline-test")
    db.bindings.reserve(epoch=epoch, binding=make_binding(source=source), owner=None)
    before, = seed_messages(db, SID, {"role": "user", "content": "before reset", "timestamp": 10})
    child = "reset-transcript"
    db.bindings.reset_target(epoch=epoch, parent_session_id=SID, entry=_child_entry(child))
    after, = seed_messages(db, child, {"role": "user", "content": "after reset", "timestamp": 20})

    for session_id in (SID, child):
        path = f"/api/sessions/{session_id}"
        params = {"include_compacted": True, "order": "latest", "limit": 1}
        latest = client.get(f"{path}/messages", params=params)
        assert latest.status_code == 200
        assert [row["id"] for row in latest.json()["messages"]] == [after]
        older = client.get(f"{path}/messages", params={**params, "offset": 1})
        assert [row["id"] for row in older.json()["messages"]] == [before]
        timeline = client.get(f"{path}/timeline")
        assert [row["row_id"] for row in timeline.json()["entries"]] == [before, after]
        jump = client.get(f"{path}/messages/around", params={"row_id": before})
        assert [row["id"] for row in jump.json()["messages"]] == [before, after]

    active = client.get(f"/api/sessions/{SID}/messages")
    assert [row["id"] for row in active.json()["messages"]] == [before]

    path = f"/api/sessions/{SID}"
    opt_out = {"follow_binding": False}
    page = client.get(f"{path}/messages", params={**opt_out, "include_compacted": True})
    assert [row["id"] for row in page.json()["messages"]] == [before]
    timeline = client.get(f"{path}/timeline", params=opt_out)
    assert [row["row_id"] for row in timeline.json()["entries"]] == [before]
    jump = client.get(f"{path}/messages/around", params={**opt_out, "row_id": before})
    assert [row["id"] for row in jump.json()["messages"]] == [before]
    assert client.get(f"{path}/messages/around", params={**opt_out, "row_id": after}).status_code == 404


@pytest.mark.parametrize("field,drift", [("chat_id", "other-chat"), ("source", "telegram")])
def test_desktop_history_refuses_a_drifted_child(timeline_store, field, drift):
    from tests.persistence_sqlite.authority.test_binding_local import SID, _child_entry, make_binding

    db, client, _ = timeline_store
    epoch = db.runtime.begin_epoch(instance_id="timeline-test")
    db.bindings.reserve(epoch=epoch, binding=make_binding(source="gui"), owner=None)
    child = "reset-transcript"
    db.bindings.reset_target(epoch=epoch, parent_session_id=SID, entry=_child_entry(child))
    row_id, = seed_messages(db, child, {"role": "user", "content": "child prompt"})
    _write_sql(db, f"UPDATE sessions SET {field} = ? WHERE id = ?", (drift, child))
    for session_id in (SID, child):
        path = f"/api/sessions/{session_id}"
        assert client.get(f"{path}/messages", params={"include_compacted": True}).status_code == 503
        assert client.get(f"{path}/timeline").status_code == 503
        assert client.get(f"{path}/messages/around", params={"row_id": row_id}).status_code == 503


@pytest.mark.parametrize("reset_first", [False, True], ids=["compression-reset", "reset-compression"])
@pytest.mark.parametrize("legacy", [False, True], ids=["indexed", "legacy"])
def test_desktop_history_dedupes_and_rewinds_only_within_compression_chain(timeline_store, reset_first, legacy):
    from agent.context_compressor import SUMMARY_PREFIX
    from tests.persistence_sqlite.authority.test_binding_local import SID, _child_entry, make_binding

    db, client, _ = timeline_store
    epoch = db.runtime.begin_epoch(instance_id="timeline-test")
    db.bindings.reserve(epoch=epoch, binding=make_binding(source="local"), owner=None)
    reset, rotated = SessionId("reset-transcript"), SessionId("compressed-transcript")
    parent = SID
    prompt = {"role": "user", "content": "repeat across reset", "timestamp": 20}
    if reset_first:
        separate, = seed_messages(db, SID, prompt)
        db.bindings.reset_target(epoch=epoch, parent_session_id=SID, entry=_child_entry(reset))
        parent = reset

    first, original, _ = seed_messages(
        db, parent, {"role": "user", "content": "old prompt", "timestamp": 10}, prompt,
        {"role": "assistant", "content": "protected reply", "timestamp": 21},
    )
    retained = transcript_rows(db, parent)[1:]
    db.lineage.publish_compression_child(parent, CompressionChild(session_id=rotated, source="local"), [
        NewMessage.from_live({"role": "user", "content": SUMMARY_PREFIX + "summary",
                              "_compressed_summary": True}),
        *[NewMessage.from_live(row) for row in retained],
    ], require_compression_lease=False)
    summary, carried, reply = [row["id"] for row in transcript_rows(db, rotated)]
    if reset_first:
        db.messages.rewind_user_turn(rotated, 0)
    else:
        db.bindings.reset_target(epoch=epoch, parent_session_id=rotated, entry=_child_entry(reset))
        separate, = seed_messages(db, reset, prompt)
    if legacy:
        _write_sql(db, "UPDATE messages SET display_identity = NULL, display_order = NULL")

    expected = [separate, first, summary] if reset_first else [first, carried, reply, summary, separate]
    prompts = [separate, first] if reset_first else [first, carried, separate]
    _assert_mixed_history(client, SID, expected, prompts, original)


def _assert_mixed_history(client, session_id, expected, prompts, original):
    path = f"/api/sessions/{session_id}"
    params = {"include_compacted": True, "order": "latest"}
    response = client.get(f"{path}/messages", params=params)
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["messages"]] == expected
    latest = client.get(f"{path}/messages", params={**params, "limit": 2}).json()
    assert [row["id"] for row in latest["messages"]] == expected[-2:]
    timeline = client.get(f"{path}/timeline").json()
    assert [row["row_id"] for row in timeline["entries"]] == prompts
    assert timeline["pagination"]["total"] == len(prompts)
    cursor = 0
    for prompt_id in prompts:
        page = client.get(f"{path}/timeline", params={"limit": 1, "after_row_id": cursor}).json()
        assert [row["row_id"] for row in page["entries"]] == [prompt_id]
        cursor = page["pagination"]["next_cursor"]
        jump = client.get(f"{path}/messages/around", params={"row_id": prompt_id})
        assert [row["id"] for row in jump.json()["messages"]] == expected[expected.index(prompt_id):]
    assert client.get(f"{path}/messages/around", params={"row_id": original}).status_code == 404


def test_timeline_pages_project_human_prompts_without_tool_payloads(timeline_store):
    db, client, _ = timeline_store
    _append_batch(db, "timeline-root", [
        {"role": "user", "content": " First\n  prompt ", "timestamp": 200},
        {"role": "assistant", "content": "answer", "tool_calls": [
            {"id": "tool-1", "function": {"name": "terminal", "arguments": "x" * 10000}}]},
        {"role": "tool", "content": "secret tool payload" * 1000, "tool_call_id": "tool-1"},
        {"role": "user", "content": "é" * 150, "timestamp": 100},
    ])
    expected_ids = [row["id"] for row in transcript_rows(db, "timeline-root") if row["role"] == "user"]
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
    _append_batch(db, sid, [
        {"role": "user", "content": "old ask", "timestamp": 10},
        {"role": "assistant", "content": "old answer", "timestamp": 11},
        {"role": "user", "content": "retained ask", "timestamp": 12},
        {"role": "assistant", "content": "retained answer", "timestamp": 13},
    ])
    retained = transcript_rows(db, sid)[2:]
    _compact(db, sid, [
        {"role": "user", "content": SUMMARY_PREFIX + "summary", "_compressed_summary": True},
        *retained,
    ])
    _append_batch(db, sid, [
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
    rewound_id = transcript_rows(db, sid)[-1]["id"]
    _write_sql(db, "UPDATE messages SET active = 0, compacted = 0 WHERE id = ?", (rewound_id,))
    if legacy:
        _write_sql(db, "UPDATE messages SET display_identity = NULL, display_order = NULL WHERE session_id = ?", (sid,))
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
    _append_batch(db, sid, [
        {"role": "user", "content": f"ask {i}", "timestamp": 100 - i} for i in range(4)
    ])
    first = client.get(f"/api/sessions/{sid}/timeline?limit=2").json()
    old_row_id = first["entries"][-1]["row_id"]
    retained = transcript_rows(db, sid)[1:]
    _compact(db, sid, retained)
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
    db.sessions.end(sid, "compression")
    seed_session(db, "timeline-tip", "desktop", parent_session_id=sid)
    _append_batch(db, "timeline-tip", [
        {"role": "user", "content": "tip ask"},
        *[{"role": "assistant", "content": f"step {i}"} for i in range(130)],
        {"role": "user", "content": "last ask"},
        {"role": "assistant", "content": "last answer"},
    ])
    seed_session(db, "delegate", "tool", parent_session_id="timeline-tip")
    seed_messages(db, "delegate", {"role": "user", "content": "delegate ask"})
    seed_session(db, "foreign", "desktop")
    foreign, = seed_messages(db, "foreign", {"role": "user", "content": "foreign ask"})
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
    with open_persistence(profile_scope(home=work)) as other:
        seed_session(other, sid, "desktop")
        seed_messages(other, sid, {"role": "user", "content": "work ask"})
    for profile, expected in (("default", "tip ask"), ("work", "work ask"), ("default", "tip ask")):
        page = client.get(f"/api/sessions/{sid}/timeline?profile={profile}").json()
        assert page["profile"] == profile
        assert page["entries"][0]["preview"] == expected
    _write_sql(db, "UPDATE sessions SET profile_name = 'wrong-owner' WHERE id = 'timeline-tip'")
    assert client.get(f"/api/sessions/{sid}/timeline").status_code == 404
    assert client.get(f"/api/sessions/{sid}/messages/around?row_id={row_id}").status_code == 404


def test_timeline_sql_never_reads_tool_columns_or_writes(timeline_store, monkeypatch):
    import sqlite3
    from contextlib import contextmanager
    db, _, home = timeline_store
    _append_batch(db, "timeline-root", [
        {"role": "user", "content": [{"type": "text", "text": "multimodal ask"},
                                      {"type": "image_url", "image_url": {"url": "data:unused"}}]},
        {"role": "tool", "content": "unread tool result", "tool_calls": [{"unused": "payload"}]},
    ])
    with open_persistence_reader(home / "state.db") as reader:
        holder = raw_holder(reader)
        original = holder._read_ctx
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

        monkeypatch.setattr(holder, "_read_ctx", guarded)
        page = reader.messages.read_session_timeline("timeline-root")
        assert page.entries[0].preview == "multimodal ask"
        assert page.total == 1
        assert accesses
