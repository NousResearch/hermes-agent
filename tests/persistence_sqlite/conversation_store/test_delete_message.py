"""Tests for deleting individual messages or exchanges without removing later conversation (#142421)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_state import SessionDB
from persistence.conversation.messages import DeleteMessageScope
from persistence.core.errors import TurnLeaseLost
from persistence.core.identity import MessageId


def _make_abc_session(db: SessionDB, session_id: str = "sess-abc") -> dict[str, int]:
    """Create a 3-exchange session A -> B -> C and return row ids."""
    db.create_session(session_id, source="cli")
    return {
        "u1": db.append_message(session_id, "user", "Question A unique_alpha"),
        "a1": db.append_message(session_id, "assistant", "Answer A"),
        "u2": db.append_message(session_id, "user", "Question B secret_bravo_token"),
        "a2": db.append_message(session_id, "assistant", "Answer B"),
        "u3": db.append_message(session_id, "user", "Question C unique_charlie"),
        "a3": db.append_message(session_id, "assistant", "Answer C"),
    }


def test_delete_single_message_preserves_later_and_clears_fts(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    ids = _make_abc_session(db)

    # Confirm FTS finds the secret before deletion
    assert any(h["session_id"] == "sess-abc" for h in db.search_messages("secret_bravo_token"))

    res = db.delete_message("sess-abc", ids["u2"], scope="message")
    assert res["deleted_ids"] == [ids["u2"]]
    assert res["remaining_message_count"] == 5
    assert res["has_compressed_summary"] is False

    # Later messages survive in order
    remaining = db.get_messages("sess-abc")
    assert [m["id"] for m in remaining] == [ids["u1"], ids["a1"], ids["a2"], ids["u3"], ids["a3"]]

    # Excluded from model context and FTS search
    conv = db.get_messages_as_conversation("sess-abc")
    assert all("secret_bravo_token" not in (m.get("content") or "") for m in conv)
    assert db.search_messages("secret_bravo_token") == []

    # Session message_count updated
    sess = db.get_session("sess-abc")
    assert sess is not None
    assert sess["message_count"] == 5


def test_delete_exchange_from_middle_preserves_a_and_c(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    ids = _make_abc_session(db)

    # Target the assistant message of exchange B with scope="exchange"
    res = db.delete_message("sess-abc", ids["a2"], scope="exchange")
    assert res["deleted_ids"] == [ids["u2"], ids["a2"]]
    assert res["remaining_message_count"] == 4

    remaining = db.get_messages("sess-abc")
    assert [m["id"] for m in remaining] == [ids["u1"], ids["a1"], ids["u3"], ids["a3"]]
    assert [m["content"] for m in remaining] == [
        "Question A unique_alpha", "Answer A", "Question C unique_charlie", "Answer C",
    ]

    # Survives a full DB reopen
    db.close()
    db2 = SessionDB(db_path=tmp_path / "state.db")
    reloaded = db2.get_messages("sess-abc")
    assert [m["id"] for m in reloaded] == [ids["u1"], ids["a1"], ids["u3"], ids["a3"]]
    db2.close()


def test_delete_tool_call_message_cascades_paired_tool_results(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess-tools", source="cli")
    u1 = db.append_message("sess-tools", "user", "Run two tools")
    a_tc = db.append_message(
        "sess-tools", "assistant", None,
        tool_calls=[
            {"id": "call_1", "type": "function", "function": {"name": "t1", "arguments": "{}"}},
            {"id": "call_2", "type": "function", "function": {"name": "t2", "arguments": "{}"}},
        ],
    )
    t1 = db.append_message("sess-tools", "tool", "out1", tool_call_id="call_1", tool_name="t1")
    t2 = db.append_message("sess-tools", "tool", "out2", tool_call_id="call_2", tool_name="t2")
    a_final = db.append_message("sess-tools", "assistant", "Done!")
    u2 = db.append_message("sess-tools", "user", "Next question")

    # Deleting a single tool result cascades to the parent assistant tool-call row + sibling tool result
    res = db.delete_message("sess-tools", t1, scope="message")
    assert res["deleted_ids"] == sorted([a_tc, t1, t2])
    assert res["remaining_message_count"] == 3

    remaining = db.get_messages("sess-tools")
    assert [m["id"] for m in remaining] == [u1, a_final, u2]
    sess = db.get_session("sess-tools")
    assert sess is not None
    assert sess["tool_call_count"] == 0


def test_delete_reports_compressed_summary_and_removes_archived_uid_copy(tmp_path: Path):
    import sqlite3

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("sess-comp", source="cli")
    u_orig = db.append_message("sess-comp", "user", "Early secret_xyz", message_uid="uid-secret-1")
    summary_id = db.append_message(
        "sess-comp", "user", "[CONTEXT COMPACTION] Summary of earlier turns",
        _compressed_summary=True,
    )
    u_clone = db.append_message("sess-comp", "user", "Early secret_xyz", message_uid="uid-secret-1")
    # Mark original as compaction-archived (active=0, compacted=1)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE messages SET active = 0, compacted = 1 WHERE id = ?", (u_orig,))

    res = db.delete_message("sess-comp", u_clone, scope="message")
    assert res["deleted_ids"] == sorted([u_orig, u_clone])
    assert res["has_compressed_summary"] is True
    assert res["remaining_message_count"] == 1
    assert [m["id"] for m in db.get_messages("sess-comp", include_inactive=True)] == [summary_id]
    assert db.search_messages("secret_xyz", include_inactive=True) == []


def test_delete_message_refused_under_active_turn_lease(tmp_path: Path):
    from persistence_assembly import open_persistence, profile_scope

    db = SessionDB(db_path=tmp_path / "state.db")
    ids = _make_abc_session(db, "sess-busy")
    db.close()

    with open_persistence(profile_scope(home=tmp_path), eager=True) as store:
        assert store.coordination.try_acquire_turn_lease(session_id="sess-busy", holder="holder-1", ttl_seconds=60.0)
        with pytest.raises(TurnLeaseLost):
            store.messages.delete_message("sess-busy", MessageId(ids["u2"]), scope=DeleteMessageScope.MESSAGE)
        store.coordination.release_turn_lease(session_id="sess-busy", holder="holder-1")

    # Message untouched
    db2 = SessionDB(db_path=tmp_path / "state.db")
    assert len(db2.get_messages("sess-busy")) == 6
    db2.close()


def test_delete_missing_message_returns_empty(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    _make_abc_session(db, "sess-miss")
    res = db.delete_message("sess-miss", 999999)
    assert res["deleted_ids"] == []
    assert res["remaining_message_count"] == 6


def test_cli_delete_message_dry_run_and_apply(tmp_path: Path, capsys):
    from hermes_cli.sessions_cmd import _cmd_delete_message
    from persistence_assembly import open_persistence, profile_scope

    with open_persistence(profile_scope(home=tmp_path), eager=True) as store:
        db = SessionDB(db_path=tmp_path / "state.db")
        ids = _make_abc_session(db, "sess-cli")
        db.close()

        # Dry run
        args = argparse.Namespace(
            session_id="sess-cli", message_id=ids["u2"], exchange=True, dry_run=True, yes=False, json=True,
        )
        rc = _cmd_delete_message(args, persistence=store)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["dry_run"] is True
        assert out["scope"] == "exchange"

        # Apply with --yes --json
        args.dry_run = False
        args.yes = True
        rc = _cmd_delete_message(args, persistence=store)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["ok"] is True
        assert out["deleted_ids"] == [ids["u2"], ids["a2"]]
        assert out["remaining_message_count"] == 4


def test_web_api_delete_session_message(tmp_path: Path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hermes_cli.web_routers import sessions as sessions_router

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    ids = _make_abc_session(db, "sess-web")
    db.close()

    from persistence_assembly import open_persistence, profile_scope

    def _fake_with_db(profile, fn, *, read_only=False):
        with open_persistence(profile_scope(home=tmp_path), eager=True) as store:
            return fn(store)

    monkeypatch.setattr(sessions_router, "_with_db", _fake_with_db)
    monkeypatch.setattr(sessions_router, "_owner_store_for", lambda req, profile: None)

    app = FastAPI()
    app.include_router(sessions_router.manage_router)
    client = TestClient(app)

    # Delete exchange B via HTTP DELETE
    resp = client.delete(f"/api/sessions/sess-web/messages/{ids['u2']}?scope=exchange")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["deleted_ids"] == [ids["u2"], ids["a2"]]
    assert body["remaining_message_count"] == 4

    # Deleting the already-deleted message returns 404
    resp2 = client.delete(f"/api/sessions/sess-web/messages/{ids['u2']}")
    assert resp2.status_code == 404

    # Unknown session returns 404
    resp3 = client.delete("/api/sessions/no-such-session/messages/1")
    assert resp3.status_code == 404


def test_delete_exchange_preserves_interleaved_system_and_session_meta(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess-meta", source="cli")
    sys_top = db.append_message("sess-meta", "system", "System prompt")
    u1 = db.append_message("sess-meta", "user", "Question A")
    a1 = db.append_message("sess-meta", "assistant", "Answer A")
    meta = db.append_message("sess-meta", "session_meta", "compaction marker")
    u2 = db.append_message("sess-meta", "user", "Question B")
    a2 = db.append_message("sess-meta", "assistant", "Answer B")
    sys_mid = db.append_message("sess-meta", "system", "Mid-session system notice")
    u3 = db.append_message("sess-meta", "user", "Question C")

    # Deleting exchange A stops at session_meta (does not consume it)
    res_a = db.delete_message("sess-meta", u1, scope="exchange")
    assert res_a["deleted_ids"] == [u1, a1]
    assert [m["id"] for m in db.get_messages("sess-meta")] == [sys_top, meta, u2, a2, sys_mid, u3]

    # Deleting exchange B stops at trailing system row (does not consume it)
    res_b = db.delete_message("sess-meta", a2, scope="exchange")
    assert res_b["deleted_ids"] == [u2, a2]
    assert [m["id"] for m in db.get_messages("sess-meta")] == [sys_top, meta, sys_mid, u3]

    # Targeting session_meta directly with scope="exchange" removes only that row
    res_meta = db.delete_message("sess-meta", meta, scope="exchange")
    assert res_meta["deleted_ids"] == [meta]
    assert [m["id"] for m in db.get_messages("sess-meta")] == [sys_top, sys_mid, u3]


def test_delete_archived_tool_result_cascades_archived_tool_group(tmp_path: Path):
    import sqlite3

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("sess-arch-tools", source="cli")
    u1 = db.append_message("sess-arch-tools", "user", "Run tools")
    a_tc = db.append_message(
        "sess-arch-tools", "assistant", None,
        tool_calls=[
            {"id": "arch_call_1", "type": "function", "function": {"name": "t1", "arguments": "{}"}},
            {"id": "arch_call_2", "type": "function", "function": {"name": "t2", "arguments": "{}"}},
        ],
    )
    t1 = db.append_message("sess-arch-tools", "tool", "out1", tool_call_id="arch_call_1", tool_name="t1")
    t2 = db.append_message("sess-arch-tools", "tool", "out2", tool_call_id="arch_call_2", tool_name="t2")
    summary = db.append_message(
        "sess-arch-tools", "user", "[CONTEXT COMPACTION] Summary", _compressed_summary=True,
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE messages SET active = 0, compacted = 1 WHERE id IN (?, ?, ?, ?)",
            (u1, a_tc, t1, t2),
        )

    res = db.delete_message("sess-arch-tools", t1, scope="message")
    assert res["deleted_ids"] == sorted([a_tc, t1, t2])
    assert res["remaining_message_count"] == 1
    assert res["has_compressed_summary"] is True
    assert [m["id"] for m in db.get_messages("sess-arch-tools", include_inactive=True)] == [u1, summary]


def test_delete_mixed_active_and_archived_tool_group(tmp_path: Path):
    import sqlite3

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("sess-mixed-tools", source="cli")
    u1 = db.append_message("sess-mixed-tools", "user", "Run mixed tools")
    a_tc = db.append_message(
        "sess-mixed-tools", "assistant", None,
        tool_calls=[
            {"id": "mix_call_1", "type": "function", "function": {"name": "t1", "arguments": "{}"}},
            {"id": "mix_call_2", "type": "function", "function": {"name": "t2", "arguments": "{}"}},
        ],
    )
    t1 = db.append_message("sess-mixed-tools", "tool", "out1", tool_call_id="mix_call_1", tool_name="t1")
    t2 = db.append_message("sess-mixed-tools", "tool", "out2", tool_call_id="mix_call_2", tool_name="t2")
    a_final = db.append_message("sess-mixed-tools", "assistant", "Done")

    # Archive only one tool result (mixed active/inactive tool-call group)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE messages SET active = 0, compacted = 1 WHERE id = ?", (t1,))

    # Deleting the archived tool result removes the active assistant call and active sibling too
    res = db.delete_message("sess-mixed-tools", t1, scope="message")
    assert res["deleted_ids"] == sorted([a_tc, t1, t2])
    assert res["remaining_message_count"] == 2
    assert [m["id"] for m in db.get_messages("sess-mixed-tools", include_inactive=True)] == [u1, a_final]
    sess = db.get_session("sess-mixed-tools")
    assert sess is not None
    assert sess["tool_call_count"] == 0


def test_reference_backend_matches_sqlite_for_meta_and_archived_tools(tmp_path: Path):
    from persistence import RowVisibility
    from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView
    from persistence.conversation.sessions import NewSession
    from persistence.core.identity import SessionId
    from persistence_assembly import profile_scope
    from tests.contracts._reference import open_reference_backend

    with open_reference_backend() as backend:
        store = backend.open(profile_scope(home=tmp_path))
        sid = SessionId("ref-sess")
        store.sessions.create(NewSession(session_id=sid, source="cli"))

        u1 = store.messages.append(sid, NewMessage(role="user", content="Question A"))
        a1 = store.messages.append(sid, NewMessage(role="assistant", content="Answer A"))
        meta = store.messages.append(sid, NewMessage(role="session_meta", content="meta"))
        u2 = store.messages.append(sid, NewMessage(role="user", content="Question B"))

        res_ex = store.messages.delete_message(sid, u1, scope=DeleteMessageScope.EXCHANGE)
        assert res_ex.deleted_ids == (u1, a1)
        assert res_ex.remaining_message_count == 2

        # Archived tool group in reference store
        a_tc = store.messages.append(
            sid,
            NewMessage(
                role="assistant",
                content="",
                tool_calls=(
                    {"id": "ref_c1", "type": "function", "function": {"name": "t1", "arguments": "{}"}},
                    {"id": "ref_c2", "type": "function", "function": {"name": "t2", "arguments": "{}"}},
                ),
            ),
        )
        t1 = store.messages.append(sid, NewMessage(role="tool", content="r1", tool_call_id="ref_c1", tool_name="t1"))
        t2 = store.messages.append(sid, NewMessage(role="tool", content="r2", tool_call_id="ref_c2", tool_name="t2"))
        # Mark tool group inactive (compaction-archived)
        transcript = store.messages._transcript(store.scope, sid)
        assert transcript is not None
        for row in transcript.rows:
            if row.id in {int(a_tc), int(t1), int(t2)}:
                row.visibility = RowVisibility.COMPACTION_ARCHIVED

        res_tc = store.messages.delete_message(sid, t1, scope=DeleteMessageScope.MESSAGE)
        assert res_tc.deleted_ids == tuple(sorted((a_tc, t1, t2)))
        assert res_tc.remaining_message_count == 2
        remaining_all = store.messages.read_transcript(sid, TranscriptRead(view=TranscriptView.ALL))
        assert [int(r.id) for r in remaining_all] == [int(meta), int(u2)]
