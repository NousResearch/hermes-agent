"""``messages.message_uid``: a durable per-message id that survives every host copy path.

The physical ``id`` is re-issued by every copy (in-place compaction generation, rotation child,
concurrent-tail clone, ``replace_messages``), and ``_row_id`` is opt-in on restore. A consumer that
keys on messages across restarts and boundaries (a context engine plugin) therefore had nothing
stable to key on and re-identified rows by content and timestamp. ``message_uid`` is minted once at
the row's first insert, stamped on the caller's dict, restored unconditionally, copied by every
clone, kept by every re-insert of the same dict and left alone by row-addressed rewrites.
"""

from __future__ import annotations

import copy
import json
import re

import pytest

from hermes_state import SessionDB

UID_RE = re.compile(r"^[0-9a-f]{32}$")


@pytest.fixture()
def db(tmp_path):
    handle = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield handle
    finally:
        handle.close()


def _rows(db, sid, *, active_only=True, include_session=False):
    clause = " AND active = 1" if active_only else ""
    cols = "id, session_id, role, content, message_uid, active, compacted"
    return [dict(r) for r in db._conn.execute(
        f"SELECT {cols} FROM messages WHERE session_id = ?{clause} ORDER BY id", (sid,)).fetchall()]


def _seed(db, sid, n=4):
    db.create_session(sid, "cli", model="m")
    for i in range(n):
        db.append_message(session_id=sid, role="user" if i % 2 == 0 else "assistant", content=f"msg {i}")
    return _rows(db, sid)


class TestMintAndRestore:
    def test_insert_mints_a_uid_and_stamps_the_callers_dict(self, db):
        db.create_session("s", "cli")
        msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        db.append_messages_batch("s", msgs)
        stored = _rows(db, "s")
        assert [r["message_uid"] for r in stored] == [m["message_uid"] for m in msgs]
        assert all(UID_RE.match(r["message_uid"]) for r in stored)

    def test_uid_is_per_occurrence_not_derived_from_content_or_time(self, db):
        db.create_session("s", "cli")
        twins = [{"role": "user", "content": "ok", "timestamp": 1_700_000_000.5},
                 {"role": "user", "content": "ok", "timestamp": 1_700_000_000.5}]
        db.append_messages_batch("s", twins)
        assert twins[0]["message_uid"] != twins[1]["message_uid"]

    def test_restore_carries_the_uid_without_include_row_ids(self, db):
        stored = _seed(db, "s")
        # The ACP / gateway / compression-adoption shape: no ``include_row_ids``.
        restored = db.get_messages_as_conversation("s", repair_alternation=True)
        assert "_row_id" not in restored[0]
        assert [m["message_uid"] for m in restored] == [r["message_uid"] for r in stored]
        # And every other projection (display resume, lineage) agrees.
        model_history, display_history = db.get_resume_conversations("s")
        assert [m["message_uid"] for m in model_history] == [r["message_uid"] for r in stored]
        assert [m["message_uid"] for m in display_history] == [r["message_uid"] for r in stored]

    def test_legacy_rows_are_backfilled_in_bounded_slices_across_opens(self, tmp_path, monkeypatch):
        """A single full-table UPDATE held the write lock for minutes on a large store; each open now
        mints at most one chunk past its time budget, keeps earlier uids, and converges."""
        import hermes_state_schema

        monkeypatch.setattr(hermes_state_schema, "_MESSAGE_UID_BACKFILL_CHUNK", 2)
        monkeypatch.setattr(hermes_state_schema, "_MESSAGE_UID_BACKFILL_BUDGET_S", 0.0)
        path = tmp_path / "legacy.db"
        first = SessionDB(db_path=path)
        try:
            _seed(first, "s", n=5)
            # Rows written by a build that predates the column: NULL uid, no backfill marker.
            first._conn.execute("UPDATE messages SET message_uid = NULL")
            first._conn.execute("DELETE FROM state_meta WHERE key = 'message_uid_backfill'")
            first._conn.commit()
        finally:
            first.close()
        seen = []
        for _ in range(3):
            handle = SessionDB(db_path=path)
            try:
                seen.append([r["message_uid"] for r in _rows(handle, "s")])
                done = handle.get_meta("message_uid_backfill")
            finally:
                handle.close()
        assert [sum(u is not None for u in uids) for uids in seen] == [2, 4, 5]
        assert seen[1][:2] == seen[0][:2] and seen[2][:4] == seen[1][:4]
        assert all(UID_RE.match(u) for u in seen[2]) and len(set(seen[2])) == 5
        assert done == "1"

    def test_a_row_inserted_without_a_uid_by_an_older_writer_gets_one(self, db):
        """The store mints for a build that predates the column (its INSERT binds no uid)."""
        db.create_session("s", "cli", model="test/model")
        db._conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES ('s', 'user', 'old writer', 1.0)"
        )
        db._conn.commit()
        uid = db._conn.execute("SELECT message_uid FROM messages WHERE content = 'old writer'").fetchone()[0]
        assert uid and UID_RE.match(uid)
        assert db.get_messages_as_conversation("s")[0]["message_uid"] == uid


class TestCopyPathsKeepTheUid:
    def test_in_place_compaction_keeps_uids_on_the_new_generation_and_the_tail_clone(self, db):
        stored = _seed(db, "s", n=4)
        watermark = stored[-1]["id"]
        # Concurrent appends after the compressor captured its watermark.
        db.append_message(session_id="s", role="user", content="late user")
        db.append_message(session_id="s", role="assistant", content="late assistant")
        late = _rows(db, "s")[4:]
        restored = db.get_messages_as_conversation("s")
        # The compressor's output: a summary the ENGINE already identified plus marker-swept COPIES of the
        # kept tail (an engine may pre-mint its own rows' uids; the host writes them as given).
        compacted = [{"role": "user", "content": "[CONTEXT COMPACTION] summary", "message_uid": "e" * 32},
                     copy.copy(restored[2]), copy.copy(restored[3])]
        db.archive_and_compact("s", compacted, watermark=watermark)
        active = _rows(db, "s")
        assert [r["content"] for r in active] == [
            "[CONTEXT COMPACTION] summary", "msg 2", "msg 3", "late user", "late assistant"]
        # New generation: the copies keep the archived originals' uids; the clone keeps the late rows' uids.
        assert [r["message_uid"] for r in active[1:3]] == [stored[2]["message_uid"], stored[3]["message_uid"]]
        assert [r["message_uid"] for r in active[3:]] == [r["message_uid"] for r in late]
        assert active[0]["message_uid"] == compacted[0]["message_uid"] == "e" * 32
        # The physical ids DID change (that is the whole point), the archived originals keep theirs too.
        assert {r["id"] for r in active}.isdisjoint({r["id"] for r in stored} | {r["id"] for r in late})
        archived = [r for r in _rows(db, "s", active_only=False) if not r["active"]]
        assert [r["message_uid"] for r in archived] == [r["message_uid"] for r in stored + late]

    def test_rotation_child_keeps_uids_on_handoff_copies_and_the_foreign_tail_clone(self, db):
        stored = _seed(db, "parent", n=2)
        watermark = stored[-1]["id"]
        db.append_message(session_id="parent", role="user", content="foreign append")
        foreign = _rows(db, "parent")[2]
        restored = db.get_messages_as_conversation("parent")
        handoff = [{"role": "user", "content": "[CONTEXT COMPACTION] summary", "message_uid": "e" * 32},
                   copy.copy(restored[1])]
        db.publish_compression_child(
            parent_session_id="parent", child_session_id="child", source="cli", messages=handoff,
            require_compression_lease=False, watermark=watermark, watermark_ceiling=foreign["id"])
        child = _rows(db, "child")
        assert [r["content"] for r in child] == ["[CONTEXT COMPACTION] summary", "msg 1", "foreign append"]
        assert child[1]["message_uid"] == stored[1]["message_uid"]
        assert child[2]["message_uid"] == foreign["message_uid"]
        assert child[0]["message_uid"] == "e" * 32
        # A restore of the child (what the resumed agent sees) carries the same uids.
        assert [m["message_uid"] for m in db.get_messages_as_conversation("child")] == [
            r["message_uid"] for r in child]

    def test_replace_messages_keeps_uids_on_the_kept_prefix_and_the_reissued_rows(self, db):
        stored = _seed(db, "s", n=4)
        restored = db.get_messages_as_conversation("s")
        # ACP non-owning persist / gateway rewrite: DELETE every active row and re-INSERT the history.
        db.replace_messages("s", restored + [{"role": "user", "content": "new"}], active_only=True)
        reissued = _rows(db, "s")
        assert [r["message_uid"] for r in reissued[:4]] == [r["message_uid"] for r in stored]
        assert reissued[4]["message_uid"] and reissued[4]["message_uid"] not in {r["message_uid"] for r in stored}
        assert {r["id"] for r in reissued[:4]}.isdisjoint({r["id"] for r in stored})
        # Rewind-style replace: the matched live prefix keeps its rows AND stamps their uids on the dicts.
        prefix = [{"role": m["role"], "content": m["content"]} for m in restored[:2]]
        db.replace_messages("s", prefix + [{"role": "user", "content": "edited"}], archive_dropped=True)
        assert [m["message_uid"] for m in prefix] == [r["message_uid"] for r in reissued[:2]]
        assert [r["message_uid"] for r in _rows(db, "s")[:2]] == [r["message_uid"] for r in reissued[:2]]

    def test_row_addressed_rewrite_keeps_the_uid(self, db):
        db.create_session("s", "cli")
        msg = {"role": "user", "content": "api variant of the prompt"}
        db.append_messages_batch("s", [msg])
        before = _rows(db, "s")[0]
        # The persist override / sanitizer rewrite: same dict, same row id and digest, new content.
        msg["content"] = "clean prompt"
        db.append_messages_batch("s", [msg])
        after = _rows(db, "s")
        assert len(after) == 1 and after[0]["id"] == before["id"]
        assert after[0]["content"] == "clean prompt"
        assert after[0]["message_uid"] == before["message_uid"] == msg["message_uid"]

    def test_rewrite_adopts_the_stored_uid_onto_a_dict_that_lacks_one(self, db):
        db.create_session("s", "cli")
        msg = {"role": "user", "content": "prompt"}
        db.append_messages_batch("s", [msg])
        stored_uid = msg.pop("message_uid")
        msg["content"] = "prompt (edited)"
        db.append_messages_batch("s", [msg])
        assert msg["message_uid"] == stored_uid
        assert [r["message_uid"] for r in _rows(db, "s")] == [stored_uid]

    def test_export_import_round_trip_keeps_the_uid(self, db, tmp_path):
        stored = _seed(db, "s", n=3)
        payload = db.export_session("s")
        assert [m["message_uid"] for m in payload["messages"]] == [r["message_uid"] for r in stored]
        other = SessionDB(db_path=tmp_path / "other.db")
        try:
            assert other.import_sessions([payload])["ok"]
            assert [r["message_uid"] for r in _rows(other, "s")] == [r["message_uid"] for r in stored]
        finally:
            other.close()


def _absorbed(db, sid):
    return [r[0] for r in db._conn.execute(
        "SELECT absorbed_message_uids FROM messages WHERE session_id = ? AND active = 1 ORDER BY id", (sid,))]


class TestPersistedMergeWitness:
    """``_absorbed_message_uids`` (which rows a consecutive-user merge folded into the survivor) rides on the
    survivor's row, so a consumer that restarts after the merge still knows the composite's constituents."""

    def test_flushed_survivor_persists_the_absorbed_uids_and_restore_returns_them(self, db):
        from agent.agent_runtime_helpers import _merge_consecutive_users

        db.create_session("s", "cli")
        dangling = {"role": "user", "content": "first, never answered"}
        db.append_messages_batch("s", [dangling])
        prompt = {"role": "user", "content": "second"}
        db.append_messages_batch("s", [prompt])
        _merge_consecutive_users([dangling, prompt])
        assert dangling["_absorbed_message_uids"] == [prompt["message_uid"]]
        # The survivor re-flushes as a row-addressed rewrite of its own row (same id, same uid).
        db.append_messages_batch("s", [dangling])
        rows = _rows(db, "s")
        assert [r["content"] for r in rows] == ["first, never answered\n\nsecond", "second"]
        assert rows[0]["message_uid"] == dangling["message_uid"]
        assert _absorbed(db, "s") == ['["%s"]' % prompt["message_uid"], None]
        restored = db.get_messages_as_conversation("s")
        assert restored[0]["_absorbed_message_uids"] == [prompt["message_uid"]]
        assert "_absorbed_message_uids" not in restored[1]

    def test_tool_call_uids_are_minted_on_the_assistant_and_paired_onto_the_results(self, db):
        db.create_session("s", "cli")
        calls = [{"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}},
                 {"id": "call_2", "type": "function", "function": {"name": "t", "arguments": "{}"}}]
        assistant = {"role": "assistant", "content": "", "tool_calls": calls}
        first = {"role": "tool", "content": "r1", "tool_call_id": "call_1", "tool_name": "t"}
        db.append_messages_batch("s", [assistant, first])
        uids = assistant["_tool_call_uids"]
        assert set(uids) == {"call_1", "call_2"} and len({*uids.values()}) == 2
        assert all(UID_RE.match(u) for u in uids.values())
        assert first["_tool_call_uid"] == uids["call_1"]
        # The provider-facing entries are untouched: the stored tool_calls JSON carries no uid.
        stored_calls = db._conn.execute(
            "SELECT tool_calls FROM messages WHERE session_id = ? AND role = 'assistant'", ("s",)).fetchone()[0]
        assert "uid" not in stored_calls and [c["id"] for c in calls] == ["call_1", "call_2"]
        # A result that lands in a LATER batch with no live list to pair it: stored NULL, derived on restore
        # from the assistant row that named it (rows are read in id order).
        second = {"role": "tool", "content": "r2", "tool_call_id": "call_2", "tool_name": "t"}
        db.append_messages_batch("s", [second])
        assert "_tool_call_uid" not in second
        restored = db.get_messages_as_conversation("s")
        assert restored[0]["_tool_call_uids"] == uids
        assert [m["_tool_call_uid"] for m in restored[1:]] == [uids["call_1"], uids["call_2"]]

    def test_tool_call_uids_survive_compaction_copies_rotation_and_import(self, db, tmp_path):
        db.create_session("s", "cli")
        assistant = {"role": "assistant", "content": "",
                     "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]}
        result = {"role": "tool", "content": "r", "tool_call_id": "call_1", "tool_name": "t"}
        db.append_messages_batch("s", [{"role": "user", "content": "q"}, assistant, result,
                                       {"role": "assistant", "content": "done"}])
        uid = assistant["_tool_call_uids"]["call_1"]
        restored = db.get_messages_as_conversation("s")
        db.archive_and_compact("s", [{"role": "user", "content": "[CONTEXT COMPACTION] s"},
                                     copy.copy(restored[1]), copy.copy(restored[2]), copy.copy(restored[3])])
        after = db.get_messages_as_conversation("s")
        assert after[1]["_tool_call_uids"] == {"call_1": uid} and after[2]["_tool_call_uid"] == uid
        db.publish_compression_child(
            parent_session_id="s", child_session_id="child", source="cli",
            messages=[copy.copy(m) for m in after], require_compression_lease=False)
        child = db.get_messages_as_conversation("child")
        assert child[1]["_tool_call_uids"] == {"call_1": uid} and child[2]["_tool_call_uid"] == uid
        other = SessionDB(db_path=tmp_path / "other.db")
        try:
            assert other.import_sessions([db.export_session("child")])["ok"]
            imported = other.get_messages_as_conversation("child")
            assert imported[1]["_tool_call_uids"] == {"call_1": uid} and imported[2]["_tool_call_uid"] == uid
        finally:
            other.close()

    def test_a_reused_provider_id_pairs_with_the_nearest_call_only(self, db):
        """Provider ids repeat: on restore a result binds to the NEAREST preceding call; a legacy call row
        without a map (an older writer's) leaves its result unpaired instead of inheriting an older uid, and
        nothing pairs across a user turn."""
        sid = "s"
        db.create_session(sid, "cli", model="m")
        call = [{"id": "call_x", "type": "function", "function": {"name": "t", "arguments": "{}"}}]
        db.append_message(session_id=sid, role="user", content="q1")
        db.append_message(session_id=sid, role="assistant", content="", tool_calls=call)
        db.append_message(session_id=sid, role="tool", content="r1", tool_call_id="call_x", tool_name="t")
        db.append_message(session_id=sid, role="user", content="q2")
        # An older writer's rows: same provider id, no uid map on the call, no uid on the result.
        db._conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, timestamp) VALUES (?, 'assistant', '', ?, 5.0)",
            (sid, json.dumps(call)))
        db._conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_call_id, tool_name, timestamp) "
            "VALUES (?, 'tool', 'r2', 'call_x', 't', 6.0)", (sid,))
        db._conn.commit()
        db.append_message(session_id=sid, role="user", content="q3")
        db.append_message(session_id=sid, role="assistant", content="", tool_calls=call)
        db.append_message(session_id=sid, role="tool", content="r3", tool_call_id="call_x", tool_name="t")
        db.append_message(session_id=sid, role="user", content="q4")
        db.append_message(session_id=sid, role="tool", content="stray", tool_call_id="call_x", tool_name="t")

        restored = db.get_messages_as_conversation(sid, repair_alternation=False)
        by_content = {m["content"]: m for m in restored}
        uid1 = by_content["r1"]["_tool_call_uid"]
        uid3 = by_content["r3"]["_tool_call_uid"]
        assert UID_RE.match(uid1) and UID_RE.match(uid3) and uid1 != uid3
        assert "_tool_call_uid" not in by_content["r2"]
        assert "_tool_call_uid" not in by_content["stray"]
        # The same rule at batch insert: a result after a user turn pairs with nothing.
        assert db._conn.execute(
            "SELECT tool_call_uid FROM messages WHERE content = 'stray'").fetchone()[0] is None

    @pytest.mark.parametrize("slots", [["1" * 32, None], [None, "1" * 32]], ids=["mapped-then-legacy",
                                                                                 "legacy-then-mapped"])
    def test_a_mixed_era_folds_unmapped_slot_survives_insert_rewrite_and_export_import(self, db, tmp_path, slots):
        """A fold of a mapped turn with an older build's unmapped one keeps ``None`` for the unmapped
        occurrence. Every write path keeps that slot (dropping the list would lose the mapped occurrence's
        uid; minting over it would invent one), and a restore pairs the result with the LAST occurrence's."""
        db.create_session("s", "cli")
        survivor = {"role": "assistant", "content": "a", "tool_calls": [_call("call_x"), _call("call_x")],
                    "_tool_call_uids": {"call_x": list(slots)}}
        db.append_messages_batch("s", [{"role": "user", "content": "q"}, survivor])
        db._conn.execute(  # a result stored without a uid: restore derives it from the composite
            "INSERT INTO messages (session_id, role, content, tool_call_id, tool_name, timestamp) "
            "VALUES ('s', 'tool', 'r', 'call_x', 't', 9.0)")
        db._conn.commit()
        assert survivor["_tool_call_uids"] == {"call_x": slots}

        def check(store, sid="s"):
            restored = store.get_messages_as_conversation(sid, repair_alternation=False, include_row_ids=True)
            assert restored[1]["_tool_call_uids"] == {"call_x": slots}
            assert restored[2].get("_tool_call_uid") == slots[-1]

        check(db)
        survivor["content"] = "a, rewritten"  # the same dict again: a row-addressed rewrite in place
        db.append_messages_batch("s", [survivor])
        assert [r["content"] for r in _rows(db, "s")] == ["q", "a, rewritten", "r"]
        assert json.loads(db._conn.execute(
            "SELECT tool_call_uids FROM messages WHERE role = 'assistant'").fetchone()[0]) == {"call_x": slots}
        check(db)
        other = SessionDB(db_path=tmp_path / "other.db")
        try:
            assert other.import_sessions([db.export_session("s")])["ok"]
            check(other)
        finally:
            other.close()

    def test_an_all_unmapped_occurrence_list_is_no_map(self, db):
        db.create_session("s", "cli")
        db.append_messages_batch("s", [{"role": "assistant", "content": "a", "tool_calls": [_call("c1")],
                                        "_tool_call_uids": {"c1": "1" * 32, "call_x": [None, None]}}])
        assert json.loads(db._conn.execute("SELECT tool_call_uids FROM messages").fetchone()[0]) == {"c1": "1" * 32}

    def test_a_composite_rewind_reports_the_replacement_rows_uid(self, db):
        from agent.context_compressor import HISTORICAL_TASK_HEADING, SUMMARY_PREFIX, _SUMMARY_END_MARKER

        sid = "s"
        db.create_session(sid, "cli", model="m")
        db.append_message(session_id=sid, role="user", content="older ask")
        db.append_message(session_id=sid, role="assistant", content="done")
        carrier = f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n{_SUMMARY_END_MARKER}\n\nREAL ASK"
        target_id = db.append_message(session_id=sid, role="user", content=carrier)
        db.append_message(session_id=sid, role="assistant", content="failed")

        result = db.rewind_to_message(sid, target_id, preserve_compaction_handoff=True,
                                      expected_target_content="REAL ASK")

        head = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]
        assert head["_row_id"] == result["replacement_message_id"]
        assert result["replacement_message_uid"] == head["message_uid"]
        assert UID_RE.match(head["message_uid"])

    def test_a_live_undo_of_a_composite_turn_installs_the_replacement_rows_identity(self, db):
        """The CLI/TUI/gateway undo path (``rewind_user_turn``) installs the hidden scaffold as the live head:
        it must carry the replacement row's uid, not wait for a restart to learn it."""
        from agent.context_compressor import HISTORICAL_TASK_HEADING, SUMMARY_PREFIX, _SUMMARY_END_MARKER

        sid = "s"
        db.create_session(sid, "cli", model="m")
        db.append_message(session_id=sid, role="user", content="q1")
        db.append_message(session_id=sid, role="assistant", content="a1")
        carrier = f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n{_SUMMARY_END_MARKER}\n\nREAL ASK"
        db.append_message(session_id=sid, role="user", content=carrier)
        db.append_message(session_id=sid, role="assistant", content="failed")
        warm = db.get_resume_conversations(sid)[0]

        outcome = db.rewind_user_turn(sid, 1, warm_history=warm)

        head = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]
        assert head["display_kind"] == "hidden" and SUMMARY_PREFIX in head["content"]
        assert outcome.prefix[-1]["_row_id"] == head["_row_id"]
        assert outcome.prefix[-1]["message_uid"] == head["message_uid"]
        assert UID_RE.match(head["message_uid"])

    def test_foreign_history_import_mints_uids_and_keeps_supplied_ones(self, db):
        origin = {"tool": "other-agent", "path": "/x/y.jsonl"}
        supplied = "f" * 32
        result = db.import_foreign_history(
            origin, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello", "message_uid": supplied}],
            title="imported", cwd="/tmp", profile="default")
        rows = _rows(db, result["session_id"])
        assert [r["content"] for r in rows] == ["hi", "hello"]
        assert UID_RE.match(rows[0]["message_uid"]) and rows[1]["message_uid"] == supplied
        assert [m["message_uid"] for m in db.get_messages_as_conversation(result["session_id"])] == [
            rows[0]["message_uid"], supplied]

    def test_a_restore_time_merge_records_the_witness(self, db):
        """Two stored user rows in a row (a turn that never got an answer) are merged while restoring with
        ``repair_alternation=True``; the survivor names the absorbed row by uid, on every restore shape."""
        db.create_session("s", "cli", model="m")
        db.append_message(session_id="s", role="user", content="first ask")
        db.append_message(session_id="s", role="user", content="second ask")
        db.append_message(session_id="s", role="assistant", content="answer")
        first, second = [r["message_uid"] for r in _rows(db, "s")][:2]
        for kwargs in ({}, {"include_row_ids": True}):
            restored = db.get_messages_as_conversation("s", repair_alternation=True, **kwargs)
            assert [m["role"] for m in restored] == ["user", "assistant"]
            assert restored[0]["content"] == "first ask\n\nsecond ask"
            assert restored[0]["message_uid"] == first
            assert restored[0]["_absorbed_message_uids"] == [second]
        assert db.get_resume_conversations("s")[0][0]["_absorbed_message_uids"] == [second]

    def test_a_row_rewrite_keeps_the_tool_call_uids(self, db):
        db.create_session("s", "cli")
        assistant = {"role": "assistant", "content": "partial",
                     "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]}
        db.append_messages_batch("s", [assistant])
        uids = assistant.pop("_tool_call_uids")  # a restored/cloned dict that lost its map: the row keeps it
        assistant["content"] = "partial, then filled by the sanitizer"
        db.append_messages_batch("s", [assistant])
        rows = _rows(db, "s")
        assert len(rows) == 1 and rows[0]["content"] == "partial, then filled by the sanitizer"
        assert db.get_messages_as_conversation("s")[0]["_tool_call_uids"] == uids

    def test_compaction_copy_and_export_import_keep_the_witness(self, db, tmp_path):
        db.create_session("s", "cli")
        composite = {"role": "user", "content": "a\n\nb", "message_uid": "a" * 32,
                     "_absorbed_message_uids": ["b" * 32]}
        db.append_messages_batch("s", [composite, {"role": "assistant", "content": "ok"}])
        restored = db.get_messages_as_conversation("s")
        db.archive_and_compact("s", [{"role": "user", "content": "[CONTEXT COMPACTION] s"},
                                     copy.copy(restored[0]), copy.copy(restored[1])])
        assert _absorbed(db, "s") == [None, '["%s"]' % ("b" * 32), None]
        payload = db.export_session("s")
        other = SessionDB(db_path=tmp_path / "other.db")
        try:
            assert other.import_sessions([payload])["ok"]
            assert other.get_messages_as_conversation("s")[1]["_absorbed_message_uids"] == ["b" * 32]
        finally:
            other.close()


def test_a_branch_copy_writes_the_uids_its_live_history_keeps(tmp_path):
    """The branch child keeps using the copied live dicts; rows minted with other uids would restore a
    different identity than the running agent saw. A dict without a uid gets one on both sides."""
    from tui_gateway import server

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("parent", source="desktop", model="m")
        history = [{"role": "user", "content": "hello", "message_uid": "a" * 32}, {"role": "assistant", "content": "hi"}]
        server._persist_branch(db, "child", "parent", "Branch", history, source="desktop", cwd=str(tmp_path),
                               profile_name="default", model="m")
        assert [r["message_uid"] for r in _rows(db, "child")] == [m.get("message_uid") for m in history]
        assert history[0]["message_uid"] == "a" * 32 and UID_RE.match(history[1]["message_uid"])
    finally:
        db.close()


def _call(cid):
    return {"id": cid, "type": "function", "function": {"name": "t", "arguments": "{}"}}


def test_a_result_pairs_with_the_assistant_that_named_it_not_the_nearest_one(db):
    """Restore (results without a stored uid) and the flush walk both skip a nearer assistant that never
    named the result's call, and pair with the one that did."""
    from agent.message_metadata import tool_call_uid_from_history

    db.create_session("s", "cli")
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [_call("call_1")]},
            {"role": "assistant", "content": "", "tool_calls": [_call("call_2")]},
            {"role": "tool", "content": "r2", "tool_call_id": "call_2"},
            {"role": "tool", "content": "r1", "tool_call_id": "call_1"}]
    db.append_messages_batch("s", msgs)
    first, second = msgs[1]["_tool_call_uids"]["call_1"], msgs[2]["_tool_call_uids"]["call_2"]
    db._conn.execute("UPDATE messages SET tool_call_uid = NULL")  # results written by an older build
    db._conn.commit()
    restored = [m for m in db.get_messages_as_conversation("s", repair_alternation=False) if m["role"] == "tool"]
    assert [m.get("_tool_call_uid") for m in restored] == [second, first]
    live = [{k: v for k, v in m.items() if k != "_tool_call_uid"} for m in msgs]
    owners: dict = {}
    assert [tool_call_uid_from_history(live, i, owners) for i in (3, 4)] == [second, first]


def test_one_response_repeating_a_provider_id_shares_one_uid_its_results_all_carry(db):
    """Two calls sharing a provider id in ONE response share one uid: both results carry it, so neither call
    looks unanswered to a context engine."""
    db.create_session("s", "cli")
    call = {"role": "assistant", "content": "", "tool_calls": [_call("call_x"), _call("call_x")]}
    results = [{"role": "tool", "content": c, "tool_call_id": "call_x", "tool_name": "t"} for c in "ab"]
    db.append_messages_batch("s", [{"role": "user", "content": "q"}, call, *results])
    shared = call["_tool_call_uids"]["call_x"]
    assert isinstance(shared, str) and [r["_tool_call_uid"] for r in results] == [shared, shared]


def test_a_fold_after_a_shared_uid_row_keeps_each_call_aligned_with_its_uid():
    """The shared uid fills each of the row's own slots before the absorbed turn's occurrence is appended, so
    the absorbed call (third) keeps its own uid instead of sliding onto the second."""
    from agent.agent_runtime_helpers import _merge_consecutive_assistants
    from agent.message_metadata import index_tool_call_uids, resolve_tool_call_uid

    shared, own = "1" * 32, "2" * 32
    first = {"role": "assistant", "content": "", "tool_calls": [_call("call_x"), _call("call_x")],
             "_tool_call_uids": {"call_x": shared}}
    second = {"role": "assistant", "content": "", "tool_calls": [_call("call_x")], "_tool_call_uids": {"call_x": own}}
    _merge_consecutive_assistants([first, second])
    assert first["_tool_call_uids"] == {"call_x": [shared, shared, own]}
    index: dict = {}
    index_tool_call_uids(index, first)
    assert resolve_tool_call_uid(index, "call_x") == own


def test_a_rewrite_keeps_a_fold_that_extends_a_stored_occurrence_list(db):
    db.create_session("s", "cli")
    a = {"role": "assistant", "content": "a", "tool_calls": [_call("call_0"), _call("call_0")],
         "_tool_call_uids": {"call_0": ["1" * 32, "2" * 32]}}
    b = {"role": "assistant", "content": "b", "tool_calls": [_call("call_0")], "_tool_call_uids": {"call_0": "3" * 32}}
    db.append_messages_batch("s", [{"role": "user", "content": "q"}, a, b])
    survivor = db.get_messages_as_conversation("s", repair_alternation=True, include_row_ids=True)[1]
    survivor.pop("_db_row_snapshot", None)
    db.append_messages_batch("s", [survivor])
    assert survivor["_tool_call_uids"] == {"call_0": ["1" * 32, "2" * 32, "3" * 32]}


def test_a_rewrite_that_drops_a_call_does_not_write_its_uid_back(db):
    db.create_session("s", "cli")
    assistant = {"role": "assistant", "content": "x", "tool_calls": [_call("c1"), _call("c2")]}
    db.append_messages_batch("s", [assistant])
    assistant["tool_calls"] = [_call("c1")]
    assistant["_tool_call_uids"] = {"c1": assistant["_tool_call_uids"]["c1"]}
    db.append_messages_batch("s", [assistant])
    stored = db._conn.execute("SELECT tool_call_uids FROM messages").fetchone()[0]
    assert sorted(json.loads(stored)) == ["c1"]


def test_a_rewrite_keeps_a_folds_per_occurrence_list_over_the_stored_single_uid(db):
    """Restore folds two stored turns that reuse an id; re-flushing the survivor must not let the stored
    pre-fold map collapse the list back to one uid (the later occurrence's results would pair with nothing)."""
    db.create_session("s", "cli")
    a = {"role": "assistant", "content": "a", "tool_calls": [_call("call_0")]}
    b = {"role": "assistant", "content": "b", "tool_calls": [_call("call_0")]}
    db.append_messages_batch("s", [{"role": "user", "content": "q"}, a, b])
    occurrences = [a["_tool_call_uids"]["call_0"], b["_tool_call_uids"]["call_0"]]
    survivor = db.get_messages_as_conversation("s", repair_alternation=True, include_row_ids=True)[1]
    survivor.pop("_db_row_snapshot", None)  # the digest-less re-flush path (replay heal, adopt mismatch)
    assert survivor["_tool_call_uids"] == {"call_0": occurrences}
    db.append_messages_batch("s", [survivor])
    assert survivor["_tool_call_uids"] == {"call_0": occurrences}


def test_a_chunked_copy_pairs_a_result_with_a_call_in_the_previous_chunk(db):
    db.create_session("s", "cli")
    call = {"role": "assistant", "content": "", "tool_calls": [_call("call_1")]}
    result = {"role": "tool", "content": "r", "tool_call_id": "call_1", "tool_name": "t"}
    db.append_messages_batch("s", [{"role": "user", "content": "q"}, call, result], chunk_rows=2)
    stored = db._conn.execute("SELECT tool_call_uid FROM messages WHERE role = 'tool'").fetchone()[0]
    assert stored == call["_tool_call_uids"]["call_1"]
