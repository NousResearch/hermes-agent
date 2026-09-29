"""A held dict may only name rows the compressor could have been handed.

The commit resolves a merged user dict to the durable ``user;user`` run behind it. A prompt that was
never persisted can carry the same text as rows another surface appended after the snapshot.
"""

from hermes_state import SessionDB


def test_unpersisted_prompt_leaves_rows_appended_after_the_snapshot_live(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sid", source="test")
    held = [db.append_message("sid", "user", "U1"), db.append_message("sid", "assistant", "A1")]
    watermark = db.get_active_message_watermark("sid")
    db.append_message("sid", "user", "same first paragraph")
    db.append_message("sid", "user", "same second paragraph")
    prompt = {"role": "user", "content": "same first paragraph\n\nsame second paragraph"}

    db.archive_and_compact(
        "sid", [{"role": "user", "content": "summary"}, {"role": "assistant", "content": "ok"}, prompt],
        watermark=watermark, covered_ids=held, unresolved_held=[prompt])

    assert [row["content"] for row in db.get_messages("sid")] == [
        "summary", "ok", prompt["content"], "same first paragraph", "same second paragraph"]
    db.close()


def _unanswered_pair(tmp_path, *after):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sid", source="test")
    for role, content in (("user", "first prompt"), ("user", "second prompt"), *after):
        db.append_message("sid", role, content)
    return db, db.get_active_message_watermark("sid")


def _live(db):
    return [row["content"] for row in db.get_messages("sid")]


def test_carried_merged_prompt_rendered_with_a_timestamp_keeps_one_copy_of_each_row(tmp_path):
    from gateway.run import _build_gateway_agent_history

    db, watermark = _unanswered_pair(tmp_path)
    restored = db.get_messages_as_conversation("sid", repair_alternation=True)
    held, _ = _build_gateway_agent_history(restored, inject_timestamps=True)
    assert held[0]["content"] != restored[0]["content"]  # the render is what hides the rows

    db.archive_and_compact(
        "sid", [{"role": "assistant", "content": "summary"}, held[0]], watermark=watermark, tail_count=1,
        covered_ids=[], unresolved_held=held)

    assert _live(db) == ["summary", held[0]["content"]]
    recalled = " ".join(row["content"] for row in db._conn.execute(
        "SELECT content FROM messages WHERE session_id = 'sid' AND (active = 1 OR compacted = 1)").fetchall())
    assert (recalled.count("first prompt"), recalled.count("second prompt")) == (1, 1)
    db.close()


def test_merged_prompt_never_names_a_run_appended_after_the_snapshot(tmp_path):
    from gateway.run import _build_gateway_agent_history

    db, watermark = _unanswered_pair(tmp_path)
    restored = db.get_messages_as_conversation("sid", repair_alternation=True)
    held, _ = _build_gateway_agent_history(restored, inject_timestamps=True)
    late = held[0]["content"].split("\n\n")
    assert len(late) == 2 and late[0] != "first prompt"  # only the late run joins to the rendered text
    for part in late:
        db.append_message("sid", "user", part)

    db.archive_and_compact(
        "sid", [{"role": "assistant", "content": "summary"}, held[0]], watermark=watermark, tail_count=1,
        covered_ids=[], unresolved_held=held)

    assert _live(db) == ["summary", held[0]["content"], *late]
    db.close()


def test_merged_prompt_names_its_pair_not_a_later_row_with_the_merged_text(tmp_path):
    db, watermark = _unanswered_pair(tmp_path, ("assistant", "A1"))
    held = db.get_messages_as_conversation("sid", repair_alternation=True)
    for message in held:
        message.pop("_row_id", None)
        message.pop("_absorbed_row_ids", None)
    db.append_message("sid", "user", held[0]["content"])

    db.archive_and_compact(
        "sid", [{"role": "user", "content": "summary"}], watermark=watermark, covered_ids=[], unresolved_held=held)

    assert _live(db) == ["summary", "first prompt\n\nsecond prompt"]
    db.close()
