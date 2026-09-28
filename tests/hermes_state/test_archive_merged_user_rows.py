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
