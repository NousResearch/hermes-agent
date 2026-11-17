"""A compressed conversation's list row reports what its running turn is doing.

After compression the agent stamps the continuation and the ended root's label is cleared, so a
row that kept the root's ``last_activity_description`` lost its working subtitle (Desktop sidebar,
dashboard Sessions page) the moment a long turn compressed.
"""
from hermes_state import SessionDB


def test_a_compressed_row_carries_the_continuations_activity(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="cli")
    db.append_message("root", role="user", content="hi")
    db.end_session("root", "compression")
    db.create_session("tip", source="cli", parent_session_id="root")
    db.append_message("tip", role="user", content="continued")
    db.touch_session_activity("tip", description="tool running: terminal")

    [row] = db.list_sessions_rich(compact_rows=True, order_by_last_active=True)

    assert (row["id"], row["last_activity_description"]) == ("tip", "tool running: terminal")
