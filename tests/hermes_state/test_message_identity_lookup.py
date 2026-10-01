"""``find_user_message_by_client_message_id`` answers with the visible user row of a dispatched prompt.

A client retry the in-memory queue cannot answer (a restart, or the prompt already drained)
reconstructs its acknowledgement from this lookup. A busy-queue accept row still carrying the
never-drained marker (#125577) is not an acknowledgement — whether it is the live tip row, one
``reopen_session`` retired, or one a compression rotation or in-place compaction preserved:
answering with it would ack a prompt that never executes.
"""

from __future__ import annotations

from hermes_state import SessionDB
from hermes_state_common import QUEUED_PROMPT_METADATA_KEY


def test_lookup_follows_the_visible_row_for_a_client_message_id(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", "tui")
        accepted = db.append_message("s", "user", "hello", display_metadata={
            "client_message_id": "cid-1", QUEUED_PROMPT_METADATA_KEY: True})
        assert db.find_user_message_by_client_message_id("s", "cid-1") is None

        replacement = db.append_message("s", "user", "hello", display_metadata={"client_message_id": "cid-1"})
        db.deactivate_message("s", accepted)
        assert db.find_user_message_by_client_message_id("s", "cid-1")["_row_id"] == replacement

        db.deactivate_message("s", replacement)
        assert db.find_user_message_by_client_message_id("s", "cid-1") is None
    finally:
        db.close()


def test_a_never_run_accept_row_is_no_acknowledgement_whatever_compaction_preserves_it(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    marked = {"client_message_id": "cid-q", QUEUED_PROMPT_METADATA_KEY: True}
    try:
        # (a) rotation: the parent's marked original outlives reopen_session(child)
        db.create_session("parent", "tui")
        db.append_message("parent", "user", "first")
        wm = db.append_message("parent", "assistant", "working")
        db.append_message("parent", "user", "q", display_metadata=marked)
        db.publish_compression_child(
            parent_session_id="parent", child_session_id="child", source="tui",
            messages=[{"role": "user", "content": "summary"}], require_compression_lease=False, watermark=wm)
        db.reopen_session("child")
        assert db.find_user_message_by_client_message_id("child", "cid-q") is None
        ran = db.append_message("child", "user", "q", display_metadata={"client_message_id": "cid-q"})
        assert db.find_user_message_by_client_message_id("child", "cid-q")["_row_id"] == ran

        # (b) in-place compaction without coverage archives the marked row (active=0, compacted=1)
        db.create_session("s", "tui")
        db.append_message("s", "user", "first")
        db.append_message("s", "assistant", "working")
        db.append_message("s", "user", "q", display_metadata=marked)
        db.archive_and_compact("s", [{"role": "user", "content": "summary"}], watermark=None)
        db.reopen_session("s")
        assert db.find_user_message_by_client_message_id("s", "cid-q") is None
    finally:
        db.close()
