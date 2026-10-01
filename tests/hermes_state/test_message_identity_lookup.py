"""``find_user_message_by_client_message_id`` answers with the row the transcript currently shows.

A client retry after a restart reconstructs its acknowledgement from this lookup. A busy-queue
accept row that a drain replaced, or that ``reopen_session`` retired because it never ran
(#125577), is not an acknowledgement: answering with it would ack a prompt that never executes.
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
        assert db.find_user_message_by_client_message_id("s", "cid-1")["_row_id"] == accepted

        replacement = db.append_message("s", "user", "hello", display_metadata={"client_message_id": "cid-1"})
        db.deactivate_message("s", accepted)
        assert db.find_user_message_by_client_message_id("s", "cid-1")["_row_id"] == replacement

        db.deactivate_message("s", replacement)
        assert db.find_user_message_by_client_message_id("s", "cid-1") is None
    finally:
        db.close()
