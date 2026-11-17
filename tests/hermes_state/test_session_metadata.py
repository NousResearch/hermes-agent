"""Client metadata on sessions: merge-patch writes, lineage-wide like a pin, exact-match list filter."""
from contextlib import closing

import pytest

from hermes_state import SessionDB
from hermes_state_session_metadata import SessionMetadataError, merge_session_metadata


def test_metadata_merge_patch_follows_the_chat_through_compression_and_filters_the_listing(tmp_path):
    with closing(SessionDB(tmp_path / "state.db")) as db:
        db.create_session("chat", source="api_server")
        db.create_session("other", source="api_server")
        assert merge_session_metadata(db, "chat", {"ticket": "ENG-1", "team": "infra"}) == {
            "ticket": "ENG-1", "team": "infra"}
        assert merge_session_metadata(db, "other", {"ticket": "ENG-2"}) == {"ticket": "ENG-2"}

        # A compression continuation carries the conversation's metadata, and a later patch
        # through either id lands on the whole lineage: the listing projects the root onto its tip.
        db.publish_compression_child(parent_session_id="chat", child_session_id="chat-2", source="api_server",
                                     messages=[{"role": "user", "content": "[summary]"}],
                                     require_compression_lease=False)
        assert merge_session_metadata(db, "chat-2", {"team": None, "ci.run": "7"}) == {
            "ticket": "ENG-1", "ci.run": "7"}
        assert '"ci.run"' in db.get_session("chat")["metadata"]

        listed = db.list_sessions_rich(metadata={"ticket": "ENG-1", "ci.run": "7"})
        assert [s["id"] for s in listed] == ["chat-2"]
        assert db.session_count(exclude_children=True, metadata={"ticket": "ENG-1", "ci.run": "7"}) == 1
        assert db.list_sessions_rich(metadata={"ticket": "ENG-1", "team": "infra"}) == []

        with pytest.raises(SessionMetadataError):
            merge_session_metadata(db, "other", {f"k{i}": "v" for i in range(32)})
        assert merge_session_metadata(db, "missing", {"a": "b"}) is None
