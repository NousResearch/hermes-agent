"""Placeholder-first creation must not hide real CLI sessions (#138831)."""
import pytest

from hermes_state import SessionDB
from hermes_cli.gateway_session_store import SurfaceLabelReader
from hermes_cli.session_listing import query_session_listing


@pytest.mark.parametrize("surface", ["cli", "tui", "gui"])
def test_placeholder_creator_remains_visible_after_gateway_routing(tmp_path, surface):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.update_token_counts("previous", input_tokens=5)
        db.create_session("previous", source=surface)
        db.set_session_title("previous", "Previous conversation")
        db.record_gateway_session_peer(
            "previous", source="local", session_key="local:previous", chat_id="previous",
        )
        rows = query_session_listing(SurfaceLabelReader(db), source=surface)
        assert [row["id"] for row in rows] == ["previous"]
        assert db.get_session("previous")["created_source"] == surface
        assert db.get_session("previous")["input_tokens"] == 5
    finally:
        db.close()


def test_known_provenance_survives_later_creators_and_accounting(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("child", source="subagent")
        db.update_token_counts("child", input_tokens=5)
        db.create_session("child", source="cli")
        db.record_gateway_session_peer("child", source="local", session_key="local:child")
        assert db.get_session("child")["created_source"] == "subagent"
        assert SurfaceLabelReader(db).search_sessions(source="cli") == []
        db.update_token_counts("pending", input_tokens=1)
        db.update_token_counts("pending", input_tokens=2)
        assert db.get_session("pending")["created_source"] == "unknown"
        assert db.get_session("pending")["input_tokens"] == 3
        db.create_session("legacy", source="cli")
        db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET created_source=NULL WHERE id=?", ("legacy",),
        ))
        db.create_session("legacy", source="tui")
        assert db.get_session("legacy")["created_source"] == "tui"
    finally:
        db.close()
