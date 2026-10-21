"""Placeholder-first creation must not hide real CLI sessions (#138831)."""
from hermes_state import SessionDB
from hermes_cli.gateway_session_store import SurfaceLabelReader
from hermes_cli.session_listing import query_session_listing


def test_placeholder_creator_remains_visible_after_gateway_routing(tmp_path):
    surface = "cli"
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


def test_routing_label_local_is_not_creation_provenance(tmp_path):
    """An upsert carrying the routing label ``local`` never stamps it, and a row that already holds
    ``local`` (or the ``unknown`` placeholder) is repaired by the real creator."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.update_token_counts("placeholder", input_tokens=1)
        db.create_session("placeholder", source="local")
        assert db.get_session("placeholder")["created_source"] == "unknown"
        db.create_session("frozen", source="local")
        assert db.get_session("frozen")["created_source"] == "local"
        for sid in ("placeholder", "frozen"):
            db.create_session(sid, source="cli")
            assert db.get_session(sid)["created_source"] == "cli", sid
    finally:
        db.close()
