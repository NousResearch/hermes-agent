"""Origins are a bounded schema, not an arbitrary metadata bag."""
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc


def test_storage_bounds_excerpt_and_discards_non_origin_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="fixture", origin={
            "profile": "default", "session_id": "fixture-session", "message_row_id": 42,
            "chat_id": "x" * 10000, "prompt_excerpt": "quote " * 2000,
            "metadata": {"secret": "FAKE-NEVER-COPY"}, "images": ["FAKE-IMAGE"],
            "session_link": "javascript:alert(1)",
        })
        origin = kb.get_task(conn, tid).origin
        assert len(origin["prompt_excerpt"]) <= 2000
        assert "chat_id" not in origin  # oversized identifiers are omitted, never changed
        assert "metadata" not in origin and "images" not in origin
        assert origin["session_link"] == "/chat?resume=fixture-session&profile=default"
