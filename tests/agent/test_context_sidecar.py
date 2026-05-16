import json

from agent.context_dag_store import ContextDAGStore
from agent.context_dag_tools import expand_context
from agent.context_sidecar import SidecarStore, project_tool_output_with_sidecar
from hermes_state import SessionDB


def _db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", "test")
    db.create_session("s2", "test")
    return db


def test_large_tool_output_sidecar_keeps_raw_message_intact_and_projects_preview(tmp_path):
    db = _db(tmp_path)
    try:
        content = "line1\n" + ("x" * 200)
        msg_id = db.append_message("s1", "tool", content, tool_name="big_tool")
        store = ContextDAGStore(db)

        projection = project_tool_output_with_sidecar(
            store,
            session_id="s1",
            message_id=msg_id,
            content=content,
            preview_chars=16,
            threshold_bytes=32,
        )

        assert projection["content"] == "line1\nxxxxxxxxxx"
        assert projection["sidecar"]["message_id"] == msg_id
        assert projection["sidecar"]["line_count"] == 2
        assert projection["sidecar"]["size_bytes"] == len(content.encode("utf-8"))
        assert projection["sidecar"]["sha256"]
        assert projection["sidecar"]["ref"].startswith("sidecar://")
        assert db.get_messages("s1")[0]["content"] == content
    finally:
        db.close()


def test_sidecar_write_is_idempotent_for_same_message_part(tmp_path):
    db = _db(tmp_path)
    try:
        msg_id = db.append_message("s1", "tool", "full output")
        sidecar = SidecarStore(ContextDAGStore(db))

        first = sidecar.write_message_part("s1", msg_id, "full output", part_index=0)
        second = sidecar.write_message_part("s1", msg_id, "full output", part_index=0)

        assert second == first
        with db._lock:
            count = db._conn.execute(
                "SELECT COUNT(*) FROM context_message_parts WHERE message_id = ?",
                (msg_id,),
            ).fetchone()[0]
        assert count == 1
    finally:
        db.close()


def test_expand_context_pages_full_sidecar_content_for_current_session(tmp_path):
    db = _db(tmp_path)
    try:
        content = "abcdef" * 20
        msg_id = db.append_message("s1", "tool", content)
        sidecar = SidecarStore(ContextDAGStore(db))
        sidecar.write_message_part("s1", msg_id, content, part_index=0)

        payload = expand_context(
            ContextDAGStore(db),
            session_id="s1",
            message_id=msg_id,
            max_chars=20,
        )

        assert payload["ok"] is True
        message = payload["messages"][0]
        assert message["content"] == content[:20]
        assert message["sidecar_parts"][0]["content"] == content[:20]
        assert message["sidecar_parts"][0]["truncated"] is True
        assert message["sidecar_parts"][0]["omitted_chars"] == len(content) - 20
    finally:
        db.close()


def test_sidecar_rejects_cross_session_message_ids(tmp_path):
    db = _db(tmp_path)
    try:
        msg_id = db.append_message("s1", "tool", "private")
        sidecar = SidecarStore(ContextDAGStore(db))

        try:
            sidecar.write_message_part("s2", msg_id, "private")
        except ValueError as exc:
            assert "belong to session" in str(exc)
        else:
            raise AssertionError("cross-session sidecar write was allowed")
    finally:
        db.close()


def test_inline_small_outputs_do_not_create_sidecar_parts(tmp_path):
    db = _db(tmp_path)
    try:
        msg_id = db.append_message("s1", "tool", "small")
        projection = project_tool_output_with_sidecar(
            ContextDAGStore(db),
            session_id="s1",
            message_id=msg_id,
            content="small",
            threshold_bytes=100,
        )

        assert projection["content"] == "small"
        assert "sidecar" not in projection
        with db._lock:
            count = db._conn.execute("SELECT COUNT(*) FROM context_message_parts").fetchone()[0]
        assert count == 0
    finally:
        db.close()
