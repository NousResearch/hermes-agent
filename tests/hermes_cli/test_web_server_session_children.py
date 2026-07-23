import asyncio
import time

from hermes_cli import web_server
from hermes_state import SessionDB


class _FakeSessionDB:
    def __init__(self):
        self.closed = False
        self.include_stale = None

    def resolve_session_id(self, session_id):
        assert session_id == "parent-prefix"
        return "parent"

    def get_session_children(self, session_id, *, include_stale=False):
        assert session_id == "parent"
        self.include_stale = include_stale
        return {
            "parent_session_id": "parent",
            "focused": [{"id": "focused"}],
            "branches": [],
            "subagents": {
                "active": [],
                "completed": [{"id": "audit"}],
                "stale": [],
                "stale_count": 1,
            },
            "other": [],
        }

    def close(self):
        self.closed = True


def test_session_children_endpoint_returns_grouped_children(monkeypatch):
    fake_db = _FakeSessionDB()
    monkeypatch.setattr(
        web_server,
        "_open_session_db_for_profile",
        lambda profile=None: fake_db,
    )

    response = asyncio.run(
        web_server.get_session_children_endpoint(
            "parent-prefix",
            include_stale=True,
            profile="default",
        )
    )

    assert fake_db.include_stale is True
    assert fake_db.closed is True
    assert [row["id"] for row in response["focused"]] == ["focused"]
    assert [row["id"] for row in response["subagents"]["completed"]] == ["audit"]
    assert response["subagents"]["stale_count"] == 1


def test_latest_descendant_prefers_compression_continuation_over_newer_delegate(
    monkeypatch,
    tmp_path,
):
    db = SessionDB(db_path=tmp_path / "state.db")
    base = int(time.time()) - 10_000
    try:
        db.create_session("root", source="cli")
        db.append_message("root", "user", "pre-compression")
        db.end_session("root", "compression")
        db.create_session("focused_cont", source="cli", parent_session_id="root")
        db.append_message("focused_cont", "assistant", "real continuation")
        db.create_session(
            "delegate_subagent",
            source="subagent",
            parent_session_id="root",
            model_config={"_delegate_from": "root"},
        )
        db.append_message("delegate_subagent", "assistant", "newer audit child")

        conn = db._conn
        assert conn is not None
        conn.execute(
            "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = 'root'",
            (base, base + 100),
        )
        conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = 'focused_cont'",
            (base + 150,),
        )
        conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = 'delegate_subagent'",
            (base + 300,),
        )
        conn.commit()

        monkeypatch.setattr(
            web_server,
            "_open_session_db_for_profile",
            lambda profile=None: db,
        )

        response = asyncio.run(web_server.get_session_latest_descendant("root"))

        assert response["session_id"] == "focused_cont"
        assert response["path"] == ["root", "focused_cont"]
        assert response["changed"] is True
    finally:
        db.close()
