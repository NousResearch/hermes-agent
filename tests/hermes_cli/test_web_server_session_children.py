import asyncio

from hermes_cli import web_server


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
            "focused": [{"id": "focused", "child_kind": "focused_continuation"}],
            "branches": [],
            "interactive": [],
            "compression": [],
            "subagents": {
                "active": [],
                "completed": [{"id": "audit", "child_kind": "delegate_subagent_completed"}],
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
