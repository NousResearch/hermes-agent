"""Owner-scoped atomic transcript reads for credential API requests."""

from __future__ import annotations

from contextlib import contextmanager

from hermes_state import SessionDB


def _owned_compression_tip(db: SessionDB, *, tip_owner: str = "owner-a") -> None:
    db.create_session("root", "api_server", credential_owner="owner-a")
    db.end_session("root", "compression")
    db.create_session(
        "tip", "api_server", parent_session_id="root", credential_owner=tip_owner
    )
    for index in range(6):
        db.append_message("tip", "user", f"message-{index}")


def test_owner_scoped_message_read_resolves_and_preserves_pagination_order(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _owned_compression_tip(db)

    oldest = db.resolve_owned_session_messages(
        "root", expected_credential_owner="owner-a", limit=2, offset=1
    )
    latest = db.resolve_owned_session_messages(
        "root", expected_credential_owner="owner-a", limit=2, offset=1, latest=True
    )

    assert oldest is not None
    assert oldest[0] == "tip"
    assert [message["content"] for message in oldest[1]] == ["message-1", "message-2"]
    assert latest is not None
    assert latest[0] == "tip"
    assert [message["content"] for message in latest[1]] == ["message-3", "message-4"]


def test_owner_scoped_message_read_rejects_requested_owner_mismatch(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _owned_compression_tip(db)

    assert db.resolve_owned_session_messages(
        "root", expected_credential_owner="owner-b", limit=500
    ) is None


def test_owner_scoped_message_read_keeps_compression_tip_when_descendants_are_empty(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", "api_server", credential_owner="owner-a")
    db.end_session("root", "compression")
    db.create_session(
        "tip", "api_server", parent_session_id="root", credential_owner="owner-a"
    )
    db.create_session(
        "empty-descendant", "api_server", parent_session_id="tip", credential_owner="owner-a"
    )

    result = db.resolve_owned_session_messages(
        "root", expected_credential_owner="owner-a", limit=500
    )

    assert result == ("tip", [])


def test_owner_scoped_message_read_snapshot_cannot_switch_to_recreated_foreign_row(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    racing_db = SessionDB(path)
    db.create_session("shared", "api_server", credential_owner="owner-a")
    db.append_message("shared", "user", "owned message")
    real_read_ctx = db._read_ctx
    replaced = False

    class ReplaceAfterOwnerCheck:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, parameters=()):
            nonlocal replaced
            if not replaced and "SELECT child.id" in sql:
                replaced = True
                assert racing_db.delete_session("shared")
                racing_db.create_session(
                    "shared", "api_server", credential_owner="owner-b"
                )
                racing_db.append_message("shared", "user", "FOREIGN_SECRET")
            return self._conn.execute(sql, parameters)

    @contextmanager
    def racing_read_ctx():
        with real_read_ctx() as conn:
            yield ReplaceAfterOwnerCheck(conn)

    monkeypatch.setattr(db, "_read_ctx", racing_read_ctx)

    result = db.resolve_owned_session_messages(
        "shared", expected_credential_owner="owner-a", limit=500
    )

    assert replaced is True
    assert result is not None
    assert [message["content"] for message in result[1]] == ["owned message"]
    assert "FOREIGN_SECRET" not in str(result)
    assert racing_db.get_session("shared")["credential_owner"] == "owner-b"


def test_owner_scoped_message_read_rejects_resolved_tip_owner_mismatch(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _owned_compression_tip(db, tip_owner="owner-b")

    assert db.resolve_owned_session_messages(
        "root", expected_credential_owner="owner-a", limit=500
    ) is None
