from contextlib import nullcontext

from hermes_state import SessionDB


def _session(db: SessionDB, session_id: str, base_url: str) -> None:
    db.create_session(
        session_id,
        source="tui",
        model="local-model",
        model_config={"provider": "custom", "base_url": base_url},
    )
    db.append_message(session_id, role="user", content="keep this transcript")


def test_unreachable_loopback_session_is_archived_without_data_loss(
    tmp_path, monkeypatch
):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _session(db, "stale", "http://127.0.0.1:65530/v1")
        monkeypatch.setattr(
            "hermes_state.socket.create_connection",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ConnectionRefusedError("offline")
            ),
        )

        assert db.archive_if_unreachable_local_endpoint("stale") is True

        row = db.get_session("stale")
        assert row["archived"] == 1
        assert row["ended_at"] is not None
        assert row["end_reason"] == "archived_local_endpoint_stale"
        assert db.get_messages("stale")[0]["content"] == "keep this transcript"
        assert db.search_sessions(source="tui") == []
        assert [r["id"] for r in db.search_sessions(
            source="tui", include_archived=True
        )] == ["stale"]
        exported = db.export_all(source="tui")
        assert [row["id"] for row in exported] == ["stale"]
        assert exported[0]["messages"][0]["content"] == "keep this transcript"
    finally:
        db.close()


def test_reachable_loopback_session_remains_resumable(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _session(db, "reachable", "http://localhost:8080/v1")
        monkeypatch.setattr(
            "hermes_state.socket.create_connection",
            lambda address, timeout: nullcontext(object()),
        )

        assert db.archive_if_unreachable_local_endpoint("reachable") is False
        row = db.get_session("reachable")
        assert row["archived"] == 0
        assert row["ended_at"] is None
    finally:
        db.close()


def test_non_loopback_endpoint_is_never_probed_or_archived(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _session(db, "remote", "https://api.example.com/v1")

        def unexpected_probe(*_args, **_kwargs):
            raise AssertionError("remote endpoints must not be probed")

        monkeypatch.setattr(
            "hermes_state.socket.create_connection", unexpected_probe
        )

        assert db.archive_if_unreachable_local_endpoint("remote") is False
        assert db.get_session("remote")["archived"] == 0
    finally:
        db.close()
