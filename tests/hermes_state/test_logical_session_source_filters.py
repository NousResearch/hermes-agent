import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _seed_session(
    db: SessionDB,
    session_id: str,
    source: str,
    *,
    handoff_platform: str | None = None,
    handoff_state: str | None = None,
):
    db.create_session(session_id=session_id, source=source)
    db.append_message(session_id, role="user", content=f"{session_id} message")
    if handoff_platform is not None or handoff_state is not None:
        db._conn.execute(
            "UPDATE sessions SET handoff_platform = ?, handoff_state = ? WHERE id = ?",
            (handoff_platform, handoff_state, session_id),
        )
        db._conn.commit()


def test_logical_source_matches_completed_handoff_origin(db):
    _seed_session(db, "desktop-wechat", "desktop", handoff_platform="weixin", handoff_state="completed")
    _seed_session(db, "plain-desktop", "desktop")

    rows = db.list_sessions_rich(logical_source="weixin", order_by_last_active=True)

    assert [row["id"] for row in rows] == ["desktop-wechat"]
    assert db.session_count(logical_source="weixin", exclude_children=True) == 1


def test_exclude_logical_sources_keeps_local_and_messaging_slices_distinct(db):
    _seed_session(db, "desktop-wechat", "desktop", handoff_platform="weixin", handoff_state="completed")
    _seed_session(db, "local-chat", "desktop")
    _seed_session(db, "slack-chat", "slack")

    local_rows = db.list_sessions_rich(
        exclude_logical_sources=["cron", "weixin", "slack"],
        order_by_last_active=True,
    )
    messaging_rows = db.list_sessions_rich(
        exclude_logical_sources=["cron", "cli", "codex", "desktop", "gateway", "local", "tui"],
        order_by_last_active=True,
    )

    assert [row["id"] for row in local_rows] == ["local-chat"]
    assert {row["id"] for row in messaging_rows} == {"desktop-wechat", "slack-chat"}
