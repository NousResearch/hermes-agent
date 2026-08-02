"""Issue #76423 — Gateway routes source.profile into telegram topic state."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hermes_state import SessionDB
from gateway.config import Platform
from gateway.session import SessionSource


CHAT = "208214988"


def _source(profile=None, thread_id="42"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id=CHAT,
        chat_id=CHAT,
        user_name="tester",
        chat_type="dm",
        thread_id=thread_id,
        profile=profile,
    )


def test_gateway_uses_source_profile_not_global(tmp_path: Path):
    from gateway.run import GatewayRunner

    assert GatewayRunner._telegram_topic_profile_name(_source("coder")) == "coder"
    assert GatewayRunner._telegram_topic_profile_name(_source(None)) == "default"

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="sess-coder", source="telegram", user_id=CHAT, profile_name="coder")
    db.enable_telegram_topic_mode(chat_id=CHAT, user_id=CHAT, profile_name="coder")

    runner = object.__new__(GatewayRunner)
    runner._session_db = db
    assert runner._telegram_topic_mode_enabled(_source("coder")) is True
    assert runner._telegram_topic_mode_enabled(_source("other")) is False
    assert runner._telegram_topic_mode_enabled(_source(None)) is False

    runner._record_telegram_topic_binding(
        _source("coder", "42"),
        SimpleNamespace(session_key="k", session_id="sess-coder"),
    )
    assert db.get_telegram_topic_binding(
        chat_id=CHAT, thread_id="42", profile_name="coder",
    ) is not None
    assert db.get_telegram_topic_binding(
        chat_id=CHAT, thread_id="42", profile_name="default",
    ) is None
    db.close()
