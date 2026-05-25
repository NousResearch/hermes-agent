"""Tests for prompt-only gateway session bridge formatting and rules."""

from datetime import datetime, timedelta

from agent.context_compressor import SUMMARY_PREFIX
from agent.session_bridge import build_session_bridge_context
from gateway.config import BridgeConfig, Platform
from gateway.session import SessionBridgeCandidate, SessionSource
from hermes_state import SessionDB


def _source(platform=Platform.TELEGRAM):
    return SessionSource(
        platform=platform,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
        user_name="tester",
    )


def _db_with_messages(tmp_path, session_id="prev", messages=None):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(
        session_id=session_id,
        source="telegram",
        user_id="user-1",
        session_key="agent:main:telegram:dm:chat-1",
        origin_json=_source().to_dict(),
    )
    for role, content in messages or []:
        db.append_message(session_id=session_id, role=role, content=content)
    return db


def _candidate(ended_at):
    return SessionBridgeCandidate(
        session_id="prev",
        ended_at=ended_at,
        reason="session_auto_reset",
    )


def test_bridge_applied_for_fresh_rotation(tmp_path):
    ended_at = datetime(2026, 5, 24, 20, 0, 0)
    db = _db_with_messages(
        tmp_path,
        messages=[
            ("user", "Peux-tu analyser le build VTT ?"),
            ("assistant", "Je regarde les erreurs tsc."),
            ("user", "Et pour les tests ?"),
        ],
    )

    result = build_session_bridge_context(
        session_db=db,
        config=BridgeConfig(max_messages=2, freshness_threshold_minutes=10),
        source=_source(),
        current_session_id="current",
        previous=_candidate(ended_at),
        now=ended_at + timedelta(minutes=12),
    )

    assert result.applied is True
    assert "[CONTEXTE DE REPRISE]" in result.text
    assert "il y a 12 minutes" in result.text
    assert "Et pour les tests ?" in result.text
    assert "Peux-tu analyser" not in result.text.split("DERNIERS ECHANGES :")[-1]


def test_bridge_skips_too_old_session(tmp_path):
    ended_at = datetime(2026, 5, 20, 20, 0, 0)
    db = _db_with_messages(
        tmp_path,
        messages=[("user", "hello"), ("assistant", "hi")],
    )

    result = build_session_bridge_context(
        session_db=db,
        config=BridgeConfig(max_age_minutes=60),
        source=_source(),
        current_session_id="current",
        previous=_candidate(ended_at),
        now=ended_at + timedelta(hours=2),
    )

    assert result.applied is False
    assert result.reason == "too_old"


def test_bridge_skips_when_less_than_two_useful_messages(tmp_path):
    ended_at = datetime(2026, 5, 24, 20, 0, 0)
    db = _db_with_messages(tmp_path, messages=[("user", "hello")])

    result = build_session_bridge_context(
        session_db=db,
        config=BridgeConfig(),
        source=_source(),
        current_session_id="current",
        previous=_candidate(ended_at),
        now=ended_at + timedelta(minutes=1),
    )

    assert result.applied is False
    assert result.reason == "too_few_messages"


def test_bridge_skips_local_cli(tmp_path):
    ended_at = datetime(2026, 5, 24, 20, 0, 0)
    db = _db_with_messages(
        tmp_path,
        messages=[("user", "hello"), ("assistant", "hi")],
    )

    result = build_session_bridge_context(
        session_db=db,
        config=BridgeConfig(),
        source=_source(Platform.LOCAL),
        current_session_id="current",
        previous=_candidate(ended_at),
        now=ended_at + timedelta(minutes=1),
    )

    assert result.applied is False
    assert result.reason == "local"


def test_bridge_uses_latest_compression_summary_outside_recent_window(tmp_path):
    ended_at = datetime(2026, 5, 24, 20, 0, 0)
    db = _db_with_messages(
        tmp_path,
        messages=[
            ("assistant", f"{SUMMARY_PREFIX}\nDiagnostic du build en cours."),
            ("user", "message 1"),
            ("assistant", "reply 1"),
            ("user", "message 2"),
            ("assistant", "reply 2"),
        ],
    )

    result = build_session_bridge_context(
        session_db=db,
        config=BridgeConfig(max_messages=2, include_summary=True),
        source=_source(),
        current_session_id="current",
        previous=_candidate(ended_at),
        now=ended_at + timedelta(minutes=1),
    )

    assert result.applied is True
    assert "Diagnostic du build en cours" in result.text


def test_bridge_falls_back_to_bookend_summary(tmp_path):
    ended_at = datetime(2026, 5, 24, 20, 0, 0)
    db = _db_with_messages(
        tmp_path,
        messages=[
            ("user", "first ask"),
            ("assistant", "first answer"),
            ("user", "middle"),
            ("assistant", "middle answer"),
            ("user", "last ask"),
            ("assistant", "last answer"),
        ],
    )

    result = build_session_bridge_context(
        session_db=db,
        config=BridgeConfig(include_summary=False),
        source=_source(),
        current_session_id="current",
        previous=_candidate(ended_at),
        now=ended_at + timedelta(minutes=1),
    )

    assert result.applied is True
    assert "first ask" in result.text
    assert "last answer" in result.text
