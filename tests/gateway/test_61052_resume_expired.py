import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
from gateway.session import SessionStore, SessionEntry, SessionSource
from gateway.config import GatewayConfig, SessionResetPolicy, Platform

def test_resume_pending_expired_respects_mode_none(tmp_path):
    # 1. Setup config with mode: none
    policy = SessionResetPolicy(mode="none", idle_minutes=1440)
    config = GatewayConfig(default_reset_policy=policy)
    store = SessionStore(sessions_dir=tmp_path, config=config)
    store._save = MagicMock()

    # 2. Setup source and get its canonical session key
    source = SessionSource(platform=Platform.LOCAL, chat_id="test_chat", chat_type="dm")
    session_key = store._generate_session_key(source)

    # Create a session entry that is resume_pending and expired
    now = datetime.now()
    entry = SessionEntry(
        session_key=session_key,
        session_id="test_session",
        created_at=now,
        updated_at=now - timedelta(seconds=5000)
    )
    entry.last_resume_marked_at = now - timedelta(seconds=5000)
    entry.resume_pending = True

    # Inject the entry into the store
    store._entries[session_key] = entry

    # 3. Attempt to get session
    recovered = store.get_or_create_session(source)
    assert recovered.session_id == "test_session"
    assert recovered.resume_pending is True

def test_resume_pending_expired_triggers_when_enabled(tmp_path):
    # 1. Setup config with mode: idle (resets enabled)
    policy = SessionResetPolicy(mode="idle", idle_minutes=1440)
    config = GatewayConfig(default_reset_policy=policy)
    store = SessionStore(sessions_dir=tmp_path, config=config)
    store._save = MagicMock()

    # 2. Setup source and get its canonical session key
    source = SessionSource(platform=Platform.LOCAL, chat_id="test_chat", chat_type="dm")
    session_key = store._generate_session_key(source)

    # Create a session entry that is resume_pending and expired
    now = datetime.now()
    entry = SessionEntry(
        session_key=session_key,
        session_id="test_session",
        created_at=now,
        updated_at=now - timedelta(seconds=5000)
    )
    entry.last_resume_marked_at = now - timedelta(seconds=5000)
    entry.resume_pending = True

    # Inject the entry into the store
    store._entries[session_key] = entry

    # 3. Attempt to get session
    recovered = store.get_or_create_session(source)
    assert recovered.session_id != "test_session"
    assert recovered.resume_pending is False


def test_resume_pending_expired_sets_correct_reason(tmp_path):
    # Setup config with mode: idle
    policy = SessionResetPolicy(mode="idle", idle_minutes=1440)
    config = GatewayConfig(default_reset_policy=policy)
    store = SessionStore(sessions_dir=tmp_path, config=config)
    store._save = MagicMock()

    source = SessionSource(platform=Platform.LOCAL, chat_id="test_chat", chat_type="dm")
    session_key = store._generate_session_key(source)

    now = datetime.now()
    entry = SessionEntry(
        session_key=session_key,
        session_id="test_session",
        created_at=now,
        updated_at=now - timedelta(seconds=5000)
    )
    entry.last_resume_marked_at = now - timedelta(seconds=5000)
    entry.resume_pending = True

    store._entries[session_key] = entry

    recovered = store.get_or_create_session(source)
    assert recovered.auto_reset_reason == "resume_pending_expired"