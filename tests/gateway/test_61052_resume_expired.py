import pytest
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
from gateway.session import SessionStore, SessionEntry, SessionSource
from gateway.config import GatewayConfig, SessionResetPolicy, Platform

@pytest.fixture
def temp_sessions():
    path = Path("/tmp/hermes_test_sessions")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    yield path
    if path.exists():
        shutil.rmtree(path)

def test_resume_pending_expired_respects_mode_none(temp_sessions):
    # 1. Setup config with mode: none
    policy = SessionResetPolicy(mode="none", idle_minutes=1440)
    config = GatewayConfig(
        default_reset_policy=policy
    )
    store = SessionStore(sessions_dir=temp_sessions, config=config)
    store._save = MagicMock() 
    
    # 2. Setup source and get its canonical session key
    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="test_chat",
        chat_type="dm"
    )
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

def test_resume_pending_expired_triggers_when_enabled(temp_sessions):
    # 1. Setup config with mode: idle (resets enabled)
    policy = SessionResetPolicy(mode="idle", idle_minutes=1440)
    config = GatewayConfig(
        default_reset_policy=policy
    )
    store = SessionStore(sessions_dir=temp_sessions, config=config)
    store._save = MagicMock()
    
    # 2. Setup source and get its canonical session key
    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="test_chat",
        chat_type="dm"
    )
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


