"""Session persistence identity tests for gateway user_id and credential_owner."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch


def _make_agent(session_db, *, session_id: str, platform: str, user_id: str | None):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            platform=platform,
            user_id=user_id,
            skip_context_files=True,
            skip_memory=True,
        )


def test_api_server_lazy_session_create_persists_credential_owner_without_reinterpreting_user_id():
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "state.db")
        try:
            agent = _make_agent(
                db,
                session_id="api-session-1",
                platform="api_server",
                user_id="gateway-user-123",
            )
            agent._credential_owner = "api-credential:owner-xyz"

            agent._ensure_db_session()

            row = db.get_session("api-session-1")
            assert row is not None
            assert row["credential_owner"] == "api-credential:owner-xyz"
            assert row["user_id"] == "gateway-user-123"
            assert row["user_id"] != row["credential_owner"]
        finally:
            db.close()


def test_telegram_lazy_session_create_keeps_user_id_persistence_unchanged():
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "state.db")
        try:
            agent = _make_agent(
                db,
                session_id="telegram-session-1",
                platform="telegram",
                user_id="telegram-user-456",
            )

            agent._ensure_db_session()

            row = db.get_session("telegram-session-1")
            assert row is not None
            assert row["user_id"] == "telegram-user-456"
            assert row["credential_owner"] is None
        finally:
            db.close()
