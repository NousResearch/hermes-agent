"""Regression test: ``_ensure_db_session()`` propagates ``agent._user_id``.

Before the fix in #35407, ``_ensure_db_session()`` hardcoded
``user_id=None`` when calling ``SessionDB.create_session``, so direct
AIAgent callers (CLI, tests, programmatic use) never had the user's
identity stored in the ``sessions`` table. The fix routes the agent's
``_user_id`` attribute into the row.

The gateway path (``gateway/session.py``) writes ``user_id`` separately
via ``record_gateway_session_peer`` and is unaffected by this fix.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

USER_ID = "test-user-123"
SESSION_ID = "test-ensure-db-session"


def _make_agent(session_db, session_id=SESSION_ID, user_id: str | None = None):
    """Build a minimal AIAgent pointing at the supplied SessionDB.

    Mirrors the ``_make_agent`` helper used in ``test_identity_flush.py``
    (avoids the heavyweight ``init_agent`` pathway — we only need the
    attributes ``_ensure_db_session`` reads).
    """
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    if user_id is not None:
        agent._user_id = user_id
    return agent


class TestEnsureDbSessionUserId:
    def test_user_id_is_persisted_to_session_row(self, tmp_path):
        """``agent._user_id`` must reach the ``sessions`` table column."""
        from hermes_state import SessionDB

        db_path = Path(tmp_path) / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            agent = _make_agent(db, user_id=USER_ID)

            # The first call creates the row; the second is a no-op thanks
            # to ``_session_db_created``.
            agent._ensure_db_session()

            row = db.get_session(SESSION_ID)
            assert row is not None, "expected a session row to be created"
            assert row["user_id"] == USER_ID, (
                f"user_id was not propagated; row={row!r}"
            )
        finally:
            db.close()

    def test_user_id_defaults_to_none_when_unset(self, tmp_path):
        """Without ``_user_id`` (CLI default), the row keeps ``user_id=NULL``
        so the pre-fix behavior on that path is preserved."""
        from hermes_state import SessionDB

        db_path = Path(tmp_path) / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            agent = _make_agent(db)  # no _user_id

            agent._ensure_db_session()

            row = db.get_session(SESSION_ID)
            assert row is not None
            assert row["user_id"] is None
        finally:
            db.close()
