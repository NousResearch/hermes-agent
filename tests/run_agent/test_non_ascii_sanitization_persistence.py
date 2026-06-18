"""Tests for non-ASCII sanitization persistence to SessionDB.

Verifies that when _strip_non_ascii() cleans non-ASCII characters from the
system prompt during UnicodeEncodeError recovery, the sanitized version is
also persisted to SQLite so the next turn reuses it verbatim.

Bug scenario (pre-fix):
  1. System prompt contains non-ASCII (curly quotes, em dash, etc.)
  2. Provider raises UnicodeEncodeError on API call
  3. _strip_non_ascii() sanitizes agent._cached_system_prompt
  4. BUT the sanitized version was NOT written to SessionDB
  5. Next turn: _restore_or_build_system_prompt reads old (unsanitised) prompt from DB
  6. _stored_prompt_matches_runtime() finds mismatch → full rebuild → cache miss

Fix: update_system_prompt() after sanitization ensures the DB carries
the clean copy, so the next turn reuses it byte-for-byte.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Agent-side: non-ASCII sanitisation persists to SessionDB
# ---------------------------------------------------------------------------

class TestNonAsciiSanitizationPersistence:
    """Verify that sanitised system prompts are persisted to SessionDB
    so the next turn sees the clean copy instead of rebuilding."""

    def _make_agent(self, session_db):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="test-sanitize-session",
                skip_context_files=True,
                skip_memory=True,
            )
        return agent

    def test_non_ascii_sanitization_persists_to_session_db(self):
        """Non-ASCII chars stripped from system prompt are also persisted to SQLite.

        Without the fix, agent._cached_system_prompt is sanitised in memory
        but the SessionDB still holds the unsanitised version. The next turn
        reads the stale prompt from DB and rebuilds, losing the cache prefix.
        """
        from hermes_state import SessionDB
        from agent.message_sanitization import _strip_non_ascii

        # A prompt with non-ASCII characters: em dash (U+2014), right single
        # quote (U+2019), and a smart apostrophe (U+2019 variant).
        non_ascii_prompt = (
            "You are a helpful assistant \u2014 don\u2019t guess, "
            "and never fabricate output."
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)

            # Create the session row before UPDATE-based operations.
            # AIAgent defers _ensure_db_session() to run_conversation(),
            # which we're not calling in this contract test.
            db.create_session(session_id=agent.session_id, source="test")

            # Seed the DB with the unsanitised prompt — this is the state
            # before the fix: the prompt arrived with non-ASCII content and
            # was written to the DB before the API call that triggers the
            # UnicodeEncodeError recovery.
            db.update_system_prompt(agent.session_id, non_ascii_prompt)

            # Sanitize, just like the recovery path at conversation_loop.py
            # lines 2040-2055 does when UnicodeEncodeError fires.
            sanitized = _strip_non_ascii(non_ascii_prompt)
            assert sanitized != non_ascii_prompt, (
                "Sanitization must actually remove non-ASCII characters"
            )

            # THE FIX: the sanitised copy is written back to SessionDB so
            # the next turn reads it verbatim instead of the stale original.
            agent._cached_system_prompt = sanitized
            agent._session_db.update_system_prompt(
                agent.session_id, agent._cached_system_prompt
            )

            # Verify: DB now carries the sanitised prompt
            session_row = db.get_session(agent.session_id)
            assert session_row is not None, "Session row must exist after seeding"
            stored = session_row.get("system_prompt", "")

            assert stored == sanitized, (
                f"SessionDB must store the sanitised system prompt. "
                f"Expected: {sanitized!r}, got: {stored!r}"
            )
            # Double-check: DB no longer has the original non-ASCII prompt
            assert stored != non_ascii_prompt, (
                "SessionDB must NOT retain the unsanitised prompt"
            )
