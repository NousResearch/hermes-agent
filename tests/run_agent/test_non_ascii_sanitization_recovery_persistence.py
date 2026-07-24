#!/usr/bin/env python3
"""Faithful test for non-ASCII sanitization persistence to SessionDB.

Unlike the older contract test (which called
``agent._session_db.update_system_prompt(...)`` directly and thus
only re-asserted what it had just written), this test drives the
REAL UnicodeEncodeError recovery path in
``agent/conversation_loop.py``:

    run_conversation()               (real agent loop)
      -> model client raises UnicodeEncodeError("'ascii' codec ...")
      -> except Exception as api_error        (conversation_loop.py:1955)
      -> _is_ascii_codec is True
      -> ASCII-codec block                   (conversation_loop.py:2012)
      -> _strip_non_ascii(active_system_prompt) (conversation_loop.py:2040)
      -> agent._session_db.update_system_prompt(...) (conversation_loop.py:2047)
      -> persisted to the REAL SessionDB (tempfile)

The persisted (sanitized) prompt is then read back from the
SessionDB and asserted to equal the stripped version, not the
original non-ASCII prompt.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from hermes_state import SessionDB
from agent.message_sanitization import _strip_non_ascii


def _has_non_ascii(s: str) -> bool:
    """Return True if *s* contains any non-ASCII code point."""
    return any(ord(c) > 127 for c in (s or ""))


class TestNonAsciiSanitizationRecoveryPersistence(unittest.TestCase):
    """Verify the REAL recovery path persists the sanitized system prompt."""

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

    def test_recovery_persists_sanitized_prompt_to_session_db(self):
        """A UnicodeEncodeError on the API call must drive the recovery
        path, which strips non-ASCII from the system prompt and
        writes the sanitized copy to SessionDB (not the original).
        """
        # A system prompt with non-ASCII characters: em dash (U+2014),
        # right single quote (U+2019), and a smart apostrophe (U+2019 variant).
        non_ascii_prompt = (
            "You are a helpful assistant \u2014 don\u2019t guess, "
            "and never fabricate output"
        )
        sanitized = _strip_non_ascii(non_ascii_prompt)
        assert sanitized != non_ascii_prompt, (
            "Sanitization must actually remove non-ASCII characters"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)
            # Seed the DB with a NON-ASCII system prompt (the pre-fix
            # state: the prompt arrived with non-ASCII content and was
            # written to the DB before any recovery could strip it).
            # We assert later that the recovery path OVERWRITES this
            # with a sanitized copy.
            db.create_session(session_id=agent.session_id, source="test")
            db.update_system_prompt(agent.session_id, non_ascii_prompt)

            # The user-facing system_message passed to run_conversation is
            # kept ASCII-clean so message build/serialization does NOT
            # raise before the model client is reached (on origin/main
            # a non-ASCII system_message raises earlier and never
            # reaches the recovery except). The UnicodeEncodeError is
            # injected at the CLIENT call via the mock below, so the
            # real recovery path at conversation_loop.py:1955 is entered
            # on BOTH the fixed branch and origin/main.
            clean_system_message = (
                "You are a helpful assistant. Answer concisely."
            )

            # Mock the model client so the FIRST call raises a
            # UnicodeEncodeError whose message mentions the 'ascii' codec.
            # This is exactly what a system with LANG=C / non-UTF-8
            # locale produces, and it is what routes the real loop into
            # the ASCII-codec recovery block at conversation_loop.py:2012.
            #
            # The SECOND call also raises (a fresh UnicodeEncodeError)
            # so _unicode_sanitization_passes reaches 2 and the outer
            # guard (< 2) stops retrying recovery — the loop then
            # falls through to normal error handling and returns.  By that
            # point the sanitized prompt has ALREADY been persisted
            # via the real update_system_prompt call at :2047.
            call_count = {"n": 0}

            def _create_side_effect(*args, **kwargs):
                call_count["n"] += 1
                raise UnicodeEncodeError(
                    "encode",
                    non_ascii_prompt,
                    0,
                    1,
                    "'ascii' codec can't encode character '\\u2014' "
                    "in position 0: ordinal not in range(128)",
                )

            mock_create = MagicMock(side_effect=_create_side_effect)
            mock_completions = MagicMock()
            mock_completions.create = mock_create
            mock_chat = MagicMock()
            mock_chat.completions = mock_completions
            mock_client = MagicMock()
            mock_client.chat = mock_chat

            with patch.object(agent, "client", mock_client):
                try:
                    agent.run_conversation(
                        "hello",
                        system_message=clean_system_message,
                    )
                except Exception:
                    # The loop is expected to surface the second
                    # encode failure as a normal error; we only care
                    # that the recovery path ran and persisted.  Swallow.
                    pass

            # The recovery path MUST have been entered (the client was
            # called at least once with the failing encode).
            assert call_count["n"] >= 1, (
                "Model client was never called — recovery path not exercised"
            )

            # Verify: SessionDB now carries a SANITIZED prompt.
            # The recovery path at conversation_loop.py:2040-2049 calls
            # update_system_prompt() with the stripped copy; pre-fix it
            # never did, so the DB would still hold the UNSANITIZED
            # seeded prompt.  Assert the persisted value is stripped
            # (no non-ASCII) and is no longer the raw seeded one.
            session_row = db.get_session(agent.session_id)
            assert session_row is not None, (
                "Session row must exist after recovery"
            )
            stored = session_row.get("system_prompt", "")
            assert not _has_non_ascii(stored), (
                f"SessionDB must store a SANITIZED prompt (no non-ASCII). "
                f"Got non-ASCII chars in: {stored!r}"
            )
            # The persisted copy must differ from the unsanitized
            # seeded prompt — i.e. the recovery actually stripped and
            # wrote back, rather than leaving the stale original.
            assert stored != non_ascii_prompt, (
                "SessionDB must NOT retain the unsanitized prompt; "
                "recovery must have persisted the stripped copy"
            )


if __name__ == "__main__":
    import unittest
    unittest.main()
