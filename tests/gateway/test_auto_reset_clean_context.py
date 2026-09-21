"""Regression tests for #35809 — compression-exhaustion auto-reset loop.

After compression is exhausted the gateway auto-resets the session so the
next message starts on a fresh, empty conversation (#9893 / #10063). That
guarantee regressed once the Telegram topic-binding heal landed
(#20470 / #29712 / #33414):

    1. Compression rotates ``session_entry.session_id`` to an oversized
       compressed *child* session mid-turn and the agent-result sync rewrites
       the ``(chat_id, thread_id) -> child`` topic binding.
    2. ``reset_session`` swaps in a clean, parentless session — but its return
       value was discarded and the topic binding was left pointing at the
       bloated child.
    3. On the next inbound message in that topic, the binding-heal walk
       ``switch_session``'d the freshly-reset lane *back* onto the bloated
       child, ``load_transcript`` reloaded the oversized transcript, and
       compression exhaustion re-fired — a new session id every loop.

The fix captures the fresh entry from ``reset_session`` and re-syncs the
topic binding to it (a no-op on non-topic lanes).

``TestAutoResetLoadsCleanContext`` — a behavioral contract on the real
``SessionStore``: after ``reset_session`` the next turn loads an EMPTY
transcript for the new session_id, never the bloated child's transcript.
"""

from __future__ import annotations

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# Behavioral contract: reset yields a clean next-turn transcript
# ---------------------------------------------------------------------------
def _make_store(tmp_path):
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    # Isolate the SQLite transcript store so we exercise per-session_id
    # transcripts without touching the developer's real state.db.
    store._db = SessionDB(db_path=tmp_path / "state.db")
    return store


def _make_source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1")


def _bloat(n):
    # Stand-in for the oversized, post-compression "child" transcript that
    # could not be compressed any further (#35809). Alternates roles so the
    # fixture is a valid conversation: load_transcript is a live-replay
    # restore site and repairs malformed assistant/tool structure on load
    # (#64934); alternate roles here to model a normal replayable conversation.
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": "x" * 2000,
        }
        for i in range(n)
    ]


class TestAutoResetLoadsCleanContext:
    """#35809: after the gateway auto-resets a session because compression
    was exhausted, the NEXT turn must load an EMPTY transcript for the new
    session_id — never the bloated compressed-child transcript."""

    def test_next_turn_transcript_is_empty_after_auto_reset(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()

        entry = store.get_or_create_session(source)
        session_key = entry.session_key
        bloated_sid = entry.session_id
        store._db.create_session(
            session_id=bloated_sid, source="telegram", user_id="u1"
        )
        store._db.replace_messages(bloated_sid, _bloat(120))
        assert len(store.load_transcript(bloated_sid)) == 120  # precondition

        new_entry = store.reset_session(session_key)
        assert new_entry is not None
        assert new_entry.session_id != bloated_sid

        resolved = store.get_or_create_session(source)
        assert resolved.session_id == new_entry.session_id
        loaded = store.load_transcript(resolved.session_id)

        assert loaded == [], (
            f"Auto-reset must yield an empty context, got {len(loaded)} "
            f"messages — the bloated compressed child leaked into the new session."
        )
        # The old transcript is still searchable, not destroyed.
        assert len(store.load_transcript(bloated_sid)) == 120

