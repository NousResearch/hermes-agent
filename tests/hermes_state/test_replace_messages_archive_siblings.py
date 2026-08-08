"""Sibling-site regression tests for the #80216 bug class.

#80216 fixed /retry destroying soft-archived (``active=0/compacted=1``)
in-place-compaction rows because ``rewrite_transcript`` defaulted to a
destructive full ``replace_messages``.  The same class existed at two more
sites, fixed here:

- ``acp_adapter/session.py`` ``_persist`` (non-owned-agent branch): probed
  ``has_archived_messages`` and FAILED OPEN into the destructive replace on
  any probe error (and could race a concurrent ``archive_and_compact``).
  Now passes ``active_only=True`` unconditionally.
- ``tui_gateway/methods_prompt.py`` edit/regenerate truncation: bare
  ``replace_messages`` deleted the archived transcript on every
  edit/regenerate of a compacted session.  Now ``active_only=True``.

Behavior contract on a fresh (never-compacted) session: every row is
``active=1``, so the active-only replace is identical to the full replace —
also pinned below.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def state_db(tmp_path):
    """A real SessionDB on a temp state.db."""
    return SessionDB(tmp_path / "state.db")


def _seed_compacted_session(db, session_id: str) -> None:
    """Create a session with archived (active=0/compacted=1) + live rows."""
    db.create_session(session_id, "test")
    msgs = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "another old question"},
        {"role": "assistant", "content": "another old answer"},
    ]
    db.append_messages_batch(session_id, msgs)
    db.archive_and_compact(
        session_id,
        [
            {"role": "assistant", "content": "summary of old turns"},
            {"role": "user", "content": "live question"},
            {"role": "assistant", "content": "live answer"},
        ],
    )


def _archived_count(db, session_id: str) -> int:
    return sum(
        1
        for m in db.get_messages(session_id, include_inactive=True)
        if not m["active"]
    )


class TestAcpPersistPreservesArchives:
    def test_persist_nonowned_branch_keeps_archived_rows(self, state_db):
        """The ACP _persist fallback replace must not delete archived rows."""
        sid = "acp-compacted"
        _seed_compacted_session(state_db, sid)
        assert _archived_count(state_db, sid) == 4

        # Drive the exact replace the non-owned-agent branch of _persist now
        # performs (active_only=True unconditionally, no probe).
        new_history = [
            {"role": "user", "content": "rewritten"},
            {"role": "assistant", "content": "rewritten answer"},
        ]
        state_db.replace_messages(sid, new_history, active_only=True)

        assert _archived_count(state_db, sid) == 4
        live = [
            m for m in state_db.get_messages_as_conversation(sid)
            if m.get("role") in ("user", "assistant")
        ]
        assert [m["content"] for m in live] == ["rewritten", "rewritten answer"]

    def test_persist_source_has_no_failopen_probe(self):
        """The fail-open has_archived_messages probe must stay dead in _persist.

        Guards the #80216 bug class at the source level: a probe that fails
        open (``except: has_archived = False``) silently reintroduces the
        destructive replace.  ``active_only=True`` must be unconditional.
        """
        import inspect
        import acp_adapter.session as acp_session

        src = inspect.getsource(acp_session.SessionManager._persist)
        # Assert on the CALL, not a local-variable name — a reintroduced probe
        # under any local name (archived = db.has_archived_messages(...))
        # must still trip this guard. The explanatory comment in _persist
        # writes the name as "(has_archived_messages)" (paren BEFORE the
        # name), so the call-shaped substring doesn't false-positive on it.
        assert "has_archived_messages(" not in src, (
            "_persist re-grew a has_archived_messages probe — #80216 class"
        )
        assert "active_only=True" in src

    def test_fresh_session_active_only_equals_full_replace(self, state_db):
        """On a never-compacted session active_only=True must behave exactly
        like the historical full replace (the safety claim the unconditional
        switch rests on)."""
        sid = "acp-fresh"
        state_db.create_session(sid, "test")
        state_db.append_messages_batch(
            sid,
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        )
        state_db.replace_messages(
            sid, [{"role": "user", "content": "only"}], active_only=True
        )
        rows = [
            m for m in state_db.get_messages_as_conversation(sid)
            if m.get("role") in ("user", "assistant")
        ]
        assert [m["content"] for m in rows] == ["only"]
        assert _archived_count(state_db, sid) == 0


class TestTuiPromptTruncationPreservesArchives:
    def test_truncation_source_uses_active_only(self):
        """The edit/regenerate persistence call must pass active_only=True."""
        import inspect
        import tui_gateway.methods_prompt as mp

        src = inspect.getsource(mp)
        # The truncation write is the only replace_messages call in the module;
        # it must carry active_only=True.
        import re
        calls = re.findall(r"db\.replace_messages\([^)]*\)", src, re.S)
        assert calls, "expected the truncation replace_messages call"
        for call in calls:
            assert "active_only=True" in call, (
                f"bare replace_messages in methods_prompt — #80216 class: {call}"
            )

    def test_truncation_write_keeps_archived_rows(self, state_db):
        """Drive the exact write shape methods_prompt now performs against a
        compacted session and assert the archive survives."""
        sid = "tui-compacted"
        _seed_compacted_session(state_db, sid)
        assert _archived_count(state_db, sid) == 4

        truncated = [{"role": "user", "content": "kept head"}]
        state_db.replace_messages(sid, truncated, active_only=True)

        assert _archived_count(state_db, sid) == 4
        live = [
            m for m in state_db.get_messages_as_conversation(sid)
            if m.get("role") in ("user", "assistant")
        ]
        assert [m["content"] for m in live] == ["kept head"]
