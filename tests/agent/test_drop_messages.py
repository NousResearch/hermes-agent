"""Unit tests for agent.drop_messages.drop_messages.

Acceptance criteria from t_8a1c4c67:
  (a) Default call: last tool result and after removed; memory and state.db
      both updated.
  (b) N-specified removal.
  (c) Orphan tool-call correction.
  (d) No tool result in history → no-op with reason.
  (e) state.db failure → in-memory history unchanged.
"""

from __future__ import annotations

import pytest

from agent.drop_messages import (
    DropResult,
    DroppedMessageInfo,
    _correct_orphan_tool_calls,
    _find_last_tool_result_start,
    drop_messages,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_DB_PERSISTED_MARKER = "_db_persisted"


def _msg(role: str, content: str = "x", *, persisted: bool = False, **kw):
    """Build a minimal message dict for tests."""
    m = {"role": role, "content": content, **kw}
    if persisted:
        m[_DB_PERSISTED_MARKER] = True
    return m


def _tool_msg(content: str = "result", *, call_id: str = "c1", persisted: bool = False):
    return _msg("tool", content, tool_call_id=call_id, persisted=persisted)


def _assistant_tool_call(call_ids, content: str = "", *, persisted: bool = False):
    tool_calls = [{"id": cid, "function": {"name": "fn", "arguments": "{}"}} for cid in call_ids]
    return _msg("assistant", content, tool_calls=tool_calls, persisted=persisted)


class FakeDB:
    """Minimal SessionDB stand-in that records calls."""

    def __init__(self, *, raise_on_call: bool = False):
        self.calls: list = []
        self.raise_on_call = raise_on_call

    def deactivate_last_n_active_messages(self, session_id: str, n: int) -> int:
        self.calls.append((session_id, n))
        if self.raise_on_call:
            raise RuntimeError("simulated DB failure")
        return n


class FakeSession:
    """Minimal session object for testing."""

    def __init__(self, messages, *, session_db=None, session_id="test-session"):
        self._session_messages = list(messages)
        self._session_db = session_db
        self.session_id = session_id
        # Refusal counters that should be reset on successful drop.
        self._empty_content_retries = 3
        self._thinking_prefill_retries = 2
        self._dropped_toolcall_retries = 1


# ---------------------------------------------------------------------------
# _find_last_tool_result_start
# ---------------------------------------------------------------------------


def test_find_last_tool_result_none_when_empty():
    assert _find_last_tool_result_start([]) is None


def test_find_last_tool_result_none_when_no_tool():
    msgs = [_msg("user"), _msg("assistant")]
    assert _find_last_tool_result_start(msgs) is None


def test_find_last_tool_result_returns_last():
    msgs = [
        _msg("user"),
        _assistant_tool_call(["c1"]),
        _tool_msg(call_id="c1"),
        _msg("user", "follow up"),
        _assistant_tool_call(["c2"]),
        _tool_msg(call_id="c2"),
        _msg("user", "last user"),
    ]
    # Last tool result is at index 5.
    assert _find_last_tool_result_start(msgs) == 5


# ---------------------------------------------------------------------------
# _correct_orphan_tool_calls
# ---------------------------------------------------------------------------


def test_correct_no_tool_results_in_tail():
    msgs = [_msg("user"), _msg("assistant", "text")]
    # Dropping last message (index 1 = assistant text, no tool_calls).
    assert _correct_orphan_tool_calls(msgs, 1) == 1


def test_correct_single_call_pulls_in_assistant():
    msgs = [
        _msg("user"),
        _assistant_tool_call(["c1"]),  # index 1
        _tool_msg(call_id="c1"),        # index 2 — dropping from here
    ]
    assert _correct_orphan_tool_calls(msgs, 2) == 1


def test_correct_multi_call_all_results_dropped():
    msgs = [
        _msg("user"),
        _assistant_tool_call(["c1", "c2"]),  # index 1
        _tool_msg(call_id="c1"),              # index 2
        _tool_msg(call_id="c2"),              # index 3 — dropping from here
    ]
    # Both results are in the tail → assistant (index 1) is included.
    assert _correct_orphan_tool_calls(msgs, 3) == 1


def test_correct_multi_call_one_result_dropped():
    msgs = [
        _msg("user"),
        _assistant_tool_call(["c1", "c2"]),  # index 1
        _tool_msg(call_id="c1"),              # index 2
        _tool_msg(call_id="c2"),              # index 3 — drop from here
    ]
    # Dropping only tool(c2) at index 3: c2 in dropped_result_ids,
    # assistant(c1,c2) at index 1 owns c2 → drop from index 1.
    assert _correct_orphan_tool_calls(msgs, 3) == 1


def test_correct_no_orphan_when_tail_has_only_user():
    msgs = [
        _msg("user"),
        _assistant_tool_call(["c1"]),
        _tool_msg(call_id="c1"),
        _msg("user", "another"),  # index 3 — drop from here
    ]
    # No tool results in tail → no orphan.
    assert _correct_orphan_tool_calls(msgs, 3) == 3


# ---------------------------------------------------------------------------
# drop_messages — (a) default call (last tool result)
# ---------------------------------------------------------------------------


def test_drop_default_removes_from_last_tool_result():
    """(a) Default: removes last tool result to end, updates memory and DB."""
    db = FakeDB()
    msgs = [
        _msg("user", persisted=True),
        _assistant_tool_call(["c1"], persisted=True),
        _tool_msg(call_id="c1", persisted=True),   # drop_start = 2
        _msg("user", "follow", persisted=True),     # also dropped
    ]
    session = FakeSession(msgs, session_db=db)

    result = drop_messages(session)

    assert result.ok, result.error
    assert result.dropped == 3  # tool result + user (after orphan correction also pulls assistant)
    # In-memory: should have [user, assistant, tool_msg] → after orphan correction
    # the assistant is pulled in too → remaining = [user(index 0)]
    # Actually let's trace:
    # _find_last_tool_result_start → index 2 (tool)
    # _correct_orphan_tool_calls(msgs, 2):
    #   dropped_result_ids = {c1}
    #   walk back: index 1 = assistant(c1). c1 ∈ dropped → drop_start = 1
    # So dropped = msgs[1:] = [assistant, tool, user] → 3 messages
    assert len(session._session_messages) == 1
    assert session._session_messages[0]["role"] == "user"

    # DB was called with 3 (all three had persisted=True)
    assert db.calls == [("test-session", 3)]

    # Refusal counters reset
    assert session._empty_content_retries == 0
    assert session._thinking_prefill_retries == 0
    assert session._dropped_toolcall_retries == 0


def test_drop_default_no_orphan_when_another_turn_follows():
    """Assistant with tool results survives when its results precede drop_start."""
    db = FakeDB()
    msgs = [
        _msg("user"),
        _assistant_tool_call(["c1"], persisted=True),
        _tool_msg(call_id="c1", persisted=True),
        _msg("user", "turn 2"),
        _assistant_tool_call(["c2"], persisted=True),
        _tool_msg(call_id="c2", persisted=True),  # last tool result at index 5
    ]
    session = FakeSession(msgs, session_db=db)

    result = drop_messages(session)

    assert result.ok
    # Last tool result at 5. _correct_orphan: assistant(c2) at 4 is pulled → drop_start = 4.
    # Dropped: messages[4:] = [assistant(c2), tool(c2)] = 2 messages.
    assert result.dropped == 2
    assert len(session._session_messages) == 4


def test_drop_default_metadata_has_no_content():
    """DropResult.messages_info has role/length/sha256 but not content."""
    db = FakeDB()
    secret = "SECRET CONTENT DO NOT LOG"
    msgs = [
        _msg("user"),
        _assistant_tool_call(["c1"]),
        _tool_msg(secret, call_id="c1", persisted=True),
    ]
    session = FakeSession(msgs, session_db=db)

    result = drop_messages(session)

    assert result.ok
    for info in result.messages_info:
        assert isinstance(info, DroppedMessageInfo)
        assert not hasattr(info, "content")
        assert secret not in info.sha256
        assert info.length > 0
        assert len(info.sha256) == 64  # hex SHA-256


# ---------------------------------------------------------------------------
# drop_messages — (b) N-specified removal
# ---------------------------------------------------------------------------


def test_drop_n_removes_exactly_n():
    """(b) N=2 removes the last 2 messages."""
    db = FakeDB()
    msgs = [
        _msg("user"),
        _msg("assistant", "reply"),
        _msg("user", "follow", persisted=True),
        _msg("assistant", "reply2", persisted=True),
    ]
    session = FakeSession(msgs, session_db=db)

    result = drop_messages(session, n=2)

    assert result.ok
    assert result.dropped == 2
    assert len(session._session_messages) == 2
    assert db.calls == [("test-session", 2)]


def test_drop_n_larger_than_history_clamps_to_zero():
    """N > len(messages) drops everything."""
    db = FakeDB()
    msgs = [_msg("user"), _msg("assistant")]
    session = FakeSession(msgs, session_db=db)

    result = drop_messages(session, n=10)

    assert result.ok
    assert result.dropped == 2
    assert session._session_messages == []


def test_drop_n_with_orphan_correction():
    """(b+c) N=1 on a tool result triggers orphan correction to also drop assistant."""
    db = FakeDB()
    msgs = [
        _msg("user"),
        _assistant_tool_call(["c1"], persisted=True),
        _tool_msg(call_id="c1", persisted=True),  # only this is 'last 1'
    ]
    session = FakeSession(msgs, session_db=db)

    result = drop_messages(session, n=1)

    # Orphan correction: drop_start moved from 2 → 1 to pull in the assistant.
    assert result.ok
    assert result.dropped == 2  # assistant + tool result
    assert len(session._session_messages) == 1  # only the user remains


# ---------------------------------------------------------------------------
# drop_messages — (c) orphan tool-call correction (standalone)
# ---------------------------------------------------------------------------


def test_drop_orphan_multi_call_cleanup():
    """(c) Multi-call assistant: all calls dropped together."""
    db = FakeDB()
    msgs = [
        _msg("user"),
        _assistant_tool_call(["c1", "c2"], persisted=True),
        _tool_msg(call_id="c1", persisted=True),
        _tool_msg(call_id="c2", persisted=True),  # last tool result
    ]
    session = FakeSession(msgs, session_db=db)

    result = drop_messages(session)

    # Last tool at index 3. Orphan correction: assistant(c1,c2) at index 1,
    # c2 in dropped → drop_start = 1. Drops [assistant, tool(c1), tool(c2)] = 3.
    assert result.ok
    assert result.dropped == 3
    assert len(session._session_messages) == 1  # only user remains


# ---------------------------------------------------------------------------
# drop_messages — (d) no tool result → no-op
# ---------------------------------------------------------------------------


def test_drop_default_no_tool_result_is_noop():
    """(d) When there is no tool result, default call is a no-op with reason."""
    db = FakeDB()
    msgs = [_msg("user"), _msg("assistant", "text"), _msg("user", "again")]
    session = FakeSession(msgs, session_db=db)

    result = drop_messages(session)

    assert not result.ok
    assert result.dropped == 0
    assert "no tool result" in (result.error or "").lower()
    # History unchanged.
    assert len(session._session_messages) == 3
    # DB not touched.
    assert db.calls == []


def test_drop_empty_history_is_noop():
    db = FakeDB()
    session = FakeSession([], session_db=db)

    result = drop_messages(session)

    assert not result.ok
    assert result.dropped == 0


# ---------------------------------------------------------------------------
# drop_messages — (e) state.db failure leaves memory unchanged
# ---------------------------------------------------------------------------


def test_drop_db_failure_leaves_memory_unchanged():
    """(e) When state.db raises, the in-memory history is NOT modified."""
    db = FakeDB(raise_on_call=True)
    msgs = [
        _msg("user"),
        _assistant_tool_call(["c1"], persisted=True),
        _tool_msg(call_id="c1", persisted=True),
        _msg("user", "follow", persisted=True),
    ]
    original_len = len(msgs)
    session = FakeSession(msgs, session_db=db)

    result = drop_messages(session)

    assert not result.ok
    assert "state.db write failed" in (result.error or "")
    assert result.dropped == 0
    # In-memory list must be untouched.
    assert len(session._session_messages) == original_len


# ---------------------------------------------------------------------------
# drop_messages — no session_db (no DB, only memory)
# ---------------------------------------------------------------------------


def test_drop_no_session_db_still_mutates_memory():
    """When session_db is None, memory is still updated (best-effort)."""
    msgs = [
        _msg("user"),
        _assistant_tool_call(["c1"]),
        _tool_msg(call_id="c1"),
    ]
    session = FakeSession(msgs, session_db=None)

    result = drop_messages(session)

    assert result.ok
    assert result.dropped == 2  # assistant + tool (orphan correction)
    assert result.db_deactivated == 0
    assert len(session._session_messages) == 1


def test_drop_no_persisted_messages_skips_db():
    """When none of the dropped messages have _db_persisted, DB is not called."""
    db = FakeDB()
    # No _db_persisted on any message.
    msgs = [
        _msg("user"),
        _assistant_tool_call(["c1"]),    # not persisted
        _tool_msg(call_id="c1"),          # not persisted
    ]
    session = FakeSession(msgs, session_db=db)

    result = drop_messages(session)

    assert result.ok
    assert db.calls == []  # DB not touched


# ---------------------------------------------------------------------------
# DropResult attributes
# ---------------------------------------------------------------------------


def test_drop_result_ok_property():
    ok = DropResult(dropped=1, messages_info=[])
    assert ok.ok is True

    err = DropResult(dropped=0, error="oops")
    assert err.ok is False
