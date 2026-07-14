"""AC1–AC5: the append-time generation gate (_persist_superseded).

When the gateway invalidates a turn's run generation (/stop, /new, stale-agent
eviction), the turn is a "zombie" whose continued CONTENT writes are unwanted —
they land AFTER the user stopped the turn (the 2026-07-14 undo-clobber incident).
_flush_messages_to_session_db suppresses those rows when agent._persist_superseded
is True.

🔴 LOAD-BEARING CARVE-OUT (I1): the interrupt-close tail
(finish_reason == interrupt_close, written by close_interrupted_tool_sequence)
MUST still persist — it is the role-alternation repair + the deliberate
restart-loop backstop / auto-continue signal (hermes #45230/#49201/#49243).
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from agent.message_sanitization import _INTERRUPT_CLOSE_FINISH_REASON

SESSION_ID = "test-superseded-gate"


def _make_agent(session_db, session_id=SESSION_ID):
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
    agent._ensure_db_session()
    return agent


def _rows(db, session_id=SESSION_ID):
    return db.get_messages(session_id)


def test_superseded_gate_suppresses_continued_content_rows():
    """AC1: with _persist_superseded=True, a NEW content row is NOT persisted."""
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "t.db")
        try:
            agent = _make_agent(db)
            agent._persist_superseded = True

            messages = [
                {"role": "user", "content": "the original question"},
                {"role": "assistant", "content": "a zombie continued-content row"},
            ]
            agent._flush_messages_to_session_db(messages)

            contents = [r["content"] for r in _rows(db)]
            assert "a zombie continued-content row" not in contents, (
                "a superseded turn's continued content row must be suppressed"
            )
        finally:
            db.close()


def test_superseded_gate_preserves_interrupt_close_tail():
    """AC2 (I1): the interrupt-close tail MUST persist even when superseded."""
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "t.db")
        try:
            agent = _make_agent(db)
            agent._persist_superseded = True

            messages = [
                {"role": "assistant", "content": "a suppressed zombie row"},
                {
                    "role": "assistant",
                    "content": "Operation interrupted.",
                    "finish_reason": _INTERRUPT_CLOSE_FINISH_REASON,
                },
            ]
            agent._flush_messages_to_session_db(messages)

            rows = _rows(db)
            finish_reasons = [r["finish_reason"] for r in rows]
            contents = [r["content"] for r in rows]
            assert _INTERRUPT_CLOSE_FINISH_REASON in finish_reasons, (
                "the interrupt-close tail must ALWAYS persist (restart-loop backstop)"
            )
            # ...and the zombie content row was still suppressed.
            assert "a suppressed zombie row" not in contents
        finally:
            db.close()


def test_normal_turn_persists_byte_identically():
    """AC3 (I3): a NON-superseded turn persists exactly as before (flag unset)."""
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "t.db")
        try:
            agent = _make_agent(db)
            # flag never set → default False

            messages = [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a normal answer that must persist"},
            ]
            agent._flush_messages_to_session_db(messages)

            contents = [r["content"] for r in _rows(db)]
            assert "a normal answer that must persist" in contents
        finally:
            db.close()


def test_gate_fails_open_on_error():
    """AC5 (I5): a raising _persist_superseded access must not lose a real row."""
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "t.db")
        try:
            agent = _make_agent(db)

            class _Boom:
                def __bool__(self):
                    raise RuntimeError("flag access boom")

            # getattr returns the object; the `if` truthiness check raises →
            # the gate is wrapped so persistence proceeds (fail-open).
            agent._persist_superseded = _Boom()

            messages = [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "must-not-be-lost answer"},
            ]
            # Must not raise out of the flush, and must persist the real row.
            agent._flush_messages_to_session_db(messages)
            contents = [r["content"] for r in _rows(db)]
            assert "must-not-be-lost answer" in contents, (
                "a guard error must fail OPEN (never lose a real row)"
            )
        finally:
            db.close()


def test_tool_result_persists_when_owner_already_durable():
    """🔴 B1 (the corruption bug): an assistant(tool_calls) persisted BEFORE
    /stop, then its `tool` result arrives in the suppression window — the
    result MUST persist (suppressing it would orphan the durable call =
    #48879 role-alternation corruption).

    This is the adversarial split-pair case the happy-path matrix missed.
    """
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "t.db")
        try:
            agent = _make_agent(db)

            # --- flush 1: the assistant(tool_calls) lands BEFORE /stop (flag off) ---
            asst = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "terminal", "arguments": "{}"}}],
            }
            messages = [asst]
            agent._flush_messages_to_session_db(messages)
            # the assistant(tool_calls) row is now durable
            assert any(r["tool_calls"] for r in _rows(db)), "owner must be persisted pre-stop"

            # --- /stop lands: flag on. The matching tool result arrives now. ---
            agent._persist_superseded = True
            tool_row = {"role": "tool", "tool_name": "terminal",
                        "content": "result payload", "tool_call_id": "call_1"}
            messages.append(tool_row)
            agent._flush_messages_to_session_db(messages)

            rows = _rows(db)
            # The tool result MUST have persisted — its owner is already durable,
            # so suppressing it would leave a dangling tool call.
            tool_rows = [r for r in rows if r["role"] == "tool" and r["tool_call_id"] == "call_1"]
            assert tool_rows, (
                "a tool result whose assistant(tool_calls) is already persisted "
                "MUST NOT be suppressed (would orphan the call → #48879 corruption)"
            )
        finally:
            db.close()


def test_clean_pair_both_suppressed_when_both_arrive_after_stop():
    """The safe case: when BOTH the assistant(tool_calls) and its tool result
    arrive AFTER /stop (in the same suppression flush), the whole pair is
    dropped atomically — no dangling call, no orphan result."""
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "t.db")
        try:
            agent = _make_agent(db)
            # seed a durable prior user row so the session isn't empty
            agent._flush_messages_to_session_db([{"role": "user", "content": "q"}])

            agent._persist_superseded = True
            messages = [
                {"role": "user", "content": "q"},  # already durable (dedup by identity? new dict → but content match)
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "call_z", "type": "function",
                                 "function": {"name": "terminal", "arguments": "{}"}}]},
                {"role": "tool", "tool_name": "terminal", "content": "zombie result",
                 "tool_call_id": "call_z"},
            ]
            agent._flush_messages_to_session_db(messages)

            rows = _rows(db)
            # Neither half of the post-stop pair persisted.
            assert not any(r["tool_call_id"] == "call_z" for r in rows), (
                "the tool result of a suppressed assistant(tool_calls) must also be suppressed"
            )
            assert not any(
                r["tool_calls"] and "call_z" in str(r["tool_calls"]) for r in rows
            ), "the assistant(tool_calls) added after /stop must be suppressed"
        finally:
            db.close()


def test_cross_flush_split_pair_no_orphan():
    """🔴 B1′ (the cross-flush corruption): the assistant(tool_calls) is
    suppressed in Flush A, and its `tool` result arrives NEW in a SEPARATE
    Flush B (exactly how the real loop persists — append assistant → flush →
    _execute_tool_calls appends result → flush). The result MUST also be
    suppressed (its owner never persisted), else it orphans into a dangling
    tool result = #48879. Requires the suppressed-id set to be AGENT-scoped,
    not per-flush.
    """
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "t.db")
        try:
            agent = _make_agent(db)
            # a durable prior user turn so the transcript isn't empty
            agent._flush_messages_to_session_db([{"role": "user", "content": "q"}])

            agent._persist_superseded = True

            # --- Flush A: the assistant(tool_calls) row (NEW, post-stop) ---
            asst = {"role": "assistant", "content": None,
                    "tool_calls": [{"id": "call_x", "type": "function",
                                    "function": {"name": "terminal", "arguments": "{}"}}]}
            messages = [{"role": "user", "content": "q"}, asst]
            agent._flush_messages_to_session_db(messages)

            # --- Flush B: the matching tool result, a SEPARATE flush.
            # Faithfully simulate the owner NOT being re-presented in Flush B's
            # list — this is what breaks a per-flush set (the owner's id isn't
            # re-added, so a per-flush set is empty and the result orphans). The
            # real loop can drop the owner from the live list via
            # scaffolding-trim / repair / compaction between flushes; an
            # agent-scoped set survives regardless. ---
            messages_b = [{"role": "user", "content": "q"},
                          {"role": "tool", "tool_name": "terminal",
                           "content": "zombie result", "tool_call_id": "call_x"}]
            agent._flush_messages_to_session_db(messages_b)

            rows = _rows(db)
            # NEITHER half of the split pair persisted → no dangling tool result.
            assert not any(r["role"] == "tool" and r["tool_call_id"] == "call_x" for r in rows), (
                "a tool result whose owner was suppressed in a PRIOR flush must ALSO "
                "be suppressed (agent-scoped set) — else it orphans (#48879)"
            )
            assert not any(
                r["tool_calls"] and "call_x" in str(r["tool_calls"]) for r in rows
            ), "the assistant(tool_calls) must stay suppressed"
        finally:
            db.close()


def test_synthetic_interrupt_close_tool_repair_persists():
    """req-3 (I1 completeness): close_interrupted_tool_sequence can leave a
    `tool` result as the tail then append the interrupt-close assistant. If a
    superseded flush reaches a `tool` row that is part of the interrupt-close
    repair (its owner is durable), it must PASS THROUGH so the reloaded
    transcript stays role/tool-pairing valid — the carve-out must not corrupt
    the repair path.
    """
    from hermes_state import SessionDB
    from agent.message_sanitization import close_interrupted_tool_sequence

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "t.db")
        try:
            agent = _make_agent(db)

            # A durable assistant(tool_calls) + its tool result land pre-stop.
            pre = [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c_r", "type": "function",
                                 "function": {"name": "terminal", "arguments": "{}"}}]},
                {"role": "tool", "tool_name": "terminal", "content": "res", "tool_call_id": "c_r"},
            ]
            agent._flush_messages_to_session_db(pre)

            # /stop lands; close_interrupted_tool_sequence appends the tail.
            agent._persist_superseded = True
            close_interrupted_tool_sequence(pre, interrupted_assistant_tail=False)
            agent._flush_messages_to_session_db(pre)

            rows = _rows(db)
            roles = [r["role"] for r in rows]
            # The durable pair survived AND the interrupt-close tail persisted.
            assert any(
                r["finish_reason"] == _INTERRUPT_CLOSE_FINISH_REASON for r in rows
            ), "the interrupt-close tail must persist"
            # No dangling: every tool row has its owner present (pairing valid).
            tool_ids = [r["tool_call_id"] for r in rows if r["role"] == "tool"]
            asst_call_ids = set()
            for r in rows:
                if r["tool_calls"]:
                    asst_call_ids |= {str(c.get("id") or c.get("tool_call_id"))
                                      for c in _as_list(r["tool_calls"])}
            for tid in tool_ids:
                assert str(tid) in asst_call_ids, f"tool result {tid} orphaned (no owner)"
        finally:
            db.close()


def _as_list(v):
    import json
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return []
    return v if isinstance(v, list) else []


def test_unexpected_new_user_row_fails_open_during_supersede():
    """Greptile-P2 / I5: a superseded turn only ever writes assistant + tool
    rows. If an anomalous NEW `user` (or system) row reaches the gate, it must
    FAIL OPEN and persist — dropping a real user message would be data loss."""
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "t.db")
        try:
            agent = _make_agent(db)
            agent._flush_messages_to_session_db([{"role": "user", "content": "q"}])

            agent._persist_superseded = True
            messages = [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "zombie continuation"},   # suppressed
                {"role": "user", "content": "REAL new user message"},       # must persist
            ]
            agent._flush_messages_to_session_db(messages)

            contents = [r["content"] for r in _rows(db)]
            assert "REAL new user message" in contents, (
                "an unexpected new user row must fail OPEN (never drop a real user message)"
            )
            assert "zombie continuation" not in contents, (
                "the assistant zombie row must still be suppressed"
            )
        finally:
            db.close()




