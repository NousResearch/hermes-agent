"""Regression tests: an internal auto-resume continuation must NOT persist an
empty user row (2026-07-16 — the `/undo` "your message (no text)" bug).

The resume trigger is MessageEvent(text="", internal=True); the model gets its
resume prompt separately, but the empty user row was being persisted durably,
polluting the transcript and giving /undo a "(no text)" boundary. The fix stamps
that row `_empty_resume_synthetic` so the flush drops it.
"""
from run_agent import AIAgent, _is_ephemeral_scaffolding, _EPHEMERAL_SCAFFOLDING_FLAGS


class _CapturingSessionDB:
    def __init__(self):
        self.rows = []

    def append_message(self, session_id, role, content=None, **kwargs):
        self.rows.append({"role": role, "content": content})
        return len(self.rows)


def _agent_with_capturing_db():
    agent = AIAgent.__new__(AIAgent)
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._session_db = _CapturingSessionDB()
    agent._session_db_created = True
    agent._last_flushed_db_idx = 0
    agent.session_id = "sess-test"
    return agent


def test_flag_is_registered_as_ephemeral():
    assert "_empty_resume_synthetic" in _EPHEMERAL_SCAFFOLDING_FLAGS
    assert _is_ephemeral_scaffolding({"role": "user", "content": "", "_empty_resume_synthetic": True})
    # a normal empty user row (no flag) is NOT dropped
    assert not _is_ephemeral_scaffolding({"role": "user", "content": ""})


def test_flush_drops_empty_resume_user_row_keeps_assistant():
    """AC1/I4: the flagged empty resume user row is dropped from persistence; the
    resumed turn's assistant reply still persists."""
    agent = _agent_with_capturing_db()
    messages = [
        {"role": "user", "content": "the real prior question"},
        {"role": "assistant", "content": "my prior reply"},
        # the synthetic empty resume-trigger user row (flagged):
        {"role": "user", "content": "", "_empty_resume_synthetic": True},
        # the resumed turn's real assistant output:
        {"role": "assistant", "content": "resumed: here is the continuation"},
    ]
    agent._flush_messages_to_session_db(messages, conversation_history=[])
    persisted = agent._session_db.rows
    # No empty user row landed; the assistant reply did.
    assert [r["role"] for r in persisted] == ["user", "assistant", "assistant"]
    assert all(r["content"] != "" for r in persisted)
    assert persisted[-1]["content"] == "resumed: here is the continuation"


def test_flush_keeps_real_resume_user_row():
    """I2/AC3 (guard-unit): a resume turn whose user row carries REAL text (NOT
    flagged) still persists — only the flagged empty row is dropped."""
    agent = _agent_with_capturing_db()
    messages = [
        {"role": "assistant", "content": "prior reply"},
        # a resume turn that DID carry a real queued user message → not flagged
        {"role": "user", "content": "a real question that arrived during the restart"},
        {"role": "assistant", "content": "answer"},
    ]
    agent._flush_messages_to_session_db(messages, conversation_history=[])
    persisted = agent._session_db.rows
    assert [r["role"] for r in persisted] == ["assistant", "user", "assistant"]
    assert any("real question that arrived" in (r["content"] or "") for r in persisted)


def test_append_gate_and_ephemeral_compose():
    """R3 (pass-1 RC3): the append-gate (#341, _persist_superseded) and the
    ephemeral drop are BOTH drop-decisions on a row. The ephemeral check runs
    FIRST in the flush loop, so a flagged empty-resume row is dropped before the
    gate evaluates it — the two compose without crashing or double-processing."""
    agent = _agent_with_capturing_db()
    agent._persist_superseded = True  # append-gate active (a /stop'd turn)
    agent._superseded_suppressed_tool_call_ids = set()
    messages = [
        # flagged empty resume row, during a superseded turn — dropped by the
        # ephemeral check (which precedes the append-gate), no interaction bug.
        {"role": "user", "content": "", "_empty_resume_synthetic": True},
    ]
    # Must not raise, and must drop the flagged row.
    agent._flush_messages_to_session_db(messages, conversation_history=[])
    persisted = agent._session_db.rows
    assert not any(r["role"] == "user" and r["content"] == "" for r in persisted)
    assert persisted == []  # nothing persisted (row was ephemeral-dropped)


def test_build_turn_context_stamps_the_actual_user_row():
    """Pass-1 RC3 + pass-3 (executing, not source-inspection): the stamp helper
    lands `_empty_resume_synthetic` on the exact user_msg dict and consumes the
    flag. Drives the REAL maybe_stamp_empty_resume_row."""
    from agent.turn_context import maybe_stamp_empty_resume_row

    class _Agent:
        _suppress_user_turn_persist = True
    a = _Agent()
    user_msg = {"role": "user", "content": ""}
    stamped = maybe_stamp_empty_resume_row(a, user_msg)
    assert stamped is True
    assert user_msg.get("_empty_resume_synthetic") is True
    # flag consumed (reset) so it can't apply to a second row
    assert a._suppress_user_turn_persist is False


def test_leaked_flag_cannot_drop_next_real_turn_row():
    """🔴 Pass-2/3/4 B1 — the invert-the-bug data-loss path.

    EXECUTED (the consume side): a leaked flag, once the gateway's per-turn reset
    has run, does NOT stamp turn N+1's REAL user row, and that row PERSISTS
    through the real flush. The consume helper (maybe_stamp_empty_resume_row) is
    driven for real.

    The gateway's top-of-turn reset itself is verified structurally (pass-4):
    (a) it EXISTS in _run_agent_inner source, and (b) it runs STRAIGHT-LINE
    before the resume-pending set — no branch/continue/return/loop between the
    reset statement and the `if _is_resume_pending:` set — so reset-before-set is
    guaranteed on every path that reaches the set, not merely by line order. The
    LIVE reset is covered end-to-end by the Apollo E2E (AC6). Driving the
    2000-line _run_agent_inner in a unit test is not tractable; the structural
    proof + executed consume + live E2E together close the silent-loss path.
    """
    from agent.turn_context import maybe_stamp_empty_resume_row
    import inspect, ast
    import gateway.run as gr

    # (a) the reset exists, and (b) it is STRAIGHT-LINE before the set — assert
    # there is no control-flow break (return/continue/break/raise) textually
    # between the reset line and the resume-pending `if` that guards the set.
    gsrc = inspect.getsource(gr.GatewayRunner._run_agent_inner)
    reset_marker = "agent._suppress_user_turn_persist = False"
    resume_if_marker = "if _is_resume_pending:"
    # Markers must be UNIQUE, else first-match str.index() could slice the wrong
    # range and the control-flow assertions pass vacuously (Greptile-P2).
    assert gsrc.count(reset_marker) == 1, "reset marker must appear exactly once"
    assert gsrc.count(resume_if_marker) == 1, "resume-if marker must appear exactly once"
    between = gsrc[gsrc.index(reset_marker) + len(reset_marker): gsrc.index(resume_if_marker)]
    # No control-flow break (at ANY indentation) between the reset and the set —
    # the only intervening code is the _is_resume_pending / _has_fresh_tool_tail
    # computations + a helper call, so the reset always runs before the set.
    import re
    for _kw in ("return", "continue", "break", "raise"):
        assert not re.search(rf"(?m)^\s+{_kw}\b", between), (
            f"unexpected control-flow break ({_kw!r}) between reset and set"
        )

    class _Agent:
        pass
    a = _Agent()

    # Turn N: gateway set the flag for an empty resume turn, then the turn ABORTED
    # before build_turn_context consumed it → flag left stale on the (cached) agent.
    a._suppress_user_turn_persist = True

    # Turn N+1: the gateway's straight-line top-of-turn reset (asserted above)
    # runs first. Execute its effect, then run the REAL consume helper against a
    # REAL user message.
    a._suppress_user_turn_persist = False  # the reset the gateway performs each turn
    real_user_msg = {"role": "user", "content": "a real question I typed"}
    stamped = maybe_stamp_empty_resume_row(a, real_user_msg)
    assert stamped is False, "a leaked flag must not stamp the next real turn"
    assert "_empty_resume_synthetic" not in real_user_msg

    # And prove the row then PERSISTS through the real flush (not dropped).
    agent = _agent_with_capturing_db()
    agent._flush_messages_to_session_db([real_user_msg], conversation_history=[])
    persisted = agent._session_db.rows
    assert any("a real question I typed" in (r["content"] or "") for r in persisted), (
        "the real user row must survive — no silent data loss from a leaked flag"
    )


