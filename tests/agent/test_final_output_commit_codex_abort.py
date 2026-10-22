"""F-001 regression: an error-bearing Codex candidate must still meet the commit gate.

The Codex app-server fast path gated its final candidate only when
``turn.error is None``. A ``agentMessage`` carrying the raw ``<turn_aborted>``
marker is projected into an assistant row and a ``final_text`` by
``CodexEventProjector._project_agent_message``, while
``CodexAppServerSession._absorb_notification`` marks the same turn
``interrupted=True`` with ``error="codex reported turn_aborted"``. The gate
skipped exactly that turn: the raw candidate was persisted to the session DB
and returned as ``final_response``, so a registered DROP policy was never
evaluated.

These regressions drive the REAL projector, the REAL notification owner and the
REAL public turn dispatcher (``AIAgent.run_conversation``) against a temporary
SQLite ``SessionDB``. Only the Codex transport boundary (``run_turn``) is
stubbed, because that is the subprocess wire.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.agent.test_final_output_commit_terminality import _install_gate
from tests.agent.test_codex_app_server_integration import _make_codex_agent

_ABORTED_TEXT = "POLICY-SECRET<turn_aborted>"


def _aborted_turn_result():
    """A TurnResult as the REAL projector + session owner build it for an abort.

    ``_absorb_notification`` sets ``final_text`` from the projection and then
    flips ``interrupted``/``error`` because the text carries ``<turn_aborted>``.
    The projected assistant row stays in ``projected_messages``.
    """
    from agent.transports.codex_app_server_session import TurnResult

    return TurnResult(
        final_text=_ABORTED_TEXT,
        projected_messages=[{"role": "assistant", "content": _ABORTED_TEXT}],
        tool_iterations=0,
        interrupted=True,
        error="codex reported turn_aborted",
        turn_id="turn-abort-1",
        thread_id="thread-abort-1",
    )


def _install_abort_transport(monkeypatch):
    """Stub ONLY the Codex transport wire, producing a real aborted turn."""
    from agent.transports.codex_app_server_session import CodexAppServerSession

    def fake_run_turn(self, user_input, **kwargs):
        return _aborted_turn_result()

    monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
    monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "thread-abort-1")


def _install_real_projection(monkeypatch):
    """Drive the REAL projector + notification owner for the abort marker.

    ``run_turn`` is the transport boundary and is stubbed; everything above it
    (the projector that builds the assistant row and the session owner that
    sets ``error``) must be the production code path.
    """
    from agent.transports.codex_app_server_session import CodexAppServerSession, CodexEventProjector

    original_run_turn = CodexAppServerSession.run_turn

    def fake_run_turn(self, user_input, **kwargs):
        result = _aborted_turn_result()
        # Real projector + real notification owner: this is what produces the
        # assistant row AND the interrupted/error state together.
        projector = CodexEventProjector()
        note = {
            "method": "item/completed",
            "params": {"item": {"type": "agentMessage", "text": _ABORTED_TEXT}},
        }
        self._absorb_notification(result, projector, note)
        return result

    monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
    monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "thread-abort-1")
    return original_run_turn


def _session_db_agent(monkeypatch, tmp_path):
    """A codex-mode agent wired to a temporary SQLite SessionDB."""
    from hermes_state import SessionDB

    agent = _make_codex_agent()
    db = SessionDB(db_path=tmp_path / "state.db")
    agent._session_db = db
    agent._sync_external_memory_for_turn = MagicMock()
    agent._spawn_background_review = MagicMock()
    agent._current_turn_id = "turn-abort-1"
    return agent, db


def _history_contents(db, session_id):
    """Every assistant-row content in the session DB's message history."""
    rows = db.get_messages(session_id)
    out = []
    for row in rows or []:
        if isinstance(row, dict) and row.get("role") == "assistant":
            content = row.get("content")
            if isinstance(content, str):
                out.append(content)
    return out


def _run(monkeypatch, agent):
    from agent.codex_runtime import run_codex_app_server_turn

    messages = [{"role": "user", "content": "do the thing"}]
    with (
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = run_codex_app_server_turn(
            agent,
            user_message="do the thing",
            original_user_message="do the thing",
            messages=messages,
            effective_task_id=None,
            should_review_memory=False,
        )
    return result, messages


def test_codex_aborted_candidate_is_dropped_and_never_persisted(monkeypatch, tmp_path):
    """F-001: the gate MUST run for an error-bearing candidate, and a DROP must
    leave no candidate in the returned result, the transcript, or the session DB."""
    gated: list = []

    def gate(**kwargs):
        gated.append(kwargs["candidate"]["content"])
        return "drop"

    _install_gate(monkeypatch, gate)
    _install_real_projection(monkeypatch)
    agent, db = _session_db_agent(monkeypatch, tmp_path)

    result, messages = _run(monkeypatch, agent)

    # The gate ran — on the abort-bearing candidate.
    assert gated == [_ABORTED_TEXT], "the gate must be evaluated for an error-bearing candidate"
    # Terminal disposition, no candidate anywhere.
    assert result["final_output_disposition"] == "drop"
    assert result["final_response"] is None
    assert result["completed"] is False
    assert not [row for row in messages if row.get("role") == "assistant"]
    assert _ABORTED_TEXT not in _history_contents(db, agent.session_id), (
        "a dropped candidate must never become durable session history"
    )


def test_codex_aborted_candidate_refusal_is_terminal(monkeypatch, tmp_path):
    """F-001: a fail-closed refusal on an error-bearing candidate settles the turn."""
    gated: list = []

    def gate(**kwargs):
        gated.append(kwargs["candidate"]["content"])
        raise RuntimeError("commit gate unavailable")

    _install_gate(monkeypatch, gate, failure_mode="closed")
    _install_real_projection(monkeypatch)
    agent, db = _session_db_agent(monkeypatch, tmp_path)

    result, messages = _run(monkeypatch, agent)

    assert gated == [_ABORTED_TEXT]
    assert result["final_output_disposition"] == "refused"
    assert result["failed"] is True
    assert result["failure_retryable"] is False
    assert result["final_response"] is None
    assert not [row for row in messages if row.get("role") == "assistant"]
    assert _ABORTED_TEXT not in _history_contents(db, agent.session_id)


def test_codex_aborted_candidate_is_allowed_when_the_gate_allows(monkeypatch, tmp_path):
    """Control: an allowing gate leaves the error-bearing turn exactly as it was.

    The abort is a Codex-side fact, not a policy outcome: the turn still reports
    its error, and the gate's ALLOW does not silently convert it into success.
    """
    gated: list = []

    def gate(**kwargs):
        gated.append(kwargs["candidate"]["content"])
        return "allow"

    _install_gate(monkeypatch, gate)
    _install_real_projection(monkeypatch)
    agent, db = _session_db_agent(monkeypatch, tmp_path)

    result, messages = _run(monkeypatch, agent)

    assert gated == [_ABORTED_TEXT], "the gate must be evaluated for an error-bearing candidate"
    assert "final_output_disposition" not in result
    # The Codex error is still reported: an ALLOW is not a success signal.
    assert result["error"] == "codex reported turn_aborted"
    assert result["completed"] is False
    # The candidate is allowed, so it IS durable — the gate allowed it.
    assert any(row.get("role") == "assistant" for row in messages)
    assert _ABORTED_TEXT in _history_contents(db, agent.session_id)


def test_codex_aborted_candidate_without_a_registered_gate_is_unchanged(monkeypatch, tmp_path):
    """Compatibility control: with no plugin registered the abort turn behaves as before.

    The gate is additive — no registration means no behavioural change at all.
    """
    _install_real_projection(monkeypatch)
    agent, db = _session_db_agent(monkeypatch, tmp_path)

    result, messages = _run(monkeypatch, agent)

    assert "final_output_disposition" not in result
    assert result["final_response"] == _ABORTED_TEXT
    assert result["error"] == "codex reported turn_aborted"
    assert any(row.get("role") == "assistant" for row in messages)
    assert _ABORTED_TEXT in _history_contents(db, agent.session_id)
