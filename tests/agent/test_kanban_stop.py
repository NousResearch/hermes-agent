"""Tests for the kanban worker turn-end stop guard.

The guard's contract, pinned here: a nudge orders the worker to end with a terminal board
call, so it may only fire when such a call can actually land — dispatcher-owned identity, a
``HERMES_KANBAN_TASK`` the tools accept *verbatim* (they scope on the raw env value, so a
padded id is refused by every lifecycle call), an int ``HERMES_KANBAN_RUN_ID``, and a board
row whose ``current_run_id`` is that run and whose status still accepts a handoff.
``HERMES_KANBAN_TASK`` in the env, alone, means nothing: a phantom id must not be nudged
about, and must never reach the gate's log line (the card's repro:
``HERMES_KANBAN_TASK=t_repro_board`` named a card no board has, while the run-ownership
guard refused every terminal call the nudge demanded). The same holds for a padded id: the
card exists and the run matches, but no terminal call can land.
"""

from __future__ import annotations

import logging

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    kanban_stop_target,
    session_called_kanban_terminal,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    """No inherited worker identity: the test process may itself be a dispatcher worker."""
    for var in (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_STOP_NUDGE",
        "HERMES_VERIFY_ON_STOP",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def board(tmp_path, monkeypatch):
    """A real, empty board DB, pinned as ``HERMES_KANBAN_DB``; returns the db module.

    A real board (not a stub) keeps these tests honest: the probe has to resolve a row, a run
    id and a status through the same connect path the kanban tools use.
    """
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb.init_db()
    return kb


def _connect():
    from hermes_cli import kanban_db_connect as kbc

    return kbc.connect_closing()


def _claimed_card(kb, title: str = "stop-gate card"):
    """Create + claim a card on the temp board; the claim opens the run the worker owns."""
    with _connect() as conn:
        tid = kb.create_task(conn, title=title)
    with _connect() as conn:
        task = kb.claim_task(conn, tid, claimer="pytest-stop-gate")
    assert task is not None and task.current_run_id is not None
    assert task.status == "running"
    return task


def _worker_env(monkeypatch, task_id: str, run_id=None) -> None:
    """Point the process env at ``task_id`` the way the dispatcher spawns a worker."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    if run_id is None:
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))


def _heartbeat_transcript() -> list[dict]:
    """A turn that made a non-terminal kanban call and then narrated a stop."""
    return [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "Let me write the comprehensive recipe.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_heartbeat", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_heartbeat", "tool_call_id": "1", "content": "ok"},
    ]


# ── The nudge fires only for a card/run this worker can actually terminate ───


def test_phantom_card_does_not_nudge(clear_kanban_env, board):
    """The card's repro: env-only TASK, no run id, no board row → no nudge at all."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_repro_board")

    assert kanban_stop_nudge_enabled() is False
    assert kanban_stop_target() is None
    assert build_kanban_stop_nudge(messages=[], attempts=0) is None
    assert build_kanban_stop_nudge(messages=_heartbeat_transcript(), attempts=0) is None


def test_phantom_card_does_not_nudge_even_with_a_decoy_card_on_the_board(clear_kanban_env, board):
    """Another card being claimed must not make a phantom id nudgeable either."""
    _claimed_card(board, title="decoy")
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_repro_board")

    assert kanban_stop_target() is None
    assert build_kanban_stop_nudge(messages=_heartbeat_transcript(), attempts=0) is None


def test_nudge_suppressed_without_a_run_id(clear_kanban_env, board):
    """A real running card, but the worker cannot name its run: every lifecycle tool is
    refused by the ownership guard, so the nudge would be an impossible demand."""
    task = _claimed_card(board)
    _worker_env(clear_kanban_env, task.id)

    assert kanban_stop_nudge_enabled() is False
    assert kanban_stop_target() is None
    assert build_kanban_stop_nudge(messages=_heartbeat_transcript(), attempts=0) is None


def test_nudge_suppressed_for_a_stale_run_id(clear_kanban_env, board):
    """A run id that is not the card's current run: ``kanban_complete``'s
    ``AND current_run_id = ?`` CAS would fail, so the nudge is withheld."""
    task = _claimed_card(board)
    _worker_env(clear_kanban_env, task.id, int(task.current_run_id) + 1)

    assert kanban_stop_target() is None
    assert build_kanban_stop_nudge(messages=_heartbeat_transcript(), attempts=0) is None


def test_nudge_suppressed_once_the_card_settled(clear_kanban_env, board):
    """Run still current but the status no longer accepts a handoff ('done'): no legal call."""
    task = _claimed_card(board)
    with _connect() as conn:
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (task.id,))
        conn.commit()
    _worker_env(clear_kanban_env, task.id, task.current_run_id)

    assert kanban_stop_target() is None
    assert build_kanban_stop_nudge(messages=_heartbeat_transcript(), attempts=0) is None


def test_nudge_suppressed_for_a_padded_env_task_id(clear_kanban_env, board):
    """The tools scope on the RAW env id, so a whitespace-padded ``HERMES_KANBAN_TASK`` is
    refused by every lifecycle call even though the card exists and the run id matches (review
    run 277). The probe must fail closed rather than nudge a call that cannot land; the strip
    in ``owned_kanban_task()`` must not paper over the mismatch."""
    task = _claimed_card(board)
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", f"  {task.id}  ")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))

    assert kanban_stop_nudge_enabled() is False
    assert kanban_stop_target() is None
    assert build_kanban_stop_nudge(messages=_heartbeat_transcript(), attempts=0) is None


def test_padded_env_task_id_is_refused_by_the_tools(clear_kanban_env, board):
    """Why the probe fails closed on a padded id: both tool paths refuse it, so a nudge in this
    state would order an impossible call. Pinned against the real ``tools.kanban_tools``
    helpers, not a restatement of them (do not widen the tools' raw compare)."""
    from tools import kanban_tools as kt

    task = _claimed_card(board)
    padded = f"  {task.id}  "
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", padded)
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))

    # Explicit clean id: the worker-scope check rejects it (the env id is the padded one).
    with pytest.raises(kt._Reject):
        kt._worker_guard("kanban_complete", {"task_id": task.id})
    # Implicit id: the guard resolves the raw padded value, which names no board row.
    assert kt._worker_guard("kanban_complete", {}) == padded
    with _connect() as conn:
        assert board.complete_task(
            conn, padded, summary="padded env id", expected_run_id=int(task.current_run_id)
        ) is False

    with _connect() as conn:
        after = board.get_task(conn, task.id)
    assert after.status == "running" and after.current_run_id == task.current_run_id


def test_probe_never_initializes_a_missing_board(clear_kanban_env, tmp_path, monkeypatch):
    """Resolving a card must stay read-only: ``kanban_db_connect.connect`` initializes a DB
    it finds missing, and a probe for a card that cannot exist must not create one."""
    missing = tmp_path / "no-such-board.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(missing))
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_repro_board")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "3")

    assert kanban_stop_target() is None
    assert build_kanban_stop_nudge(messages=_heartbeat_transcript(), attempts=0) is None
    assert not missing.exists()


def test_nudge_fires_for_the_owned_running_card(clear_kanban_env, board):
    """The case the guard exists for keeps working: dispatcher-owned worker, live run."""
    task = _claimed_card(board)
    _worker_env(clear_kanban_env, task.id, task.current_run_id)

    target = kanban_stop_target()
    assert target is not None
    assert (target.task_id, target.run_id) == (task.id, int(task.current_run_id))
    assert kanban_stop_nudge_enabled() is True

    nudge = build_kanban_stop_nudge(messages=_heartbeat_transcript(), attempts=0)
    assert nudge is not None
    assert task.id in nudge
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "kanban_request_review" in nudge
    assert "protocol" in nudge.lower()

    # A caller that already resolved the target gets the same text without a second probe.
    assert build_kanban_stop_nudge(messages=_heartbeat_transcript(), attempts=0, target=target) is not None
    # ... and the attempt budget still caps the loop.
    assert build_kanban_stop_nudge(messages=_heartbeat_transcript(), attempts=2) is None


def test_refused_terminal_call_ends_the_loop(clear_kanban_env, board):
    """Load-bearing escape: matching is on the CALL, not its success, so a terminal call the
    ownership guard refuses still ends the nudge loop (this is what ended the phantom probe)."""
    task = _claimed_card(board)
    _worker_env(clear_kanban_env, task.id, task.current_run_id)
    refusal = (
        "kanban_complete refused: this worker cannot resolve its HERMES_KANBAN_RUN_ID, so it "
        "cannot prove ownership of the card's current run."
    )
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": refusal},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages, attempts=0) is None


# ── Env switch + handoff suppression (unchanged behaviour) ───────────────────


def test_env_can_disable(clear_kanban_env, board):
    task = _claimed_card(board)
    _worker_env(clear_kanban_env, task.id, task.current_run_id)
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")

    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_disabled_inside_delegated_child(clear_kanban_env, board):
    from agent.delegation_context import delegated_child_context

    task = _claimed_card(board)
    _worker_env(clear_kanban_env, task.id, task.current_run_id)

    assert kanban_stop_nudge_enabled() is True
    with delegated_child_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None
    assert kanban_stop_nudge_enabled() is True


def test_nudge_disabled_inside_non_dispatcher_context(clear_kanban_env, board):
    from agent.delegation_context import non_dispatcher_owned_context

    task = _claimed_card(board)
    _worker_env(clear_kanban_env, task.id, task.current_run_id)

    assert kanban_stop_nudge_enabled() is True
    with non_dispatcher_owned_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None
    assert kanban_stop_nudge_enabled() is True


def test_no_nudge_after_kanban_complete(clear_kanban_env, board):
    task = _claimed_card(board)
    _worker_env(clear_kanban_env, task.id, task.current_run_id)
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


@pytest.mark.parametrize(
    "tool_name,who",
    [
        ("kanban_request_review", "build worker handing off for same-card review"),
        ("kanban_request_changes", "review agent sending the card back"),
    ],
)
def test_no_nudge_after_handoff_tool(clear_kanban_env, board, tool_name, who):
    """Handoff tools end the worker's turn just like complete/block.

    Both move the card out of ``running``, and the worker is told to call
    them — goals.py's continuation/finalize prompts name
    ``kanban_request_review``; the force-loaded sdlc-review skill names
    ``kanban_request_changes``. Nudging afterwards asks a worker that did
    the right thing to close a card it must not close.
    """
    task = _claimed_card(board)
    _worker_env(clear_kanban_env, task.id, task.current_run_id)
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": tool_name, "tool_call_id": "1", "content": "ok"},
    ]
    assert session_called_kanban_terminal(messages) is True, who
    assert build_kanban_stop_nudge(messages=messages) is None


def test_nudge_still_fires_for_non_terminal_kanban_tool(clear_kanban_env, board):
    """Widening the set must not swallow the case the guard exists for."""
    task = _claimed_card(board)
    _worker_env(clear_kanban_env, task.id, task.current_run_id)
    messages = [
        {
            "role": "assistant",
            "content": "Let me open the review next.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_comment", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_comment", "tool_call_id": "1", "content": "ok"},
    ]
    assert session_called_kanban_terminal(messages) is False
    nudge = build_kanban_stop_nudge(messages=messages)
    assert nudge is not None
    # The nudge offers every worker exit, not just close-out; a card that must go
    # through review must never be steered to ``kanban_complete`` alone.
    assert "kanban_request_review" in nudge and "kanban_block" in nudge


# ── Stop-gate integration: silent on phantom, and it logs the validated card ─


class _FakeAgent:
    """The surface ``apply_stop_gates`` touches, nothing more."""

    def __init__(self) -> None:
        self.session_id = "s_stop_gate_test"
        self.platform = "cli"
        self._session_messages: list[dict] = []
        self._kanban_stop_nudges = 0
        self.interims: list[dict] = []
        self.diagnostics: list[str] = []

    def _interim_content_was_streamed(self, content) -> bool:
        return False

    def _emit_interim_assistant_message(self, msg) -> None:
        self.interims.append(msg)

    def _flush_messages_to_session_db(self, messages, history) -> None:
        return None

    def _emit_diagnostic_status(self, text) -> None:
        self.diagnostics.append(text)


def _run_stop_gates(agent, messages):
    from agent.turn_stop_gates import apply_stop_gates

    content = "All set — I will finish this later."
    final_msg = {"role": "assistant", "content": content}
    return apply_stop_gates(
        agent,
        final_msg,
        final_response=content,
        messages=messages,
        conversation_history=[],
        pending_verification_response=None,
        pending_verification_response_previewed=None,
    )


def test_stop_gate_is_silent_for_a_card_the_board_never_had(clear_kanban_env, board, caplog):
    """The unsatisfiable-nudge state is unreachable through the gate: no nudge row, no
    diagnostic, and no log line naming a card the board has never had."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_repro_board")
    agent = _FakeAgent()
    messages = [{"role": "user", "content": "work kanban task"}]

    with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
        verdict = _run_stop_gates(agent, messages)

    assert verdict.continue_turn is False
    assert verdict.final_response == "All set — I will finish this later."
    assert agent._kanban_stop_nudges == 0
    assert len(messages) == 1  # candidate answer kept, no synthetic nudge row appended
    assert agent.diagnostics == []
    assert "t_repro_board" not in caplog.text
    assert "kanban stop-loop nudge issued" not in caplog.text


def test_stop_gate_is_silent_for_a_padded_env_id(clear_kanban_env, board, caplog):
    """Gate-level pin for the padded-id residual (review run 277): the card exists and the run
    id matches, but the tools compare the raw env id — so no terminal call could land, and the
    gate must produce no nudge row, no diagnostic, and no log line."""
    task = _claimed_card(board)
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", f"  {task.id}  ")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    agent = _FakeAgent()
    messages = [{"role": "user", "content": "work kanban task"}]

    with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
        verdict = _run_stop_gates(agent, messages)

    assert verdict.continue_turn is False
    assert verdict.final_response == "All set — I will finish this later."
    assert agent._kanban_stop_nudges == 0
    assert len(messages) == 1  # candidate answer kept, no synthetic nudge row appended
    assert agent.diagnostics == []
    assert "kanban stop-loop nudge issued" not in caplog.text


def test_stop_gate_nudges_and_logs_only_the_validated_card(clear_kanban_env, board, caplog):
    task = _claimed_card(board)
    _worker_env(clear_kanban_env, task.id, task.current_run_id)
    agent = _FakeAgent()
    messages = [{"role": "user", "content": "work kanban task"}]

    with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
        verdict = _run_stop_gates(agent, messages)

    assert verdict.continue_turn is True
    assert agent._kanban_stop_nudges == 1
    assert any(m.get("role") == "user" and task.id in str(m.get("content")) for m in messages)
    assert f"kanban stop-loop nudge issued (attempt 1) task={task.id}" in caplog.text
    assert agent.diagnostics  # operator-visible status for the same event
