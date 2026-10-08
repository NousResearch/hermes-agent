"""Terminality regressions for the ``llm_final_output_commit`` final-output gate.

Re-review at ``0557d8e`` raised two P1s at the ownership boundary the hook is
supposed to control:

* **P1-A** — an explicit ``DROP`` was reconstituted by ``finalize_turn``: the
  loop handed a populated ``final_response`` to the finalizer, whose
  ``_close_transcript_tail`` invariant ("delivered final_response ⇒ assistant
  row") re-created the row, persisted it and returned it. Final candidates
  *minted inside the finalizer* (the max-iteration summary and stream recovery)
  never reached the gate at all.
* **P1-B** — a fail-closed gate refusal is raised after ``_run_api_retry_loop()``
  has finished, so it never reaches ``handle_api_error()`` /``classify_api_error()``; the surrounding ``except Exception`` fed it to
  ``handle_outer_loop_error()``, which had no refusal branch and returned
  ``fallthrough`` — dispatching another provider iteration through the same
  fail-closed boundary.

These are native conversation-loop regressions (real ``AIAgent`` +
``run_conversation``, real finalizer), not middleware-function tests.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Reuses the real-AIAgent fixture (mirrors test_run_agent's) and the response builders.
from tests.agent.test_dropped_tool_call_recovery import loop_agent  # noqa: F401
from tests.agent.test_run_agent import _mock_response

_SECRET = "TOP-SECRET-CANDIDATE"


# --------------------------------------------------------------------------- helpers


def _install_gate(monkeypatch, callback, *, failure_mode: str = "closed"):
    """Register one ``llm_final_output_commit`` callback on a bare delivery manager."""
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    callback._hermes_failure_mode = failure_mode
    manager._middleware.setdefault("llm_final_output_commit", []).append(callback)
    monkeypatch.setattr("hermes_cli.plugins._delivery_manager", lambda: manager)
    return manager


def _recording_persist(target):
    """Patch ``_persist_session`` so the durable snapshot the finalizer wrote is kept."""
    rows: list[list[dict]] = []
    return (
        patch.object(
            target,
            "_persist_session",
            side_effect=lambda messages, conversation_history=None: rows.append(list(messages)),
        ),
        rows,
    )


def _run_turn(agent, message, conversation_history=None):
    """Run one turn with the persistence seams the loop tests stub out.

    ``conversation_history`` is how the CLI/gateway hand the previous turn's
    transcript to the next one — the replay surface a DROP must never reach.
    """
    persist_patch, persisted = _recording_persist(agent)
    with (
        persist_patch,
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(message, conversation_history=conversation_history)
    return result, persisted


def _content_rows(rows) -> list:
    """Assistant contents from either a message list or a list of persisted snapshots."""
    flat: list[dict] = []
    if rows and isinstance(rows[0], dict):
        flat = list(rows)
    else:
        for row in rows or []:
            flat.extend(m for m in row if isinstance(m, dict))
    return [m.get("content") for m in flat if m.get("role") == "assistant"]


def _sent_messages(agent, index: int = -1) -> list[dict]:
    call = agent.client.chat.completions.create.call_args_list[index]
    payload = call.kwargs.get("messages") if call.kwargs else None
    if payload is None:
        payload = call.args[0].get("messages")
    return list(payload or [])


# ------------------------------------------------------------------- P1-A: DROP


def test_explicit_drop_is_terminal_through_finalization(monkeypatch, loop_agent):
    """DROP must not be reconstituted by finalize_turn.

    The loop previously broke with ``final_response`` still populated, and the
    finalizer's tail-close invariant re-appended it as a durable assistant row.
    """
    gated: list = []

    def gate(**kwargs):
        gated.append(kwargs["candidate"]["content"])
        return "drop"

    _install_gate(monkeypatch, gate)
    loop_agent.client.chat.completions.create.side_effect = [
        _mock_response(content=_SECRET, finish_reason="stop"),
    ]

    result, persisted = _run_turn(loop_agent, "say something secret")

    # No second provider iteration was dispatched.
    assert loop_agent.client.chat.completions.create.call_count == 1
    # The dropped candidate is not returned ...
    assert result["final_response"] is None
    assert result["final_output_disposition"] == "drop"
    assert result["turn_exit_reason"] == "final_output_dropped"
    assert result["completed"] is False
    # ... not present in the returned transcript ...
    assert _SECRET not in _content_rows([result["messages"]])
    # ... and not re-created by the finalizer's durable snapshot.
    assert persisted, "finalize_turn never persisted the transcript"
    assert _SECRET not in _content_rows(persisted)
    assert not any(row.get("role") == "assistant" for row in persisted[-1])
    # One evaluation only: the finalizer must not re-run (and could not reverse) it.
    assert gated == [_SECRET]


def test_dropped_candidate_is_not_replayed_on_the_next_turn(monkeypatch, loop_agent):
    """A dropped candidate has no replay eligibility: neither the transcript handed to
    the next turn nor the request built from it may contain it."""
    _install_gate(monkeypatch, lambda **kwargs: "drop")
    loop_agent.client.chat.completions.create.side_effect = [
        _mock_response(content=_SECRET, finish_reason="stop"),
    ]
    result, _ = _run_turn(loop_agent, "say something secret")
    assert result["final_response"] is None
    assert _SECRET not in _content_rows([result["messages"]])

    # Turn 2 replays turn 1's transcript exactly the way the CLI/gateway hand it over.
    _install_gate(monkeypatch, lambda **kwargs: "allow")
    loop_agent.client.chat.completions.create.side_effect = [
        _mock_response(content="second answer", finish_reason="stop"),
    ]
    second, _ = _run_turn(
        loop_agent, "and now?", conversation_history=result["messages"]
    )
    assert second["final_response"] == "second answer"

    replayed_text = [
        m.get("content") for m in _sent_messages(loop_agent, -1) if isinstance(m, dict)
    ]
    # Turn 1's user turn reaches turn 2 (alternation repair merges it with the new
    # prompt, because no assistant row closed it) — carrying no dropped candidate.
    assert any("say something secret" in (text or "") for text in replayed_text), (
        "turn-1 history should carry to turn 2"
    )
    assert _SECRET not in replayed_text, "a dropped candidate must never be replayed"


def test_allow_is_evaluated_once_across_commit_point_and_finalizer(monkeypatch, loop_agent):
    """One-evaluation semantics plus the ALLOW persistence control."""
    gated: list = []

    def gate(**kwargs):
        gated.append(kwargs["candidate"]["content"])
        return "allow"

    _install_gate(monkeypatch, gate)
    loop_agent.client.chat.completions.create.side_effect = [
        _mock_response(content="Here is your answer.", finish_reason="stop"),
    ]

    result, persisted = _run_turn(loop_agent, "hello")

    assert result["completed"] is True
    assert result["final_response"] == "Here is your answer."
    assert "final_output_disposition" not in result
    assert _content_rows(persisted[-1]) == ["Here is your answer."]
    assert gated == ["Here is your answer."], (
        "the allowed candidate must be gated exactly once across "
        "finish_text_response + finalize_turn"
    )


# ------------------------------------------------- P1-A: finalizer-minted candidates


def _finalizer_agent():
    from tests.agent.test_turn_finalizer_iteration_limit_exit import _LimitAgent

    agent = _LimitAgent()
    # _resolve_budget_fallback reports the extra summary request on the diagnostic
    # surface before it calls _handle_max_iterations.
    agent._emit_diagnostic_status = lambda *args, **kwargs: None
    return agent


def _finalize(agent, **overrides):
    from agent.turn_finalizer import finalize_turn

    params = dict(
        final_response=None,
        api_call_count=60,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="unknown",
    )
    params.update(overrides)
    return finalize_turn(agent, **params)


@pytest.fixture(autouse=True)
def _no_plugin_hooks(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])


def test_max_iteration_summary_is_gated_and_can_be_dropped(monkeypatch):
    """The budget-fallback summary is created INSIDE the finalizer: it must cross the
    gate instead of escaping as durable, replayable text."""
    gated: list = []

    def gate(**kwargs):
        gated.append(kwargs["candidate"]["content"])
        return "drop"

    _install_gate(monkeypatch, gate)
    agent = _finalizer_agent()

    result = _finalize(agent)

    assert agent._handle_max_iterations_called is True
    assert gated == ["summary from extra call"], "the finalizer-minted summary must be gated"
    assert result["final_response"] is None
    assert result["final_output_disposition"] == "drop"
    assert result["completed"] is False
    assert [row.get("role") for row in agent.persisted_messages] == ["user"]


def test_stream_recovered_final_text_is_gated_and_can_be_dropped(monkeypatch):
    """Stream recovery promotes ``_current_streamed_assistant_text`` to the final
    candidate inside the finalizer: same commit point, same gate."""
    gated: list = []

    def gate(**kwargs):
        gated.append(kwargs["candidate"]["content"])
        return "drop"

    _install_gate(monkeypatch, gate)
    agent = _finalizer_agent()
    agent._current_streamed_assistant_text = "STREAMED-CANDIDATE"
    agent.iteration_budget = SimpleNamespace(remaining=50, used=10, max_total=60)

    result = _finalize(agent, api_call_count=10)

    assert gated == ["STREAMED-CANDIDATE"]
    assert result["final_response"] is None
    assert result["final_output_disposition"] == "drop"
    assert [row.get("role") for row in agent.persisted_messages] == ["user"]


def test_stream_recovered_text_is_persisted_when_the_gate_allows(monkeypatch):
    """Positive control: the recovery path itself must keep working under an allowing
    gate (the gate is a policy check, not a blocker)."""
    gated: list = []

    def gate(**kwargs):
        gated.append(kwargs["candidate"]["content"])
        return "allow"

    _install_gate(monkeypatch, gate)
    agent = _finalizer_agent()
    agent._current_streamed_assistant_text = "STREAMED-RECOVERY"
    agent.iteration_budget = SimpleNamespace(remaining=50, used=10, max_total=60)

    result = _finalize(agent, api_call_count=10)

    assert result["final_response"] == "STREAMED-RECOVERY"
    assert "final_output_disposition" not in result
    assert _content_rows(agent.persisted_messages) == ["STREAMED-RECOVERY"]
    assert gated == ["STREAMED-RECOVERY"]


def test_finalizer_refusal_settles_the_turn_instead_of_being_swallowed(monkeypatch):
    """A fail-closed refusal raised at the finalizer's own commit point is past the
    loop's error owner: it must settle the result, not disappear into
    ``cleanup_errors`` while the refused text is still returned."""
    def gate(**kwargs):
        raise RuntimeError("commit gate unavailable")

    _install_gate(monkeypatch, gate, failure_mode="closed")
    agent = _finalizer_agent()
    agent._current_streamed_assistant_text = "REFUSED-CANDIDATE"
    agent.iteration_budget = SimpleNamespace(remaining=50, used=10, max_total=60)

    result = _finalize(agent, api_call_count=10)

    assert result["final_response"] is None
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["failure_reason"] == "llm_stream_middleware_refusal"
    assert result["failure_retryable"] is False
    assert result["final_output_disposition"] == "refused"
    assert result.get("cleanup_errors") is None, "the refusal must not be a cleanup error"
    assert [row.get("role") for row in agent.persisted_messages] == ["user"]


# ---------------------------------------------------- P1-B: terminal refusal branch


@pytest.mark.parametrize("shape", ["exception", "malformed", "deferred"])
def test_closed_gate_refusal_never_dispatches_a_second_provider_call(
    monkeypatch, loop_agent, shape
):
    """The reviewer's witness: ``api_call_count=1``, ``max_iterations=20``, a fail-closed
    refusal at the gate. It used to return ``fallthrough`` and run the loop again."""
    def gate(**kwargs):
        if shape == "exception":
            raise RuntimeError("commit gate unavailable")
        if shape == "malformed":
            return {"txt": "SAFE"}
        async def _deferred():
            return "SAFE"
        return _deferred()

    _install_gate(monkeypatch, gate, failure_mode="closed")
    loop_agent.client.chat.completions.create.side_effect = [
        _mock_response(content=_SECRET, finish_reason="stop"),
    ]

    result, persisted = _run_turn(loop_agent, "say something secret")

    assert loop_agent.client.chat.completions.create.call_count == 1, (
        "a fail-closed final-output refusal is terminal — no retry, no failover"
    )
    assert result["failed"] is True
    assert result["failure_reason"] == "llm_stream_middleware_refusal"
    assert result["failure_retryable"] is False
    assert result["final_output_disposition"] == "refused"
    assert result["completed"] is False
    assert _SECRET not in (result["final_response"] or "")
    assert _SECRET not in _content_rows([result["messages"]])
    assert persisted and _SECRET not in _content_rows(persisted)


class _OuterLoopAgent:
    """Minimal double for ``handle_outer_loop_error`` (the outer owner every provider
    loop shares)."""

    max_iterations = 30
    suppress_status_output = True
    session_id = "sess-outer"
    log_prefix = ""

    def __init__(self, provider="openrouter", api_mode="chat_completions"):
        self.provider = provider
        self.api_mode = api_mode

    def _safe_print(self, *args, **kwargs):
        pass

    def _persist_session(self, messages, conversation_history):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _outer_error(agent, error, **overrides):
    from agent.turn_loop_errors import handle_outer_loop_error

    params = dict(
        e=error,
        _outer_error_count=0,
        api_call_count=1,
        messages=[],
        conversation_history=[],
        _turn_exit_reason="unknown",
        failed=False,
        final_response=None,
    )
    params.update(overrides)
    return handle_outer_loop_error(agent, **params)


@pytest.mark.parametrize(
    ("provider", "api_mode"),
    [
        ("openrouter", "chat_completions"),
        ("bedrock", "anthropic_messages"),
        ("openai-codex", "responses"),
    ],
)
def test_outer_owner_breaks_on_a_refusal_before_generic_retry(provider, api_mode):
    """Typed terminal branch ahead of the generic classification, on the owner that
    chat, Anthropic and Codex response processing all share."""
    from hermes_cli.middleware import LLMStreamMiddlewareRefusal

    agent = _OuterLoopAgent(provider=provider, api_mode=api_mode)
    refusal = LLMStreamMiddlewareRefusal(
        RuntimeError("commit gate unavailable"), callback_name="privacy_gate"
    )

    verdict = _outer_error(agent, refusal, final_response=_SECRET)

    assert verdict.action == "break"
    assert verdict._turn_exit_reason == "final_output_refused"
    assert verdict.failed is True, "a fail-closed refusal settles the attempt as refused"
    assert verdict.final_response is None, "the refused candidate must not be returned"
    assert verdict._outer_error_count == 0, "the outer-error retry budget is not consumed"


def test_ordinary_outer_error_still_retries(loop_agent):
    """Control: an ordinary escaped exception must keep its recovery behavior
    (``fallthrough`` → another provider iteration)."""
    from agent.turn_loop_errors import handle_outer_loop_error

    agent = _OuterLoopAgent()
    try:
        raise RuntimeError("upstream hiccup")
    except RuntimeError as exc:
        verdict = handle_outer_loop_error(
            agent,
            e=exc,
            _outer_error_count=0,
            api_call_count=1,
            messages=[],
            conversation_history=[],
            _turn_exit_reason="unknown",
            failed=False,
            final_response=None,
        )
    assert verdict.action == "fallthrough"
    assert verdict._outer_error_count == 1

    # ... and the same remains true end to end: a malformed provider response that
    # escapes response processing is retried, not settled.
    loop_agent.client.chat.completions.create.side_effect = [
        SimpleNamespace(model="test/model"),  # no .choices → response processing raises
        _mock_response(content="recovered", finish_reason="stop"),
    ]
    result, _ = _run_turn(loop_agent, "hello")
    assert loop_agent.client.chat.completions.create.call_count == 2, (
        "an ordinary outer error must still retry"
    )
    assert result["final_response"] == "recovered"


# --------------------------------------------------------- terminal state plumbing


def test_failed_turn_closer_honors_a_terminal_drop(monkeypatch):
    """The core closer appends a Hermes boundary row when a turn ends on a user tail;
    a DROP must be exempt — no assistant row may follow the dropped candidate."""
    from agent.conversation_loop import _close_durable_failed_turn

    class _DB:
        def latest_conversation_role(self, session_id):
            return "user"

    def _agent():
        return SimpleNamespace(
            _session_db=_DB(),
            session_id="sess-drop",
            _persist_disabled=False,
            _flush_messages_to_session_db=MagicMock(),
        )

    dropped = {
        "completed": False,
        "messages": [{"role": "user", "content": "task"}],
        "final_output_disposition": "drop",
    }
    _close_durable_failed_turn(_agent(), dropped)
    assert [m["role"] for m in dropped["messages"]] == ["user"], (
        "a DROP must not be followed by any assistant row"
    )

    # Control: without the disposition the closer still closes the turn as before.
    ordinary = {"completed": False, "messages": [{"role": "user", "content": "task"}]}
    _close_durable_failed_turn(_agent(), ordinary)
    assert [m["role"] for m in ordinary["messages"]] == ["user", "assistant"]


def test_stale_disposition_does_not_leak_into_the_next_turn(monkeypatch):
    """Agents are reused across turns: a recorded DROP must be cleared at turn start."""
    from agent.turn_final_output import (
        final_output_disposition,
        reset_final_output_disposition,
    )

    agent = SimpleNamespace(_current_turn_id="turn-1")
    reset_final_output_disposition(agent)
    assert final_output_disposition(agent) is None

    agent._final_output_disposition = {"turn_id": "turn-1", "verdict": "drop", "content": _SECRET}
    assert final_output_disposition(agent, "turn-1") == "drop"
    # Same record, later turn: not this turn's verdict.
    agent._current_turn_id = "turn-2"
    assert final_output_disposition(agent) is None

    reset_final_output_disposition(agent)
    assert agent._final_output_disposition is None


# ============ production-shaped producers (re-review at 0e25ab97f7, 2026-10-05) ============
#
# Both remaining P1s concern PRODUCTION producers that earlier regressions stood in for
# with doubles: `handle_max_iterations()` appends the summary row itself, and the codex
# app-server fast path returns before either generic gate owner.

_CODEX_FINAL = "TOP-SECRET-CODEX-FINAL"
_SUMMARY_TEXT = "MODEL-SUMMARY-SECRET"


def test_production_max_iteration_summary_is_gated_before_its_first_append(monkeypatch, loop_agent):
    """P1 #1: production `handle_max_iterations()` APPENDS the summary row before the
    finalizer's gate sees it, so a late DROP left the row durable and replayable.

    Production producer, only the provider attempt mocked: zero assistant-row durability,
    zero returned candidate, zero next-turn replay, one gate evaluation.
    """
    gated: list = []

    def gate(**kwargs):
        gated.append(kwargs["candidate"]["content"])
        return "drop"

    _install_gate(monkeypatch, gate)
    # The only provider-side mock: every producer statement after this (nudge append,
    # api_messages build, assistant-row append) runs for real.
    monkeypatch.setattr(
        "agent.chat_completion_helpers._chat_summary_attempt",
        lambda agent, api_messages, request_id: (lambda retry_count: _SUMMARY_TEXT),
    )

    agent = loop_agent
    agent._current_turn_id = "turn"
    persist_patch, persisted = _recording_persist(agent)
    with (
        persist_patch,
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = _finalize(
            agent,
            api_call_count=agent.max_iterations,
            messages=[{"role": "user", "content": "do the work"}],
            _turn_exit_reason="unknown",
        )

    # One evaluation, at the producer — before any append.
    assert gated == [_SUMMARY_TEXT]
    # Zero assistant-row durability (the snapshot finalize_turn actually persisted) ...
    assert persisted, "finalize_turn never persisted the transcript"
    assert _SUMMARY_TEXT not in _content_rows(persisted)
    # ... zero returned candidate, zero next-turn replay ...
    assert result["final_response"] is None
    assert result["final_output_disposition"] == "drop"
    assert result["completed"] is False
    assert _SUMMARY_TEXT not in _content_rows([result["messages"]])
    # ... and the production producer really was the one that ran (the unanswered
    # summary nudge it appended is popped again on a terminal verdict).
    assert [row.get("role") for row in result["messages"]] == ["user"]


def _codex_events_recorder(monkeypatch, events: list):
    """Record the gate/persist ordering on the codex fast path."""
    from agent import codex_runtime

    original = codex_runtime._persist_projected_messages

    def wrapper(agent, turn, messages):
        events.append("persist")
        return original(agent, turn, messages)

    monkeypatch.setattr(codex_runtime, "_persist_projected_messages", wrapper)


def _codex_agent_with_recorders(events: list, gated: list, memory: list):
    from tests.agent.test_codex_app_server_integration import _make_codex_agent

    agent = _make_codex_agent()
    agent._sync_external_memory_for_turn = lambda **kwargs: memory.append(
        kwargs.get("final_response")
    )
    agent._spawn_background_review = MagicMock()
    agent._current_turn_id = "turn"
    return agent


@pytest.fixture
def codex_secret_session(monkeypatch):
    """Codex app-server stub whose turn ends on a final candidate plus tool rows."""
    from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult

    def fake_run_turn(self, user_input, **kwargs):
        return TurnResult(
            final_text=_CODEX_FINAL,
            projected_messages=[
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "exec_1", "type": "function",
                                 "function": {"name": "exec_command", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "exec_1", "content": "ok"},
                {"role": "assistant", "content": _CODEX_FINAL},
            ],
            tool_iterations=1,
            interrupted=False,
            error=None,
            turn_id="turn-codex-1",
            thread_id="thread-codex-1",
        )

    monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
    monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "thread-stub-1")


def test_codex_fast_path_drops_the_candidate_before_persist_and_return(monkeypatch, codex_secret_session):
    """P1 #2 (DROP): `_run_conversation_turn` returns the codex result before either
    generic gate owner, so the decision has to happen on that path — before projected
    persistence, before the consumers, before the return."""
    events: list = []
    gated: list = []
    memory: list = []

    def gate(**kwargs):
        gated.append(kwargs["candidate"]["content"])
        events.append("gate")
        return "drop"

    _install_gate(monkeypatch, gate)
    _codex_events_recorder(monkeypatch, events)
    agent = _codex_agent_with_recorders(events, gated, memory)
    persist_patch, _ = _recording_persist(agent)

    with (
        persist_patch,
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do the thing")

    # The decision ran first; persistence ran second, already without the candidate.
    assert events == ["gate", "persist"], "the gate must precede projected persistence"
    assert gated == [_CODEX_FINAL]
    assert result["final_response"] is None
    assert result["completed"] is False
    assert result["final_output_disposition"] == "drop"
    assert _CODEX_FINAL not in _content_rows([result["messages"]])
    # The turn's other projected rows stay durable.
    assert any(
        isinstance(row, dict) and row.get("role") == "tool" for row in result["messages"]
    )
    # Consumers saw the committed candidate: None, never the dropped text.
    assert memory == [None]
    assert agent._spawn_background_review.call_count == 0


def test_codex_fast_path_allows_the_candidate_and_delivers_it(monkeypatch, codex_secret_session):
    """Control: an allowing gate keeps the codex path exactly as it was."""
    events: list = []
    gated: list = []
    memory: list = []

    def gate(**kwargs):
        gated.append(kwargs["candidate"]["content"])
        events.append("gate")
        return "allow"

    _install_gate(monkeypatch, gate)
    _codex_events_recorder(monkeypatch, events)
    agent = _codex_agent_with_recorders(events, gated, memory)
    persist_patch, _ = _recording_persist(agent)

    with (
        persist_patch,
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do the thing")

    assert events == ["gate", "persist"]
    assert gated == [_CODEX_FINAL]
    assert result["final_response"] == _CODEX_FINAL
    assert result["completed"] is True
    assert "final_output_disposition" not in result
    assert _CODEX_FINAL in _content_rows([result["messages"]])
    assert memory == [_CODEX_FINAL]


def test_codex_fast_path_refusal_settles_without_consuming_the_text(monkeypatch, codex_secret_session):
    """P1 #2 (refusal): a fail-closed gate on the codex path settles in place — failed,
    non-retryable, no candidate row, no consumer sees the text, no background review."""
    gated: list = []
    memory: list = []

    def gate(**kwargs):
        gated.append(kwargs["candidate"]["content"])
        raise RuntimeError("commit gate unavailable")

    _install_gate(monkeypatch, gate, failure_mode="closed")
    _codex_events_recorder(monkeypatch, [])
    agent = _codex_agent_with_recorders([], gated, memory)
    persist_patch, _ = _recording_persist(agent)
    messages = [{"role": "user", "content": "do the thing"}]

    from agent.codex_runtime import run_codex_app_server_turn

    with (
        persist_patch,
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = run_codex_app_server_turn(
            agent,
            user_message="do the thing",
            original_user_message="do the thing",
            messages=messages,
            effective_task_id=None,
            should_review_memory=True,
        )

    assert gated == [_CODEX_FINAL]
    assert result["final_output_disposition"] == "refused"
    assert result["failed"] is True
    assert result["failure_reason"] == "llm_stream_middleware_refusal"
    assert result["failure_retryable"] is False
    assert result["final_response"] is None
    assert result["completed"] is False
    assert _CODEX_FINAL not in _content_rows([messages])
    assert memory == [None]
    assert agent._spawn_background_review.call_count == 0, (
        "a refused candidate must never reach the background-review consumer"
    )
