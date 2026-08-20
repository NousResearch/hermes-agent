from types import SimpleNamespace
from typing import Any

import pytest

from agent.turn_finalizer import finalize_turn


class FakeAgent:
    def __init__(self):
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = True
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages: list[dict[str, Any]] | None = None
        self._persist_user_message_idx: int | None = None
        self._persist_user_message_override: Any = None
        self._persist_user_message_timestamp: float | None = None

    def _handle_max_iterations(self, messages, api_call_count):
        raise AssertionError("not expected")

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        # Capture the durable write before finalization restores API-local
        # guidance to the returned/live transcript.
        self.persisted_messages = [dict(message) for message in messages]

    def _apply_persist_user_message_override(self, messages):
        idx = self._persist_user_message_idx
        override = self._persist_user_message_override
        if idx is not None and override is not None:
            messages[idx]["content"] = override

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def test_finalizer_restores_clean_api_local_text_before_return(monkeypatch):
    """One-shot CLI notes do not replay through same-process history."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "[MODEL SWITCH NOTE]\n\nclean prompt"},
        {"role": "assistant", "content": "Done."},
    ]
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "clean prompt"
    agent._persist_user_message_timestamp = None

    result = finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="[MODEL SWITCH NOTE]\n\nclean prompt",
        original_user_message="clean prompt",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert agent.persisted_messages is not None
    assert agent.persisted_messages[0]["content"] == "clean prompt"
    assert result["messages"][0]["content"] == "clean prompt"


def test_finalizer_stamps_deferred_adoption_on_closing_assistant(monkeypatch):
    """The durable adoption identity rides the final assistant transaction."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    agent._deferred_notification_ids = (
        "async_delegation:deleg-crash-window",
    )
    messages = [
        {"role": "user", "content": "Next request"},
        {"role": "assistant", "content": "Done."},
    ]

    result = finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="Next request",
        original_user_message="Next request",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    expected = ["async_delegation:deleg-crash-window"]
    assert agent.persisted_messages[-1]["_deferred_notification_ids"] == expected
    assert "_deferred_notification_ids" not in result["messages"][-1]


def test_empty_success_persists_adoption_and_acks_without_retry_or_replay(
    tmp_path, monkeypatch
):
    """An empty successful reply still closes and consumes its durable batch."""
    from hermes_state import SessionDB
    from tools import async_delegation
    from tui_gateway import server

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "state.db"
    event_id = "async_delegation:deleg-empty-success"
    db = SessionDB(db_path)
    db.create_session("sess-test", source="tui")
    assert async_delegation.persist_deferred_notification(
        "sess-test",
        event_id,
        "durable empty-turn result",
        {
            "type": "async_delegation",
            "delegation_id": "deleg-empty-success",
            "session_key": "sess-test",
        },
        db_path=db_path,
    )

    agent = FakeAgent()
    agent._deferred_notification_ids = (event_id,)

    def persist(messages, _conversation_history):
        for message in messages:
            db.append_message(
                "sess-test",
                message.get("role", "unknown"),
                message.get("content"),
                deferred_notification_ids=message.get(
                    "_deferred_notification_ids"
                ),
            )

    agent._persist_session = persist
    result = finalize_turn(
        agent,
        final_response="",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "Next request"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="Next request",
        original_user_message="Next request",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["completed"] is True
    closing = result["messages"][-1]
    assert isinstance(closing["timestamp"], float)  # upstream stamps the closing row
    assert {k: v for k, v in closing.items() if k != "timestamp"} == {
        "role": "assistant",
        "content": "",
    }
    assert db._conn.execute(
        "SELECT COUNT(*) FROM deferred_notification_adoptions WHERE event_id=?",
        (event_id,),
    ).fetchone()[0] == 1

    retry_threads = []

    class _RecordingThread:
        def __init__(self, *args, name=None, **kwargs):
            retry_threads.append(name)

        def start(self):
            return None

    monkeypatch.setattr(server.threading, "Thread", _RecordingThread)
    session = {
        "session_key": "sess-test",
        "profile_home": str(tmp_path),
    }
    assert server._ack_consumed_deferred_notifications(session, {event_id}) is True
    assert retry_threads == []
    assert async_delegation.load_deferred_notifications(
        "sess-test", db_path=db_path
    ) == []
    assert db._conn.execute(
        "SELECT COUNT(*) FROM deferred_notification_adoptions WHERE event_id=?",
        (event_id,),
    ).fetchone()[0] == 0

    history = db.get_messages_as_conversation("sess-test")
    db.close()
    assert [(message["role"], message["content"]) for message in history] == [
        ("user", "Next request"),
        ("assistant", ""),
    ]
    restarted = server._deferred_session_record(
        "sess-test",
        cols=80,
        cwd=".",
        history=history,
        lease=None,
        profile_home=tmp_path,
    )
    assert restarted["deferred_notification_texts"] == []
    assert restarted["deferred_notification_event_ids"] == set()
    assert restarted["defer_notifications_until_user"] is False


def _persist_messages_to_session_db(db, session_id, messages):
    for message in messages:
        db.append_message(
            session_id,
            message.get("role", "unknown"),
            message.get("content"),
            deferred_notification_ids=message.get("_deferred_notification_ids"),
        )


def _adopted_closing_message(db, event_id):
    row = db._conn.execute(
        """SELECT messages.role, messages.content
             FROM deferred_notification_adoptions AS adoption
             JOIN messages ON messages.id = adoption.message_id
            WHERE adoption.event_id=?""",
        (event_id,),
    ).fetchone()
    return tuple(row) if row is not None else None


def _run_iteration_limit_with_durable_notification(
    tmp_path,
    monkeypatch,
    *,
    delegation_id,
    summarize,
):
    from hermes_state import SessionDB
    from tools import async_delegation

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "state.db"
    event_id = f"async_delegation:{delegation_id}"
    db = SessionDB(db_path)
    db.create_session("sess-test", source="tui")
    assert async_delegation.persist_deferred_notification(
        "sess-test",
        event_id,
        "durable iteration-limit result",
        {
            "type": "async_delegation",
            "delegation_id": delegation_id,
            "session_key": "sess-test",
        },
        db_path=db_path,
    )

    agent = FakeAgent()
    agent.max_iterations = 1
    agent.iteration_budget = SimpleNamespace(remaining=0, used=1, max_total=1)
    agent._deferred_notification_ids = (event_id,)
    agent._handle_max_iterations = summarize
    agent._persist_session = lambda messages, _history: _persist_messages_to_session_db(
        db, "sess-test", messages
    )

    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "Next request"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="Next request",
        original_user_message="Next request",
        _should_review_memory=False,
        _turn_exit_reason="unknown",
    )
    return db, event_id, result


def test_iteration_limit_summary_persists_durable_closing_adoption(
    tmp_path, monkeypatch
):
    """A summary is accepted and adopted even though completed stays false."""

    def summarize(messages, _api_call_count):
        report = "Summary of completed work."
        messages.append({"role": "assistant", "content": report})
        return report

    db, event_id, result = _run_iteration_limit_with_durable_notification(
        tmp_path,
        monkeypatch,
        delegation_id="deleg-summary-success",
        summarize=summarize,
    )
    try:
        assert result["completed"] is False
        assert result["final_response"] == "Summary of completed work."
        assert result["messages"][-1] == {
            "role": "assistant",
            "content": "Summary of completed work.",
        }
        assert _adopted_closing_message(db, event_id) == (
            "assistant",
            "Summary of completed work.",
        )
    finally:
        db.close()


def test_iteration_limit_summary_exception_fallback_closes_and_adopts(
    tmp_path, monkeypatch
):
    """The handler's visible exception fallback is durable without a helper row."""
    fallback = "I reached the maximum iterations (1) but couldn't summarize. Error: API down"

    def summarize(messages, _api_call_count):
        messages.append({"role": "user", "content": "Summarize without tools."})
        return fallback

    db, event_id, result = _run_iteration_limit_with_durable_notification(
        tmp_path,
        monkeypatch,
        delegation_id="deleg-summary-fallback",
        summarize=summarize,
    )
    try:
        assert result["completed"] is False
        assert result["final_response"] == fallback
        closing = result["messages"][-1]
        assert isinstance(closing["timestamp"], float)  # upstream stamps the closing row
        assert {k: v for k, v in closing.items() if k != "timestamp"} == {
            "role": "assistant",
            "content": fallback,
        }
        assert _adopted_closing_message(db, event_id) == ("assistant", fallback)
    finally:
        db.close()


@pytest.mark.parametrize(
    ("interrupted", "failed", "exit_reason"),
    [
        (True, False, "interrupted_by_user"),
        (False, True, "provider_failure"),
    ],
)
def test_failed_or_interrupted_response_never_adopts_deferred_notifications(
    tmp_path, monkeypatch, interrupted, failed, exit_reason
):
    """Visible partial/error text is not proof that deferred work was accepted."""
    from hermes_state import SessionDB
    from tools import async_delegation

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "state.db"
    event_id = f"async_delegation:deleg-{exit_reason}"
    db = SessionDB(db_path)
    db.create_session("sess-test", source="tui")
    assert async_delegation.persist_deferred_notification(
        "sess-test",
        event_id,
        "must remain pending",
        {
            "type": "async_delegation",
            "delegation_id": f"deleg-{exit_reason}",
            "session_key": "sess-test",
        },
        db_path=db_path,
    )
    agent = FakeAgent()
    agent._deferred_notification_ids = (event_id,)
    agent._persist_session = lambda messages, _history: _persist_messages_to_session_db(
        db, "sess-test", messages
    )

    try:
        finalize_turn(
            agent,
            final_response="Visible partial or error report.",
            api_call_count=1,
            interrupted=interrupted,
            failed=failed,
            messages=[
                {"role": "user", "content": "Next request"},
                {"role": "assistant", "content": "Visible partial or error report."},
            ],
            conversation_history=[],
            effective_task_id="task",
            turn_id="turn",
            user_message="Next request",
            original_user_message="Next request",
            _should_review_memory=False,
            _turn_exit_reason=exit_reason,
        )

        assert _adopted_closing_message(db, event_id) is None
        assert async_delegation.load_deferred_notifications(
            "sess-test", db_path=db_path
        )
    finally:
        db.close()


def test_finalizer_restores_clean_api_local_multimodal_before_return(monkeypatch):
    """A queued note does not remain in the next-turn native image payload."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    clean_content = [
        {"type": "text", "text": "Describe the image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    api_content = [
        {"type": "text", "text": "[MODEL SWITCH NOTE]\n\nDescribe the image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    messages = [
        {"role": "user", "content": api_content},
        {"role": "assistant", "content": "Done."},
    ]
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = clean_content
    agent._persist_user_message_timestamp = None




def test_final_response_closes_tool_tail_before_persistence(monkeypatch):
    """A recovered/previewed final response must be durable in session history.

    Regression for turns where the caller receives a non-empty final_response,
    but the message transcript still ends at a tool result. If persisted that
    way, the next turn reloads a stale/malformed history and can appear to loop
    because the assistant's visible final answer is missing from durable state.
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "I'll check.",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "terminal", "content": "ok"},
    ]

    result = finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="do it",
        original_user_message="do it",
        _should_review_memory=False,
        _turn_exit_reason="fallback_prior_turn_content",
    )

    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"] == "Done."
    assert isinstance(result["messages"][-1]["timestamp"], float)
    assert agent.persisted_messages is not None
    assert agent.persisted_messages[-1] == result["messages"][-1]


def test_fallback_timestamp_survives_delayed_sqlite_persistence(
    monkeypatch, tmp_path
):
    """The durable row records message creation, not the later DB flush."""
    from hermes_state import SessionDB

    created_at = 1_781_976_577.25
    persisted_at = created_at + 600
    monkeypatch.setattr("agent.message_metadata.wall_time", lambda: created_at)
    monkeypatch.setattr("hermes_state.time.time", lambda: persisted_at)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess-test", source="cli")
    agent = FakeAgent()

    def persist_to_sqlite(messages, _conversation_history):
        db.replace_messages(agent.session_id, messages)
        agent.persisted_messages = db.get_messages_as_conversation(agent.session_id)

    agent._persist_session = persist_to_sqlite
    messages = [
        {"role": "user", "content": "do it", "timestamp": created_at - 1},
        {"role": "tool", "content": "ok", "tool_call_id": "call-1"},
    ]

    finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="do it",
        original_user_message="do it",
        _should_review_memory=False,
        _turn_exit_reason="fallback_prior_turn_content",
    )

    assert agent.persisted_messages[-1]["timestamp"] == created_at
    assert agent.persisted_messages[-1]["timestamp"] != persisted_at


def test_final_response_fills_pure_tool_call_tail(monkeypatch):
    """A tail assistant row that is a *pure tool-call turn* carries no answer.

    The role check alone ("tail is assistant ⇒ nothing to do") leaves the
    #43849/#44100 invariant unmet when the tail is ``assistant(tool_calls)``
    with no text of its own: the caller and the gateway already delivered
    ``final_response``, but it never reaches the transcript. The next turn then
    replays the user backlog and the model re-answers it — the exact symptom
    that block exists to prevent.
    """
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}
            ],
        },
    ]

    result = finalize_turn(
        agent,
        final_response="Here is your answer.",
        api_call_count=3,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="t",
        turn_id="tid",
        user_message="q",
        original_user_message="q",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    persisted = agent.persisted_messages
    assert any(
        m.get("role") == "assistant" and m.get("content") == result["final_response"]
        for m in persisted
    ), "delivered final_response never reached the durable transcript"
    # Filled in place — no assistant→assistant pair, tool_calls preserved.
    assert persisted[-1]["content"] == "Here is your answer."
    assert persisted[-1]["tool_calls"]
    assert sum(1 for m in persisted if m.get("role") == "assistant") == 1






def test_final_response_fill_invalidates_flush_scan_cursor():
    """The fill's marker pop must invalidate the bounded flush-scan cursor.

    The cursor (run_agent.py) skips the identity-matched prefix of its
    previous snapshot assuming no live dict loses ``_db_persisted`` in place
    — the fill is the one path that pops it. Without invalidation, the
    turn-end flush skips the filled row as 'already stamped' and the
    delivered answer never reaches state.db (the #43849 class resurfacing).
    """
    agent = FakeAgent()
    agent._db_flush_scan_prefix = ["prior-snapshot"]
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}
            ],
            "_db_persisted": True,
        },
    ]

    finalize_turn(
        agent,
        final_response="Here is your answer.",
        api_call_count=3,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="t",
        turn_id="tid",
        user_message="q",
        original_user_message="q",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    assert agent._db_flush_scan_prefix is None
