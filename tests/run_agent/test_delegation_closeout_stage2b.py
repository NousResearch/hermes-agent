"""Gate and finalizer tests for task-scoped delegation closeout."""

import json
from types import SimpleNamespace

import run_agent
from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.turn_finalizer import finalize_turn
from tools import async_delegation as ad


def _agent():
    agent = object.__new__(run_agent.AIAgent)
    agent._delegate_depth = 0
    agent._current_work_id = ""
    agent._current_streamed_assistant_text = ""
    agent._stream_needs_break = False
    agent._stream_think_scrubber = None
    agent._stream_context_scrubber = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    agent.interim_assistant_callback = None
    agent.session_id = "session-1"
    agent.model = "model"
    agent.provider = "provider"
    agent.platform = "cli"
    agent._current_turn_id = "turn-1"
    agent._api_call_count = 1
    return agent


def _finalizer_agent():
    agent = _agent()
    agent.max_iterations = 20
    agent.iteration_budget = SimpleNamespace(remaining=10, used=2, max_total=20)
    agent.quiet_mode = True
    agent.base_url = ""
    agent.context_compressor = SimpleNamespace(last_prompt_tokens=0)
    for name in (
        "input", "output", "cache_read", "cache_write", "reasoning",
        "prompt", "completion", "total",
    ):
        setattr(agent, f"session_{name}_tokens", 0)
    agent.session_estimated_cost_usd = 0
    agent.session_cost_status = "unknown"
    agent.session_cost_source = "test"
    agent.request_overrides = {}
    agent._tool_guardrail_halt_decision = None
    agent._interrupt_message = None
    agent._response_was_previewed = False
    agent._skill_nudge_interval = 0
    agent._iters_since_skill = 0
    agent.valid_tool_names = []
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    agent._persisted = []
    agent._admitted = []
    agent._save_trajectory = lambda *_a, **_k: None
    agent._cleanup_task_resources = lambda *_a, **_k: None
    agent._drop_trailing_empty_response_scaffolding = lambda *_a, **_k: None
    agent._apply_persist_user_message_override = lambda *_a, **_k: None
    agent._persist_session = lambda messages, _history: agent._persisted.append(
        [dict(message) for message in messages]
    )
    agent._file_mutation_verifier_enabled = lambda: False
    agent._turn_completion_explainer_enabled = lambda: False
    agent._drain_pending_steer = lambda: None
    agent.clear_interrupt = lambda: None
    agent._sync_external_memory_for_turn = lambda **_k: None
    agent._discard_conversational_response = lambda: None
    agent._admit_conversational_response = lambda: agent._admitted.append("admitted")
    agent._handle_max_iterations = lambda *_a: (_ for _ in ()).throw(
        AssertionError("budget fallback must not run")
    )
    return agent


def _finalize(agent, messages, **kwargs):
    return finalize_turn(
        agent,
        final_response=kwargs.pop("final_response", None),
        api_call_count=kwargs.pop("api_call_count", 2),
        interrupted=kwargs.pop("interrupted", False),
        failed=kwargs.pop("failed", False),
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id=kwargs.pop("turn_id", "turn-1"),
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason=kwargs.pop("exit_reason", "text_response(stop)"),
        **kwargs,
    )


def _group(work_id):
    with ad._transaction() as conn:
        conn.row_factory = __import__("sqlite3").Row
        row = conn.execute(
            "SELECT * FROM async_delegation_work_groups WHERE work_id=?",
            (work_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def test_text_then_later_tool_delta_leaks_to_no_observer(monkeypatch):
    agent = _agent()
    display, tts, interim, hooks = [], [], [], []
    agent.stream_delta_callback = display.append
    agent._stream_callback = tts.append
    agent.interim_assistant_callback = lambda text, **kw: interim.append(text)
    monkeypatch.setattr(
        "agent.plugin_stream_hooks.enqueue_plugin_stream_hook",
        lambda name, **payload: hooks.append((name, payload)),
    )
    monkeypatch.setattr(agent, "_conversational_admission_required", lambda: True)

    agent._reset_stream_delivery_tracking()
    agent._fire_stream_delta("I am done")
    agent._fire_streamed_codex_commentary("Commentary")
    agent._emit_stream_end(final_text="I am done", finished=True, error=None)
    assert display == tts == interim == []
    assert hooks == []

    agent._settle_conversational_response(has_tool_calls=True)
    agent._close_stream_display_segment()
    assert display == tts == interim == []
    assert hooks == []


def test_no_tool_response_admits_exactly_once(monkeypatch):
    agent = _agent()
    display, tts = [], []
    agent.stream_delta_callback = display.append
    agent._stream_callback = tts.append
    monkeypatch.setattr(agent, "_conversational_admission_required", lambda: True)

    agent._reset_stream_delivery_tracking()
    agent._fire_stream_delta("hel")
    agent._fire_stream_delta("lo")
    agent._settle_conversational_response(has_tool_calls=False)
    assert display == []
    assert tts == []
    agent._admit_conversational_response()
    agent._settle_conversational_response(has_tool_calls=False)

    assert display == ["hel", "lo"]
    assert tts == ["hel", "lo"]


def test_tool_iteration_discards_prose_then_next_final_admits(monkeypatch):
    agent = _agent()
    delivered = []
    agent.stream_delta_callback = delivered.append
    monkeypatch.setattr(agent, "_conversational_admission_required", lambda: True)

    agent._reset_stream_delivery_tracking()
    agent._fire_stream_delta("premature prose")
    agent._settle_conversational_response(has_tool_calls=True)
    agent._reset_stream_delivery_tracking()
    agent._fire_stream_delta("actual final")
    agent._settle_conversational_response(has_tool_calls=False)
    assert delivered == []
    agent._admit_conversational_response()

    assert delivered == ["actual final"]


def test_rejected_no_tool_candidate_cannot_leak_before_later_delegation(monkeypatch):
    agent = _agent()
    delivered = []
    agent.stream_delta_callback = delivered.append
    monkeypatch.setattr(agent, "_conversational_admission_required", lambda: True)

    agent._reset_stream_delivery_tracking()
    agent._fire_stream_delta("premature final")
    agent._settle_conversational_response(has_tool_calls=False)
    assert delivered == []

    # A verification continuation supersedes that candidate and the next
    # complete response contains a tool call (which could be delegate_task).
    agent._reset_stream_delivery_tracking()
    agent._fire_stream_delta("delegating now")
    agent._settle_conversational_response(has_tool_calls=True)
    assert delivered == []


def test_inherited_work_discards_no_tool_candidate(monkeypatch):
    agent = _agent()
    agent._current_work_id = "work-1"
    delivered = []
    agent.stream_delta_callback = delivered.append
    monkeypatch.setattr(agent, "_conversational_admission_required", lambda: True)

    agent._reset_stream_delivery_tracking()
    agent._fire_stream_delta("candidate")
    agent._settle_conversational_response(has_tool_calls=False)

    assert delivered == []


def test_feature_off_keeps_immediate_streaming(monkeypatch):
    agent = _agent()
    delivered = []
    agent.stream_delta_callback = delivered.append
    monkeypatch.setattr(agent, "_conversational_admission_required", lambda: False)

    agent._reset_stream_delivery_tracking()
    agent._fire_stream_delta("legacy")

    assert delivered == ["legacy"]


def test_waiting_turn_seals_persists_provider_valid_hidden_tail(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_k: [])
    ad._reset_for_tests()
    assert ad.register_work_group_member(
        work_id="work-1", owner_turn_id="turn-1", delegation_id="child-1",
        feature_config={"delegation": {"task_scoped_closeout": True}},
    )
    agent = _finalizer_agent()
    agent._current_work_id = "work-1"
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "d1"}]},
        {"role": "tool", "tool_call_id": "d1", "content": "dispatched"},
    ]

    result = _finalize(agent, messages)

    assert result["waiting_on_delegates"] is True
    assert result["final_response"] is None
    assert result["completed"] is False
    assert result["turn_exit_reason"] == "delegation_waiting"
    assert agent._persisted[-1][-1]["role"] == "assistant"
    assert agent._persisted[-1][-1]["display_kind"] == "delegation_waiting"
    assert agent._persisted[-1][-1]["display_metadata"]["hidden"] is True
    assert _group("work-1")["state"] == "sealed"


def test_closeout_persists_then_exact_identity_cas_then_admits(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_k: [])
    ad._reset_for_tests()
    assert ad.register_work_group_member(
        work_id="work-1", owner_turn_id="owner", delegation_id="child-1",
        feature_config={"delegation": {"task_scoped_closeout": True}},
    )
    ad.persist_group_member_completion(
        "child-1", {"status": "completed"}, {"status": "completed", "summary": "ok"}
    )
    assert ad.seal_work_group("work-1", "owner")
    claim = ad.claim_ready_work_group("work-1", "test")
    envelope = claim["envelope"]
    assert ad.bind_work_group_closeout_turn(
        "work-1", envelope["delivery_id"], claim["claim_id"], "turn-1"
    )
    agent = _finalizer_agent()
    agent._current_work_id = "work-1"
    agent._current_work_generation = 0
    agent._current_work_delivery_id = envelope["delivery_id"]
    agent._current_work_claim_id = claim["claim_id"]
    order = []
    agent._persist_session = lambda *_a: order.append("persist")
    agent._admit_conversational_response = lambda: order.append("admit")

    result = _finalize(
        agent, [{"role": "user", "content": "close"}],
        final_response="All work reconciled.", closeout_terminal_candidate=True,
    )

    assert result["final_response"] == "All work reconciled."
    assert order == ["persist", "persist", "admit"]
    assert "display_kind" not in result["messages"][-1]
    group = _group("work-1")
    assert group["state"] == "closed"
    assert group["terminal_disposition"] == "success"


def test_closeout_rewrites_incrementally_persisted_row_for_hide_and_reveal(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_k: [])
    ad._reset_for_tests()
    assert ad.register_work_group_member(
        work_id="work-1",
        owner_turn_id="owner",
        delegation_id="child-1",
        feature_config={"delegation": {"task_scoped_closeout": True}},
    )
    ad.persist_group_member_completion(
        "child-1", {"status": "completed"}, {"status": "completed"}
    )
    assert ad.seal_work_group("work-1", "owner")
    claim = ad.claim_ready_work_group("work-1", "test")
    envelope = claim["envelope"]
    assert ad.bind_work_group_closeout_turn(
        "work-1", envelope["delivery_id"], claim["claim_id"], "turn-1"
    )
    agent = _finalizer_agent()
    agent._current_work_id = "work-1"
    agent._current_work_generation = 0
    agent._current_work_delivery_id = envelope["delivery_id"]
    agent._current_work_claim_id = claim["claim_id"]
    snapshots = []

    def persist(messages, _history):
        snapshots.append([dict(message) for message in messages])
        for message in messages:
            message[_DB_PERSISTED_MARKER] = True
        agent._db_flush_scan_prefix = tuple(messages)

    agent._persist_session = persist
    candidate = {
        "role": "assistant",
        "content": "final",
        _DB_PERSISTED_MARKER: True,
    }
    agent._db_flush_scan_prefix = ({"role": "user", "content": "close"}, candidate)

    result = _finalize(
        agent,
        [{"role": "user", "content": "close"}, candidate],
        final_response="final",
        closeout_terminal_candidate=True,
    )

    assert snapshots[0][-1]["display_kind"] == "delegation_closeout_provisional"
    assert _DB_PERSISTED_MARKER not in snapshots[0][-1]
    assert "display_kind" not in snapshots[1][-1]
    assert _DB_PERSISTED_MARKER not in snapshots[1][-1]
    assert result["final_response"] == "final"


def test_closeout_persist_failure_keeps_closing_and_never_admits(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_k: [])
    ad._reset_for_tests()
    assert ad.register_work_group_member(
        work_id="work-1", owner_turn_id="owner", delegation_id="child-1",
        feature_config={"delegation": {"task_scoped_closeout": True}},
    )
    ad.persist_group_member_completion("child-1", {"status": "completed"}, {"status": "completed"})
    assert ad.seal_work_group("work-1", "owner")
    claim = ad.claim_ready_work_group("work-1", "test")
    envelope = claim["envelope"]
    assert ad.bind_work_group_closeout_turn(
        "work-1", envelope["delivery_id"], claim["claim_id"], "turn-1"
    )
    agent = _finalizer_agent()
    agent._current_work_id = "work-1"
    agent._current_work_delivery_id = envelope["delivery_id"]
    agent._current_work_claim_id = claim["claim_id"]
    agent._persist_session = lambda *_a: (_ for _ in ()).throw(OSError("disk full"))

    result = _finalize(
        agent, [{"role": "user", "content": "close"}],
        final_response="provisional", closeout_terminal_candidate=True,
    )

    assert result["final_response"] is None
    assert result["failed"] is True
    assert agent._admitted == []
    assert result["messages"][-1]["display_kind"] == "delegation_closeout_provisional"
    assert result["messages"][-1]["display_metadata"]["hidden"] is True
    assert _group("work-1")["state"] == "closing"


def test_closeout_cas_failure_keeps_hidden_candidate_recoverable(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_k: [])
    monkeypatch.setattr(ad, "close_work_group", lambda *_a, **_k: False)
    agent = _finalizer_agent()
    agent._current_work_id = "work-1"
    agent._current_work_generation = 7
    agent._current_work_delivery_id = "delivery-7"
    agent._current_work_claim_id = "claim-7"

    result = _finalize(
        agent, [{"role": "user", "content": "close"}],
        final_response="must stay hidden", closeout_terminal_candidate=True,
    )

    assert result["final_response"] is None
    assert result["failed"] is True
    assert agent._admitted == []
    hidden = agent._persisted[-1][-1]
    assert hidden["display_kind"] == "delegation_closeout_provisional"
    assert hidden["display_metadata"] == {
        "hidden": True, "work_id": "work-1", "generation": 7,
        "delivery_id": "delivery-7", "claim_id": "claim-7", "turn_id": "turn-1",
    }


def test_crash_window_replay_reuses_one_canonical_provisional(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_k: [])
    ad._reset_for_tests()
    assert ad.register_work_group_member(
        work_id="work-1", owner_turn_id="owner", delegation_id="child-1",
        feature_config={"delegation": {"task_scoped_closeout": True}},
    )
    ad.persist_group_member_completion(
        "child-1", {"status": "completed"}, {"status": "completed"}
    )
    assert ad.seal_work_group("work-1", "owner")
    claim = ad.claim_ready_work_group("work-1", "first-process")
    envelope = claim["envelope"]
    delivery = envelope["delivery_id"]
    assert ad.bind_work_group_closeout_turn(
        "work-1", delivery, claim["claim_id"], "dead-turn"
    )
    # Exact crash window: assistant provisional committed, group still closing.
    with ad._transaction() as conn:
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT, "
            "display_kind TEXT, display_metadata TEXT)"
        )
        conn.execute(
            "INSERT INTO messages VALUES (1, ?, ?, ?)",
            (
                "canonical terminal",
                "delegation_closeout_provisional",
                json.dumps({
                    "hidden": True, "work_id": "work-1",
                    "delivery_id": delivery,
                }),
            ),
        )
        conn.execute(
            "UPDATE async_delegation_work_groups SET closeout_owner_pid=2147483647, "
            "closeout_owner_started_at=1"
        )
    recovered = ad.reclaim_stale_work_group_claim("work-1", "replacement")
    assert recovered is not None
    assert ad.bind_work_group_closeout_turn(
        "work-1", delivery, recovered["claim_id"], "turn-1"
    )

    agent = _finalizer_agent()
    agent._current_work_id = "work-1"
    agent._current_work_generation = 0
    agent._current_work_delivery_id = delivery
    agent._current_work_claim_id = recovered["claim_id"]

    def persist(messages, _history):
        candidate = messages[-1]
        if candidate.get(_DB_PERSISTED_MARKER):
            return
        with ad._transaction() as conn:
            values = (
                candidate.get("content"),
                candidate.get("display_kind"),
                json.dumps(candidate.get("display_metadata")),
            )
            if candidate.get("_row_id"):
                conn.execute(
                    "UPDATE messages SET content=?, display_kind=?, "
                    "display_metadata=? WHERE id=?",
                    (*values, candidate["_row_id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO messages (content, display_kind, display_metadata) "
                    "VALUES (?, ?, ?)",
                    values,
                )
        candidate[_DB_PERSISTED_MARKER] = True

    agent._persist_session = persist
    result = _finalize(
        agent, [{"role": "user", "content": "replay"}],
        final_response="different replay candidate", closeout_terminal_candidate=True,
    )

    assert result["final_response"] == "canonical terminal"
    with ad._transaction() as conn:
        rows = conn.execute(
            "SELECT content, display_kind FROM messages ORDER BY id"
        ).fetchall()
    assert rows == [("canonical terminal", None)]


def test_failed_parent_is_sealed_for_diagnostic_closeout(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_k: [])
    ad._reset_for_tests()
    assert ad.register_work_group_member(
        work_id="work-1", owner_turn_id="turn-1", delegation_id="child-1",
        feature_config={"delegation": {"task_scoped_closeout": True}},
    )
    agent = _finalizer_agent()
    agent._current_work_id = "work-1"

    result = _finalize(
        agent, [{"role": "user", "content": "task"}],
        failed=True, exit_reason="provider_failure",
    )

    assert result["failed"] is True
    assert result["waiting_on_delegates"] is True
    assert result["final_response"] is None
    assert result["turn_exit_reason"] == "delegation_waiting_after_owner_failure"
    assert result["delegation_diagnostic"]["code"] == "owner_failed_group_sealed"
    group = _group("work-1")
    assert group["state"] == "sealed"
    assert "provider_failure" in group["terminal_diagnostics"]


def test_owner_mismatch_fails_closed_with_explicit_diagnostic(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_k: [])
    ad._reset_for_tests()
    assert ad.register_work_group_member(
        work_id="work-1", owner_turn_id="real-owner", delegation_id="child-1",
        feature_config={"delegation": {"task_scoped_closeout": True}},
    )
    agent = _finalizer_agent()
    agent._current_work_id = "work-1"

    result = _finalize(agent, [{"role": "user", "content": "task"}])

    assert result["failed"] is True
    assert result["final_response"] is None
    assert result["waiting_on_delegates"] is False
    assert result["delegation_diagnostic"]["code"] == "owner_turn_mismatch"
    assert _group("work-1")["state"] == "open"


def test_shared_admission_seam_releases_display_tts_interim_and_plugins(monkeypatch):
    agent = _agent()
    display, tts, interim, hooks = [], [], [], []
    agent.stream_delta_callback = display.append
    agent._stream_callback = tts.append
    agent.interim_assistant_callback = lambda text, **kw: interim.append((text, kw))
    agent._delivered_interim_texts = set()
    monkeypatch.setattr(agent, "_conversational_admission_required", lambda: True)
    monkeypatch.setattr(
        "agent.plugin_stream_hooks.enqueue_plugin_stream_hook",
        lambda name, **payload: hooks.append((name, payload)),
    )

    agent._reset_stream_delivery_tracking()
    # These are the shared calls used by Chat Completions (including its old
    # direct callback bypass), Anthropic/Bedrock normalization, Responses
    # commentary/interim delivery, display/TTS, and stream-end plugins.
    agent._fire_suppressed_tool_text("direct")
    agent._fire_stream_delta("delta")
    agent._fire_streamed_codex_commentary("commentary")
    agent._emit_interim_assistant_message({"role": "assistant", "content": "interim"})
    agent._emit_stream_end(final_text="directdelta", finished=True, error=None)
    agent._settle_conversational_response(has_tool_calls=False)
    assert display == tts == interim == hooks == []

    agent._admit_conversational_response()

    assert display == ["direct", "delta"]
    assert tts == ["direct", "delta"]
    assert [item[0] for item in interim] == ["commentary", "interim"]
    assert [name for name, _ in hooks] == [
        "on_stream_delta", "on_stream_delta", "on_interim_message", "on_stream_end"
    ]


def test_replayed_closeout_with_lost_claim_is_silent(monkeypatch):
    agent = _agent()
    monkeypatch.setattr(ad, "bind_work_group_closeout_turn", lambda *_a: False)

    result = run_agent.AIAgent.run_conversation(
        agent,
        "trusted aggregate",
        conversation_history=[{"role": "user", "content": "original"}],
        task_id="session-1",
        origin_work_id="work-1",
        work_generation=2,
        work_delivery_id="delivery-2",
        work_claim_id="stale-claim",
    )

    assert result["duplicate_closeout"] is True
    assert result["final_response"] == ""
    assert result["messages"] == [{"role": "user", "content": "original"}]
    assert agent._current_work_id == ""
