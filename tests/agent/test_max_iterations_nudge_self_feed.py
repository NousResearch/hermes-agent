"""Regression: max-iteration summary nudge must not durable-persist or replay."""

from agent.context_compressor import MAX_ITERATIONS_SUMMARY_REQUEST
from agent.session_persistence import _is_ephemeral_scaffolding
from agent.turn_finalizer import finalize_turn
from hermes_state import SessionDB, _strip_iteration_limit_summary_nudges
from run_agent import AIAgent
from types import SimpleNamespace
from unittest.mock import MagicMock


class _CapturingSessionDB:
    def __init__(self):
        self.rows = []

    def append_messages_batch(self, session_id, messages, **kwargs):
        for m in messages:
            self.rows.append({"role": m.get("role"), "content": m.get("content")})
        return list(range(1, len(self.rows) + 1))


def _agent_with_capturing_db():
    agent = AIAgent.__new__(AIAgent)
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._session_db = _CapturingSessionDB()
    agent._session_db_created = True
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._db_flush_scan_prefix = []
    agent.session_id = "sess-max-iter-nudge"
    agent._persist_disabled = False
    return agent


def _synthetic_nudge_message(*, flagged: bool = True):
    msg = {
        "role": "user",
        "content": MAX_ITERATIONS_SUMMARY_REQUEST,
    }
    if flagged:
        msg["_max_iterations_summary_synthetic"] = True
    return msg


def test_flush_skips_flagged_max_iterations_nudge():
    agent = _agent_with_capturing_db()
    messages = [
        {"role": "user", "content": "human task"},
        _synthetic_nudge_message(flagged=True),
        {"role": "assistant", "content": "summary text"},
    ]
    agent._flush_messages_to_session_db(messages, conversation_history=[])
    assert agent._session_db.rows == [
        {"role": "user", "content": "human task"},
        {"role": "assistant", "content": "summary text"},
    ]


def test_flush_keeps_real_user_when_text_matches_nudge_copy(tmp_path):
    """A-3: content alone must never classify a human turn as ephemeral."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "sess-real-user-nudge-text"
    db.create_session(sid, source="cli")
    human = {"role": "user", "content": MAX_ITERATIONS_SUMMARY_REQUEST}
    assert not _is_ephemeral_scaffolding(human)
    db.append_messages_batch(sid, [human])
    row = db._read_one(
        "SELECT content FROM messages WHERE session_id = ? AND role = 'user' AND active = 1",
        (sid,),
    )
    assert row is not None
    assert db._decode_content(row[0]) == MAX_ITERATIONS_SUMMARY_REQUEST


def test_replay_strip_removes_persisted_legacy_nudges():
    polluted = [
        {"role": "user", "content": "human task"},
        {"role": "assistant", "content": "partial"},
        {"role": "user", "content": MAX_ITERATIONS_SUMMARY_REQUEST},
        {"role": "assistant", "content": "summary"},
    ]
    cleaned = _strip_iteration_limit_summary_nudges(polluted)
    assert cleaned == [
        {"role": "user", "content": "human task"},
        {"role": "assistant", "content": "partial"},
        {"role": "assistant", "content": "summary"},
    ]


def test_replay_falsification_polluted_db_row_is_stripped(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "legacy-polluted"
    db.create_session(sid, source="cli")
    db.append_message(sid, "user", "human task")
    db.append_message(sid, "user", MAX_ITERATIONS_SUMMARY_REQUEST)
    db.append_message(sid, "assistant", "summary")
    loaded = db.get_messages_as_conversation(sid)
    assert all(
        m.get("content") != MAX_ITERATIONS_SUMMARY_REQUEST for m in loaded
    )


def test_finalize_turn_strips_nudge_before_persist():
    agent = MagicMock()
    agent.quiet_mode = True
    agent.max_iterations = 60
    agent.iteration_budget = SimpleNamespace(remaining=0, used=60, max_total=60)
    agent.model = "m"
    agent.provider = "p"
    agent.base_url = ""
    agent.session_id = "s"
    agent.context_compressor = SimpleNamespace(last_prompt_tokens=0)
    agent._tool_guardrail_halt_decision = None
    agent._interrupt_message = None
    agent._response_was_previewed = False
    agent._turn_received_provider_response = True
    agent.valid_tool_names = []
    agent._handle_max_iterations = MagicMock(return_value="done")
    agent._emit_status = MagicMock()
    agent._safe_print = MagicMock()
    agent._save_trajectory = MagicMock()
    agent._cleanup_task_resources = MagicMock()
    agent._drop_trailing_empty_response_scaffolding = MagicMock()
    agent._file_mutation_verifier_enabled = MagicMock(return_value=False)
    agent._turn_completion_explainer_enabled = MagicMock(return_value=False)
    agent._skill_nudge_interval = 0
    agent._iters_since_skill = 0
    agent.clear_interrupt = MagicMock()
    agent._sync_external_memory_for_turn = MagicMock()
    agent._stream_callback = None
    agent.flushed = None

    def _capture_persist(msgs, history):
        agent.flushed = [m.copy() for m in msgs]

    agent._persist_session = _capture_persist

    messages = [
        {"role": "user", "content": "task"},
        _synthetic_nudge_message(flagged=True),
        {"role": "assistant", "content": "summary"},
    ]

    finalize_turn(
        agent,
        final_response="summary",
        api_call_count=60,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="t",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="max_iterations_reached(60/60)",
    )
    assert agent.flushed is not None
    assert all(m.get("content") != MAX_ITERATIONS_SUMMARY_REQUEST for m in agent.flushed)


def test_multi_round_flush_does_not_accumulate_nudges():
    agent = _agent_with_capturing_db()
    user = {"role": "user", "content": "task"}
    agent._flush_messages_to_session_db(
        [
            user,
            _synthetic_nudge_message(flagged=True),
            {"role": "assistant", "content": "summary-1"},
        ],
        conversation_history=[],
    )
    for idx in range(2, 4):
        agent._flush_messages_to_session_db(
            [
                user,
                _synthetic_nudge_message(flagged=True),
                {"role": "assistant", "content": f"summary-{idx}"},
            ],
            conversation_history=[user],
        )
    assert agent._session_db.rows.count(
        {"role": "user", "content": MAX_ITERATIONS_SUMMARY_REQUEST}
    ) == 0