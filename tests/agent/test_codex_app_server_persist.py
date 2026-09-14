"""Regression for #49225 — codex app-server turns must reach the session DB
exactly once.

The codex app-server runtime (``run_codex_app_server_turn``) is an early-return
path that bypasses ``conversation_loop`` and therefore never runs the loop's
per-step ``_persist_session()`` flushes. Before the fix, the projected
assistant/tool messages were persisted *nowhere* (state.db got only
session_meta rows), leaving ``session_search`` (FTS) and conversation-distill
blind to real gateway conversations.

The fix has the codex runtime flush its own projected messages via
``_flush_messages_to_session_db()`` (idempotent through the intrinsic
``_DB_PERSISTED_MARKER``) and return ``agent_persisted=True`` so the gateway
skips its own ``append_to_transcript`` DB write. This is critical: the inbound
user turn is already flushed at turn start (``turn_context._persist_session``),
and ``append_message`` is a raw INSERT with no dedup — a gateway re-write would
duplicate the user turn (#860 / #42039). This test locks in:

1. ``run_codex_app_server_turn`` flushes projected messages and returns
   ``agent_persisted=True``.
2. Exactly-once persistence: the already-flushed user turn is NOT re-written,
   and the new projected assistant message lands once.
3. The gateway resolution expression preserves standard-runtime behaviour.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.codex_runtime import run_codex_app_server_turn
from hermes_state import SessionDB
from run_agent import AIAgent


def _make_turn():
    return SimpleNamespace(
        interrupted=False,
        error=None,
        thread_id="thread-1",
        turn_id="turn-1",
        projected_messages=[{"role": "assistant", "content": "CODEX_ASSISTANT"}],
        tool_iterations=0,
        final_text="CODEX_ASSISTANT",
        should_retire=False,
    )


def _make_agent(session_db=None, session_id="sess-codex"):
    agent = MagicMock()
    # Pre-seed the session so run_codex_app_server_turn skips the spawn block.
    agent._codex_session = MagicMock()
    agent._codex_session.run_turn.return_value = _make_turn()
    agent.tool_progress_callback = None
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._session_db = session_db
    agent._session_db_created = True
    agent.session_id = session_id
    return agent


def test_codex_success_flushes_and_reports_persisted():
    """Codex success turn must self-persist and return agent_persisted=True."""
    agent = _make_agent(session_db=None)  # no DB -> flush is a no-op, still True
    result = run_codex_app_server_turn(
        agent,
        user_message="hello",
        original_user_message="hello",
        messages=[{"role": "user", "content": "hello"}],
        effective_task_id="task-1",
    )
    assert result["completed"] is True
    assert isinstance(result["messages"][-1]["timestamp"], float)
    # With the agent as sole persister, the gateway must SKIP its DB write.
    assert result["agent_persisted"] is True


def test_codex_blocked_candidate_is_repaired_before_persist_or_delivery():
    agent = _make_agent(session_db=None)
    blocked = _make_turn()
    blocked.final_text = "UNCERTIFIED_CLINICAL_CONTENT"
    blocked.projected_messages = [{"role": "assistant", "content": blocked.final_text}]
    repaired = _make_turn()
    repaired.turn_id = "turn-2"
    repaired.final_text = "CERTIFIED_CLINICAL_CONTENT"
    repaired.submitted_user_text = "repair directive"
    repaired.projected_messages = [
        {"role": "user", "content": "repair directive"},
        {"role": "assistant", "content": repaired.final_text},
    ]
    agent._codex_session.run_turn.side_effect = [blocked, repaired]
    decisions = [
        SimpleNamespace(action="block", response_text="safe fallback", reason="missing evidence"),
        SimpleNamespace(action="allow", response_text=repaired.final_text, reason=""),
    ]

    with patch("agent.turn_finalizer._prepare_output_for_delivery", side_effect=decisions), \
         patch("agent.turn_stop_gates._build_pre_delivery_repair_directive", return_value="repair directive"):
        result = run_codex_app_server_turn(
            agent,
            user_message="clinical question",
            original_user_message="clinical question",
            messages=[{"role": "user", "content": "clinical question"}],
            effective_task_id="task-1",
        )

    assert agent._codex_session.run_turn.call_count == 2
    assert result["final_response"] == "CERTIFIED_CLINICAL_CONTENT"
    persisted_text = "\n".join(str(message.get("content") or "") for message in result["messages"])
    assert "UNCERTIFIED_CLINICAL_CONTENT" not in persisted_text
    assert "repair directive" not in persisted_text


def test_codex_repair_threshold_escalates_until_certified_without_fallback():
    agent = _make_agent(session_db=None)
    from agent.turn_stop_gates import _MAX_PRE_DELIVERY_REPAIR_ATTEMPTS

    turns = []
    expected_provider_turns = _MAX_PRE_DELIVERY_REPAIR_ATTEMPTS + 2
    for index in range(expected_provider_turns):
        turn = _make_turn()
        turn.turn_id = f"turn-{index + 1}"
        turn.final_text = f"UNCERTIFIED_{index + 1}"
        turn.projected_messages = [{"role": "assistant", "content": turn.final_text}]
        turns.append(turn)
    agent._codex_session.run_turn.side_effect = turns
    decisions = [
        SimpleNamespace(action="block", response_text="SAFE_TERMINAL", reason="missing evidence")
        for _ in turns[:-1]
    ]
    decisions.append(SimpleNamespace(
        action="allow", response_text="CERTIFIED_AFTER_ESCALATION", reason=""
    ))

    with patch("agent.turn_finalizer._prepare_output_for_delivery", side_effect=decisions), \
         patch("agent.turn_stop_gates._build_pre_delivery_repair_directive", return_value="repair directive"):
        result = run_codex_app_server_turn(
            agent,
            user_message="clinical question",
            original_user_message="clinical question",
            messages=[{"role": "user", "content": "clinical question"}],
            effective_task_id="task-1",
        )

    assert agent._codex_session.run_turn.call_count == expected_provider_turns
    assert result["final_response"] == "CERTIFIED_AFTER_ESCALATION"
    persisted_text = "\n".join(str(message.get("content") or "") for message in result["messages"])
    assert "UNCERTIFIED_" not in persisted_text
    assert "SAFE_TERMINAL" not in persisted_text
    assert persisted_text.endswith("CERTIFIED_AFTER_ESCALATION")


def test_codex_repair_error_never_delivers_or_persists_uncertified_text():
    agent = _make_agent(session_db=None)
    blocked = _make_turn()
    blocked.final_text = "UNCERTIFIED_INITIAL"
    blocked.projected_messages = [{"role": "assistant", "content": blocked.final_text}]
    failed_repair = _make_turn()
    failed_repair.turn_id = "turn-2"
    failed_repair.error = "provider failed"
    failed_repair.final_text = "UNCERTIFIED_PARTIAL"
    failed_repair.submitted_user_text = "repair directive"
    failed_repair.projected_messages = [
        {"role": "user", "content": "repair directive"},
        {"role": "assistant", "content": failed_repair.final_text},
    ]
    agent._codex_session.run_turn.side_effect = [blocked, failed_repair]

    with patch(
        "agent.turn_finalizer._prepare_output_for_delivery",
        return_value=SimpleNamespace(
            action="block", response_text="FORBIDDEN_DEGRADATION", reason="missing evidence"
        ),
    ), patch(
        "agent.turn_stop_gates._build_pre_delivery_repair_directive",
        return_value="repair directive",
    ):
        result = run_codex_app_server_turn(
            agent,
            user_message="clinical question",
            original_user_message="clinical question",
            messages=[{"role": "user", "content": "clinical question"}],
            effective_task_id="task-1",
        )

    assert result["completed"] is False
    assert result["final_response"] is None
    persisted_text = "\n".join(str(message.get("content") or "") for message in result["messages"])
    assert "UNCERTIFIED" not in persisted_text
    assert "FORBIDDEN_DEGRADATION" not in persisted_text
    assert "repair directive" not in persisted_text


def test_codex_user_interrupt_is_reported_and_cleared():
    agent = _make_agent(session_db=None)
    turn = _make_turn()
    turn.interrupted = True
    turn.final_text = ""
    agent._codex_session.run_turn.return_value = turn
    agent._interrupt_requested = True
    agent._interrupt_message = "new correction"

    def clear_interrupt():
        agent._interrupt_requested = False
        agent._interrupt_message = None

    agent.clear_interrupt.side_effect = clear_interrupt
    result = run_codex_app_server_turn(
        agent,
        user_message="hello",
        original_user_message="hello",
        messages=[{"role": "user", "content": "hello"}],
        effective_task_id="task-1",
    )

    assert result["interrupted"] is True
    assert result["interrupt_message"] == "new correction"
    agent.clear_interrupt.assert_called_once_with()
    assert agent._interrupt_requested is False


def test_codex_turn_persists_each_message_exactly_once():
    """The user turn (flushed at turn start) must not be duplicated; the
    projected assistant message must land once.  Uses a real SessionDB and the
    real AIAgent._flush_messages_to_session_db to prove no #860/#42039
    duplicate-write regression on the codex path."""
    tmp = tempfile.mkdtemp(prefix="codex_persist_")
    db = None
    try:
        db = SessionDB(Path(tmp) / "state.db")
        sid = "sess-codex-once"
        db.create_session(session_id=sid, source="telegram", model="codex")

        # Real agent bound to this DB/session, minimal construction.
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=db,
            session_id=sid,
        )
        agent._session_db_created = True
        agent._codex_session = MagicMock()
        agent._codex_session.run_turn.return_value = _make_turn()
        agent.tool_progress_callback = None

        # Model the real flow: the inbound user turn is flushed at turn start
        # (turn_context._persist_session) on the SAME `messages` list the codex
        # path later reuses. That flush stamps _DB_PERSISTED_MARKER on the user
        # dict, so the codex-path flush skips it — no duplicate.
        user_msg = {"role": "user", "content": "USER_TURN"}
        messages = [user_msg]
        agent._flush_messages_to_session_db(messages)  # turn-start flush

        result = run_codex_app_server_turn(
            agent,
            user_message="USER_TURN",
            original_user_message="USER_TURN",
            messages=messages,
            effective_task_id="task-1",
        )
        assert result["agent_persisted"] is True

        rows = db.get_messages(sid, include_inactive=True)
        contents = [r["content"] for r in rows]
        # Exactly one user turn, exactly one assistant turn — no duplicates.
        assert contents.count("USER_TURN") == 1, contents
        assert contents.count("CODEX_ASSISTANT") == 1, contents
        assistant_row = next(
            row for row in rows if row["content"] == "CODEX_ASSISTANT"
        )
        assert isinstance(assistant_row["timestamp"], float)
        # session_search can now see the codex conversation.
        hits = {r["session_id"] for r in db.search_messages("CODEX_ASSISTANT")}
        assert sid in hits
    finally:
        import shutil

        if db is not None:
            db.close()
        shutil.rmtree(tmp, ignore_errors=True)


class TestGatewayPersistedResolution:
    """The gateway default must preserve standard-runtime skip-db behaviour."""

    @staticmethod
    def _resolve_persistence_block(agent_result, session_db_present):
        # gateway/run.py persistence block:
        #   agent_persisted = agent_result.get("agent_persisted", self._session_db is not None)
        return agent_result.get("agent_persisted", session_db_present)

    @staticmethod
    def _resolve_passthrough(result_holder0):
        # gateway/run.py result_holder passthrough:
        #   result_holder[0].get("agent_persisted", True) if result_holder[0] else True
        return result_holder0.get("agent_persisted", True) if result_holder0 else True

    def test_codex_result_keeps_gateway_skip(self):
        # Codex now self-persists → gateway must SKIP (agent_persisted True).
        codex = {"agent_persisted": True}
        assert self._resolve_persistence_block(codex, True) is True
        assert self._resolve_persistence_block(codex, False) is True
        assert self._resolve_passthrough(codex) is True

    def test_standard_runtime_preserves_skip_db(self):
        # Standard runtime omits the key → old behaviour: skip iff DB present.
        standard = {"final_response": "ok"}
        assert self._resolve_persistence_block(standard, True) is True
        assert self._resolve_persistence_block(standard, False) is False
        assert self._resolve_passthrough(standard) is True

    def test_missing_result_holder_defaults_persisted(self):
        assert self._resolve_passthrough(None) is True
