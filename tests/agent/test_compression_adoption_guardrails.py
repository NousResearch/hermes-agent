"""A late compression contender re-anchors its own ongoing-turn guardrails."""

import pytest

from agent.conversation_compression import _adopt_live_compression_child
from hermes_state import SessionDB
from run_agent import AIAgent


def _agent(db, session_id):
    agent = AIAgent(
        api_key="test-key",
        base_url="http://127.0.0.1:9/v1",
        model="test/model",
        platform="telegram",
        quiet_mode=True,
        session_db=db,
        session_id=session_id,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        save_trajectories=False,
    )
    # Adoption returns before summary generation. Avoid only the unrelated
    # one-time provider feasibility probe; all adoption/DB/guardrail code is real.
    agent._compression_feasibility_checked = True
    agent._cached_system_prompt = "sys"
    return agent


def _read(guardrails, tool_name, args, result):
    before = guardrails.before_call(tool_name, args)
    assert before.action == "allow", before
    after = guardrails.after_call(tool_name, args, result, failed=False)
    observed = guardrails.observe_call(tool_name, args, result)
    return after, observed


def _publish_child(db, parent, child, compacted):
    holder = "test-compression-winner"
    assert db.try_acquire_compression_lock(parent, holder)
    try:
        db.publish_compression_child(
            parent_session_id=parent,
            child_session_id=child,
            source="telegram",
            messages=compacted,
            system_prompt="sys",
            compression_lock_holder=holder,
        )
    finally:
        db.release_compression_lock(parent, holder)


@pytest.mark.parametrize("tool_name", ["read_file", "skill_view"])
def test_live_child_adoption_reanchors_ongoing_turn(tmp_path, tool_name):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("parent", source="telegram")
        agent = _agent(db, "parent")
        guardrails = agent._tool_guardrails
        args = {"path": "app.py", "offset": 1, "limit": 20} if tool_name == "read_file" else {"name": "editing"}
        result = "unchanged contents\n" * 40
        failed_args = {"command": "missing-command"}
        for _ in range(guardrails.config.exact_failure_block_after - 1):
            guardrails.after_call("terminal", failed_args, '{"error":"not found"}', failed=True)
        for _ in range(guardrails.config.no_progress_block_after - 1):
            _read(guardrails, tool_name, args, result)
        assert guardrails.halt_decision is None

        compacted = [{"role": "user", "content": "continue the edit from summary"}]
        _publish_child(db, "parent", "middle", compacted)
        _publish_child(db, "middle", "child", compacted)
        returned, _ = agent._compress_context(
            [{"role": "user", "content": "stale context " * 100}], "sys", approx_tokens=10000
        )
        assert agent.session_id == "child"
        assert [(m["role"], m["content"]) for m in returned] == [
            (m["role"], m["content"]) for m in compacted
        ]
        after, observed = _read(guardrails, tool_name, args, result)
        assert guardrails.halt_decision is None
        assert after.action == "allow"
        assert observed.notice is None
        assert observed.stub is None  # The original full result left the live transcript.

        # Seeing the same already-adopted tip cannot replenish the grace.
        assert _adopt_live_compression_child(agent, db, "parent") is not None

        # Adoption cannot provide unlimited grace for a genuine subsequent loop.
        for _ in range(guardrails.config.no_progress_block_after - 2):
            _read(guardrails, tool_name, args, result)
            assert guardrails.halt_decision is None
        _read(guardrails, tool_name, args, result)
        assert guardrails.halt_decision.code == "identical_call_streak_halt"

        # Re-anchoring must preserve failures; it is not a fresh turn reset.
        guardrails.after_call("terminal", failed_args, '{"error":"not found"}', failed=True)
        assert guardrails.before_call("terminal", failed_args).code == "repeated_exact_failure_block"
    finally:
        db.close()


@pytest.mark.parametrize("handoff", ["missing", "empty", "ended", "newer_empty_tip", "changed_after_load"])
def test_rejected_adoption_preserves_ongoing_turn_guardrails(tmp_path, monkeypatch, handoff):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("parent", source="telegram")
        agent = _agent(db, "parent")
        guardrails = agent._tool_guardrails
        args = {"path": "app.py"}
        result = "unchanged contents\n" * 40
        for _ in range(guardrails.config.no_progress_block_after - 1):
            _read(guardrails, "read_file", args, result)
        db.end_session("parent", "compression")
        if handoff != "missing":
            db.create_session("child", source="telegram", parent_session_id="parent")
            if handoff != "empty":
                db.replace_messages("child", [{"role": "user", "content": "summary"}])
            if handoff == "ended":
                db.end_session("child", "user_close")
            if handoff == "newer_empty_tip":
                db.create_session("rival", source="telegram", parent_session_id="parent")
        if handoff == "changed_after_load":
            original_loader = type(db).get_messages_as_conversation

            def load_then_race(self, session_id):
                recovered = original_loader(self, session_id)
                self.create_session("rival", source="telegram", parent_session_id="parent")
                return recovered

            # Inject only race timing. Both transcript reads and lineage validation
            # still execute against real SQLite rows.
            monkeypatch.setattr(type(db), "get_messages_as_conversation", load_then_race)
        stale = [{"role": "user", "content": "stale context"}]
        returned, _ = agent._compress_context(stale, "sys", approx_tokens=10000)
        assert agent.session_id == "parent"
        assert returned == stale
        after, observed = _read(guardrails, "read_file", args, result)
        assert after.code == "idempotent_no_progress_warning"
        assert observed.notice is not None
        assert guardrails.halt_decision.code == "identical_call_streak_halt"
    finally:
        db.close()
