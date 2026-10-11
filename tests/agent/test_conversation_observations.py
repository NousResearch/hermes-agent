"""The real facade publishes admitted generations and actual terminal branches."""
from types import SimpleNamespace
import pytest
from hermes_state import SessionDB
from run_agent import AIAgent


def test_facade_brackets_real_lease_and_fences_next_turn(tmp_path, monkeypatch):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "desktop", profile_name="default")
        agent = AIAgent.__new__(AIAgent)
        for key, value in dict(session_id="s", platform="desktop", model="test", _session_db=db,
            _session_db_created=True, _persist_disabled=False, _parent_session_id=None,
            _relay_pending_turn_id=None, log_prefix="", status_callback=None,
            _interrupt_requested=False, _interrupt_message=None, _pending_redirect=None,
            _execution_thread_id=None, _interrupt_thread_signal_pending=False).items():
            setattr(agent, key, value)
        agent._reset_activity_labels_after_turn = lambda: None
        agent._conversation_root_id = lambda: "s"
        agent._vprint = lambda *a, **k: None
        def loop(agent, *args, **kwargs):
            row = db.read_session_observations(["s"], profile="default")[0]
            assert row["provenance"] == "native", "admitted turn did not publish native proof"
            assert row["turn_id"] == agent._observation_turn_id
            assert row["execution"] == "running"
            assert db.open_session_attention("s", row["turn_id"], "validation")
            return {"completed": True, "messages": [], "final_response": "synthetic"}
        monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
        agent.run_conversation("synthetic", conversation_history=[])
        first = db.read_session_observations(["s"], profile="default")[0]
        assert first["execution"] == "idle" and first["last_result"]["status"] == "complete"
        assert first["attention"]["kind"] == "validation"
        def fail(*args, **kwargs):
            raise RuntimeError("synthetic failure")
        monkeypatch.setattr("agent.conversation_loop.run_conversation", fail)
        with pytest.raises(RuntimeError, match="synthetic"):
            agent.run_conversation("synthetic", conversation_history=[])
        second = db.read_session_observations(["s"], profile="default")[0]
        assert second["last_result"]["status"] == "error" and second["turn_id"] != first["turn_id"]
        assert second["attention"] == first["attention"]


def test_prose_is_not_a_completion_proof(tmp_path, monkeypatch):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "desktop", profile_name="default")
        agent = AIAgent.__new__(AIAgent)
        for key, value in dict(session_id="s", platform="desktop", model="test", _session_db=db,
            _session_db_created=True, _persist_disabled=False, _parent_session_id=None,
            _relay_pending_turn_id=None, log_prefix="", status_callback=None,
            _interrupt_requested=False, _interrupt_message=None, _pending_redirect=None,
            _execution_thread_id=None, _interrupt_thread_signal_pending=False).items():
            setattr(agent, key, value)
        agent._reset_activity_labels_after_turn = lambda: None
        agent._conversation_root_id = lambda: "s"
        agent._vprint = lambda *a, **k: None
        monkeypatch.setattr("agent.conversation_loop.run_conversation", lambda *a, **k:
                            {"completed": False, "final_response": "synthetic pending continuation", "messages": []})
        agent.run_conversation("synthetic", conversation_history=[])
        row = db.read_session_observations(["s"], profile="default")[0]
        assert row["execution"] == "unknown" and row["last_result"] is None, "prose inferred complete"


def test_uninstrumented_human_requests_remain_unknown(tmp_path, monkeypatch):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "cli", profile_name="default")
        agent = AIAgent.__new__(AIAgent)
        for key, value in dict(session_id="s", platform="cli", model="test", _session_db=db,
            _session_db_created=True, _persist_disabled=False, _parent_session_id=None,
            _relay_pending_turn_id=None, log_prefix="", status_callback=None,
            _interrupt_requested=False, _interrupt_message=None, _pending_redirect=None,
            _execution_thread_id=None, _interrupt_thread_signal_pending=False).items():
            setattr(agent, key, value)
        agent._reset_activity_labels_after_turn = lambda: None
        agent._conversation_root_id = lambda: "s"
        agent._vprint = lambda *a, **k: None
        def loop(*args, **kwargs):
            row = db.read_session_observations(["s"], profile="default")[0]
            assert row["attention"]["kind"] == "unknown", "no callback coverage is not absence of attention"
            return {"completed": True, "messages": []}
        monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
        agent.run_conversation("synthetic", conversation_history=[])
