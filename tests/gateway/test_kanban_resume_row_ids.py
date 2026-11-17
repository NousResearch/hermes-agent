"""A managed kanban worker hands its agent history rows that compaction can name by id.

In-place compression in a managed worker names the durable rows it held by ``_row_id``
(``agent/conversation_compression_archive.coverage_for_commit``); a history dict without one is
sent to the session owner whole, as an unresolved row inside ONE ``compression.archive`` journal entry. A
long or resumed attempt holds hundreds of such rows, so that one entry can outgrow the worker
journal's byte cap; the store then latches ``outbox_full`` and the attempt dies. The in-process
reload already asks for row ids (``agent/turn_facade_lease.py``); the managed worker's first
load and the kanban continuation reloads (the in-place retry and the goal loop) did not.
"""

import json
import sqlite3
import threading
from types import SimpleNamespace

from tests.gateway.test_managed_turn_contract import assignment


def test_managed_worker_first_turn_loads_history_with_row_ids(tmp_path, monkeypatch):
    from agent import managed_worker as worker

    frame, _ = assignment(tmp_path, monkeypatch)
    loads = []

    class Agent:
        def __init__(self, **kwargs):
            self.session_prompt_tokens = self.session_completion_tokens = 0

        def run_conversation(self, text, **kwargs):
            return {"final_response": "ok", "completed": True}

    class Store:
        def __init__(self, *args):
            pass

        def get_messages_as_conversation(self, sid, **kwargs):
            loads.append(kwargs)
            return []

        def flush_token_counts(self):
            pass

        def finish(self):
            pass

        def close(self):
            pass

    class Controls:
        def __init__(self, *args):
            self.stopped, self.finish = threading.Event(), threading.Event()
            self.finish.set()

        def approval(self, data):
            pass

        def clarify(self, questions):
            pass

    monkeypatch.setattr(worker, "bind_worker_policy", lambda frame: None)
    monkeypatch.setattr(worker, "discover_profile_mcp", lambda policy: None)
    monkeypatch.setattr(worker, "retire_agent", lambda agent: None)
    monkeypatch.setattr(worker, "WorkerControls", Controls)
    monkeypatch.setattr("agent.runtime_session_store.WorkerRPC", lambda home: lambda *a, **k: {"owner_epoch": 1})
    monkeypatch.setattr("agent.runtime_session_store.RuntimeSessionStore", Store)
    monkeypatch.setattr("tools.process_registry.process_registry.recover_from_checkpoint", lambda: None)
    monkeypatch.setattr("run_agent.AIAgent", Agent)
    monkeypatch.setattr("agent.title_generator.wait_for_title_upgrades", lambda: None)

    worker.execute(frame, SimpleNamespace(send=lambda kind, **payload: None))

    assert loads, "the worker never loaded its session history"
    assert loads[0].get("include_row_ids") is True, loads


def test_continuation_turns_hand_the_agent_rows_compaction_can_name(tmp_path, monkeypatch):
    """The in-place retry and the goal loop reload the session; both must keep the row ids."""
    import agent.kanban_turn_recovery as recovery
    from agent.conversation_compression_archive import coverage_for_commit
    from gateway import session_kanban
    from hermes_cli import goals, turn_exit
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("resumed", source="kanban", model="worker-model")
        for i in range(8):
            db.append_message("resumed", role="user" if i % 2 == 0 else "assistant", content=f"row {i}")
        with sqlite3.connect(tmp_path / "state.db") as conn:
            ids = [row[0] for row in conn.execute(
                "SELECT id FROM messages WHERE session_id='resumed' AND active=1 ORDER BY id")]
        seen = []

        class Agent:
            session_id = "resumed"
            _session_db = db

            def run_conversation(self, prompt, conversation_history=None, **kwargs):
                seen.append((prompt, conversation_history))
                return {"final_response": "done"}  # no ``messages``: the retry reloads the store

        monkeypatch.setattr(recovery, "recover_failed_kanban_turns", lambda turn, result, **kw: turn("nudge"))
        monkeypatch.setattr(turn_exit, "turn_exit_code", lambda result, **kw: 0)
        monkeypatch.setattr(goals, "run_kanban_goal_loop", lambda **kw: kw["run_turn"]("continue"))
        context = {"skills": [], "context": "", "task_id": "t_00000001", "run_id": 1, "claim_lock": "fixture",
                   "db": str(tmp_path / "kanban.db"), "goal_mode": True, "goal_text": "goal", "goal_max_turns": 1}
        session_kanban._run_task_turns(Agent(), {"text": "work"}, [], context)
        reloads = {prompt: history for prompt, history in seen if prompt in ("nudge", "continue")}
        assert set(reloads) == {"nudge", "continue"}, [p for p, _ in seen]
        for prompt, history in reloads.items():
            assert [m.get("_row_id") for m in history] == ids, prompt
            covered, unresolved = coverage_for_commit(db, "resumed", history)
            assert covered == ids and unresolved == [], (prompt, json.dumps(unresolved)[:200])
    finally:
        db.close()
