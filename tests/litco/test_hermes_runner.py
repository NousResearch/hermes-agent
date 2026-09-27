"""HermesTurnRunner event mapping and session handling, with the AIAgent replaced by a fake."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from litco import hermes_runner
from litco.hermes_runner import HermesTurnRunner, _EventMapper, _summarize_result
from litco.turn_server import TurnContext, TurnRequest


def _ctx(tmp_path: Path, session_id="s1", kind="channel", user_id="u1", text="hi"):
    events = []
    cwd = tmp_path / ("shared" if kind == "channel" else f"users/{user_id}")
    (cwd / "deliverables").mkdir(parents=True, exist_ok=True)
    req = TurnRequest(matter_id="m1", user_id=user_id, session_id=session_id, text=text, attachments=[],
                      channel="slack", kind=kind)
    ctx = TurnContext(turn_id="turn_1", request=req, home=tmp_path, cwd=cwd,
                      emit=lambda t, f: events.append((t, f)))
    return ctx, events


def test_mapper_deltas_resets_and_tools(tmp_path):
    ctx, events = _ctx(tmp_path)
    m = _EventMapper(ctx)
    m.delta(None)                   # flush before anything streamed: no reset
    m.delta("Checking")
    m.delta(None)                   # flush before tools
    m.tool_start("c1", "terminal", {"command": "ls"})
    m.progress("tool.started", "terminal", "ls", {})   # duplicate hook: ignored
    m.tool_complete("c1", "terminal", {"command": "ls"}, '{"output": "a\\nb", "exit_code": 0}')
    m.delta("Done")
    types = [t for t, _ in events]
    assert types == ["assistant_delta", "tool_started", "tool_complete", "assistant_reset", "assistant_delta"]
    assert events[1][1]["call"] == {"toolCallId": "c1", "name": "terminal"}
    result = events[2][1]["result"]
    assert result["status"] == "ok" and "a\nb" not in result["summary"]


def test_mapper_subagent_progress(tmp_path):
    ctx, events = _ctx(tmp_path)
    m = _EventMapper(ctx)
    m.tool_start("d1", "delegate_task", {"goal": "x"})
    m.progress("subagent.start", "delegate_task", "Researching venue")
    assert events[-1] == ("tool_progress", {"call": {"toolCallId": "d1", "name": "delegate_task"},
                                            "message": "Researching venue"})


def test_summaries_flag_errors_and_hide_content():
    assert _summarize_result('{"error": "file not found"}') == {"status": "error", "summary": "file not found"}
    assert _summarize_result({"success": False, "message": "denied"})["status"] == "error"
    ok = _summarize_result('{"content": "PRIVILEGED MEMO TEXT"}')
    assert ok["status"] == "ok" and "PRIVILEGED" not in ok["summary"]
    assert _summarize_result("Error: boom")["status"] == "error"


class FakeAgent:
    def __init__(self, mapper, *, rotate_to=None, block=None):
        self.mapper = mapper
        self.session_id = "unset"
        self.session_prompt_tokens = 120
        self.session_completion_tokens = 30
        self.session_cache_read_tokens = 100
        self.session_cache_write_tokens = 0
        self.model = "fake/model"
        self.rotate_to = rotate_to
        self.block = block
        self.interrupted = threading.Event()
        self.seen = {}

    def interrupt(self, message=None, **kw):
        self.interrupted.set()

    def run_conversation(self, user_message, conversation_history, task_id):
        from agent.runtime_cwd import scoped_session_cwd
        from tools.terminal_tool import get_session_cwd
        self.seen = {"message": user_message, "history": conversation_history, "task_id": task_id,
                     "cwd": get_session_cwd(task_id), "scope_cwd": scoped_session_cwd()}
        self.mapper.delta("partial")
        if self.block is not None:
            self.interrupted.wait(5)
            return {"final_response": "cut short", "interrupted": True}
        self.mapper.tool_start("c1", "write_file", {"path": "deliverables/x.md"})
        self.mapper.tool_complete("c1", "write_file", {}, '{"bytes_written": 3}')
        if self.rotate_to:
            self.session_id = self.rotate_to
        return {"final_response": "all done"}


@pytest.fixture
def runner(monkeypatch, tmp_path):
    r = HermesTurnRunner()
    r._session_map = hermes_runner._SessionMap(tmp_path / "sessions.json")

    class _DB:
        def get_messages_as_conversation(self, sid):
            return [{"role": "user", "content": f"earlier in {sid}"}]

    monkeypatch.setattr(r, "_session_db", lambda: _DB())
    return r


def test_run_maps_outcome_cwd_and_session(runner, tmp_path, monkeypatch):
    agents = []

    def build(ctx, sid, mapper):
        agent = FakeAgent(mapper, rotate_to="litco_s1_rotated")
        agent.session_id = sid
        agents.append(agent)
        return agent

    monkeypatch.setattr(runner, "_build_agent", build)
    ctx, events = _ctx(tmp_path, kind="dm", user_id="u7", text="draft it")
    outcome = runner.run(ctx)
    agent = agents[0]
    assert outcome.text == "all done"
    assert (outcome.input_tokens, outcome.output_tokens, outcome.cache_read_tokens) == (120, 30, 100)
    assert outcome.model_used == "fake/model" and outcome.halted is None and outcome.error is None
    assert agent.seen["task_id"] == "litco_s1"
    assert agent.seen["history"] == [{"role": "user", "content": "earlier in litco_s1"}]
    assert agent.seen["cwd"] == str(ctx.cwd) and agent.seen["scope_cwd"] == str(ctx.cwd)
    assert agent.seen["message"] == "draft it"
    assert [t for t, _ in events] == ["assistant_delta", "tool_started", "tool_complete"]
    # compaction rotated the Hermes id: the next turn on this thread resumes the new one
    assert runner._sessions().get("s1") == "litco_s1_rotated"


def test_run_interrupt(runner, tmp_path, monkeypatch):
    holder = {}

    def build(ctx, sid, mapper):
        holder["agent"] = FakeAgent(mapper, block=True)
        return holder["agent"]

    monkeypatch.setattr(runner, "_build_agent", build)
    ctx, _ = _ctx(tmp_path)
    t = threading.Thread(target=lambda: holder.setdefault("outcome", runner.run(ctx)))
    t.start()
    for _ in range(500):
        if "agent" in holder and holder["agent"].seen:
            break
        threading.Event().wait(0.01)
    ctx.interrupt("interrupted")
    t.join(5)
    assert holder["agent"].interrupted.is_set()
    assert holder["outcome"].halted == "interrupted"
    assert holder["outcome"].text == "cut short"


def test_run_classifies_exceptions(runner, tmp_path, monkeypatch):
    def build(ctx, sid, mapper):
        raise RuntimeError("401 invalid api key")

    monkeypatch.setattr(runner, "_build_agent", build)
    ctx, _ = _ctx(tmp_path)
    outcome = runner.run(ctx)
    assert outcome.error and outcome.error_category == "auth"
