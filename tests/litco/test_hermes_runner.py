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


def test_failed_terminal_command_is_an_error():
    # the terminal tool's result for `ls missing`: no error text, just a non-zero exit
    raw = '{"output": "ls: missing: No such file or directory", "exit_code": 1, "error": null}'
    out = _summarize_result(raw, "terminal")
    assert out == {"status": "error", "summary": "command failed with exit code 1"}
    assert "No such file" not in out["summary"]  # output stays private
    with_error = _summarize_result({"output": "", "exit_code": 127, "error": "command not found"}, "terminal")
    assert with_error == {"status": "error", "summary": "command not found (exit code 127)"}
    # a trailing hint after the JSON does not hide the exit code
    assert _summarize_result(raw + "\n[hint: cwd is /x]", "terminal")["status"] == "error"
    assert _summarize_result('{"output": "ok", "exit_code": 0}', "terminal")["status"] == "ok"


def test_mapper_applies_hermes_error_verdict(tmp_path):
    """Hermes reports is_error on tool.completed just before tool_complete_callback; a spilled result
    (unparseable for us) still arrives as an error."""
    ctx, events = _ctx(tmp_path)
    m = _EventMapper(ctx)
    raw = '{"output": "' + "x" * 50 + '", "exit_code": 2, "error": null}'
    spilled = "<persisted-output>\nThis tool result was too large...\n</persisted-output>"
    m.tool_start("c1", "terminal", {"command": "make"})
    m.progress("tool.completed", "terminal", None, None, duration=0.1, is_error=True, result=raw)
    m.tool_complete("c1", "terminal", {"command": "make"}, spilled)
    assert events[-1][1]["result"]["status"] == "error"
    assert events[-1][1]["result"]["summary"] == "command failed with exit code 2"
    # and a success verdict stays ok
    m.tool_start("c2", "terminal", {"command": "true"})
    m.progress("tool.completed", "terminal", None, None, duration=0.1, is_error=False,
               result='{"output": "", "exit_code": 0}')
    m.tool_complete("c2", "terminal", {}, '{"output": "", "exit_code": 0}')
    assert events[-1][1]["result"]["status"] == "ok"
    # no verdict reported: Hermes's classifier is consulted directly
    m.tool_start("c3", "web_fetch", {})
    m.tool_complete("c3", "web_fetch", {}, '{"error": "HTTP 404"}')
    assert events[-1][1]["result"] == {"status": "error", "summary": "HTTP 404", "durationMs": events[-1][1]["result"]["durationMs"]}


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


# ---------------------------------------------------------------------------
# memory scoping
# ---------------------------------------------------------------------------

class MemoryAgent(FakeAgent):
    """Records the built-in memory it was given, then writes a note to it."""

    def __init__(self, mapper, note, target="user"):
        super().__init__(mapper)
        from tools.memory_tool import MemoryStore
        self._memory_store = MemoryStore()   # what AIAgent builds: the profile-wide store
        self._memory_store.load_from_disk()
        self.note, self.target = note, target

    def run_conversation(self, user_message, conversation_history, task_id):
        store = self._memory_store
        self.seen = {"memory": list(store.memory_entries), "user": list(store.user_entries),
                     "prompt": store.format_for_system_prompt("user") or ""}
        assert store.add(self.target, self.note)["success"] is True
        return {"final_response": "noted"}


def test_memory_is_scoped_per_thread(runner, tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    agents = []
    plan = iter([("private fact for u1", "user"), ("private fact for u2", "user"),
                 ("team fact", "memory"), ("second team fact", "memory"), ("u1 again", "user")])

    def build(ctx, sid, mapper):
        note, target = next(plan)
        agent = MemoryAgent(mapper, note, target)
        agent.session_id = sid
        agents.append(agent)
        return agent

    monkeypatch.setattr(runner, "_build_agent", build)
    profile_user_md = get_hermes_home() / "memories" / "USER.md"
    profile_user_md.parent.mkdir(parents=True, exist_ok=True)
    profile_user_md.write_text("profile-wide entry from before the fix", encoding="utf-8")

    runner.run(_ctx(tmp_path, session_id="d1", kind="dm", user_id="u1")[0])
    runner.run(_ctx(tmp_path, session_id="d2", kind="dm", user_id="u2")[0])
    # u2's private turn saw neither u1's note nor the profile-wide file
    assert agents[1].seen["user"] == [] and agents[1].seen["memory"] == []
    assert (tmp_path / "users/u1/memories/USER.md").read_text().count("private fact for u1") == 1
    assert "u1" not in (tmp_path / "users/u2/memories/USER.md").read_text()

    runner.run(_ctx(tmp_path, session_id="c1", kind="channel")[0])
    runner.run(_ctx(tmp_path, session_id="c2", kind="channel")[0])
    # a second channel thread sees the shared matter memory, and nothing private
    assert agents[3].seen["memory"] == ["team fact"] and agents[3].seen["user"] == []
    assert (tmp_path / "shared/memories/MEMORY.md").is_file()

    runner.run(_ctx(tmp_path, session_id="d3", kind="dm", user_id="u1")[0])
    # u1's next private thread sees u1's own note (in the prompt too), not u2's, not the team's
    assert agents[4].seen["user"] == ["private fact for u1"]
    assert "private fact for u1" in agents[4].seen["prompt"] and "u2" not in agents[4].seen["prompt"]
    assert agents[4].seen["memory"] == []
    # the profile-wide memory was never read into a turn nor written by one
    assert profile_user_md.read_text(encoding="utf-8") == "profile-wide entry from before the fix"
    assert not (get_hermes_home() / "memories" / "MEMORY.md").exists() or \
        "team fact" not in (get_hermes_home() / "memories" / "MEMORY.md").read_text()


def test_memory_off_stays_off(runner, tmp_path, monkeypatch):
    holder = {}

    def build(ctx, sid, mapper):
        agent = FakeAgent(mapper)
        agent._memory_store = None
        holder["agent"] = agent
        return agent

    monkeypatch.setattr(runner, "_build_agent", build)
    runner.run(_ctx(tmp_path, kind="dm", user_id="u1")[0])
    assert holder["agent"]._memory_store is None


def test_verdict_logs_when_hermes_classification_raises(monkeypatch, caplog):
    import agent.display as display

    def boom(name, result):
        raise RuntimeError("classifier broke")

    monkeypatch.setattr(display, "_detect_tool_failure", boom)
    with caplog.at_level("WARNING", logger="litco.hermes_runner"):
        assert hermes_runner._hermes_verdict("terminal", "{}") == (False, "")
    assert any("classification raised for terminal" in r.getMessage() and r.exc_info for r in caplog.records)


def test_scoped_memory_store_class_is_built_once(tmp_path):
    from litco import memory_scope

    a = memory_scope.scoped_store(tmp_path / "a")
    b = memory_scope.scoped_store(tmp_path / "b")
    assert type(a) is type(b) is memory_scope._store_class()
    assert a.directory != b.directory
