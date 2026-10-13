"""Lean compaction lists this session's still-running background processes in its appendix.

A backgrounded command's handle sits in a tool result the summary can drop; the appendix
re-lists the live handles from the process registry on every compaction.
"""
import json
import shlex
import subprocess
import sys
import time

import pytest

import agent.context_compressor as cc
from tools.process_registry import ProcessRegistry
from tools.terminal_tool import terminal_tool

HEADING = "## Background Processes (still running)"
SESSION = "sess-bg-current"


@pytest.fixture
def registry(monkeypatch):
    import tools.process_registry as processes

    registry = ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    yield registry
    registry.kill_all()


@pytest.fixture
def compressor(monkeypatch):
    monkeypatch.setattr(cc, "get_model_context_length", lambda *a, **k: 100_000)
    monkeypatch.setattr(cc, "call_llm", lambda **kw: {"choices": [{
        "message": {"content": "## Goal\nSynthetic work is recorded."}, "finish_reason": "stop"}]})
    compressor = cc.ContextCompressor(
        model="test/model", config_context_length=100_000,
        protect_first_n=0, protect_last_n=3, quiet_mode=True, tail_mode="lean",
    )
    compressor.tail_token_budget = 50
    compressor.bind_session_state(session_id=SESSION)
    return compressor


def _spawn(task_id, tmp_path, code="import time; time.sleep(60)"):
    command = shlex.quote(sys.executable) + " -c " + shlex.quote(code)
    result = json.loads(terminal_tool(command, background=True, task_id=task_id, workdir=str(tmp_path)))
    return result["session_id"]


def _summary(compressor):
    rows = [{"role": "system", "content": "Synthetic test conversation."}]
    for i in range(10):
        rows += [{"role": "user", "content": f"Record ordinary progress {i}."},
                 {"role": "assistant", "content": f"Progress {i} recorded."}]
    out = compressor.compress(rows, force=True)
    assert not compressor._last_summary_fallback_used
    return next(row["content"] for row in out if cc.SUMMARY_PREFIX in (row.get("content") or ""))


def _section(summary):
    assert summary.count(HEADING) == 1, summary
    return summary.split(HEADING, 1)[1].split("\n## ", 1)[0]


def test_running_process_listed_and_finished_one_not(registry, compressor, tmp_path):
    running = _spawn(SESSION, tmp_path)
    finished = _spawn(SESSION, tmp_path, code="pass")
    deadline = time.time() + 15
    while registry.poll(finished)["status"] == "running" and time.time() < deadline:
        time.sleep(0.05)
    assert registry.poll(finished)["status"] != "running"

    section = _section(_summary(compressor))

    entry = next(line for line in section.splitlines() if running in line)
    assert entry.startswith(f"- {running} (started ") and shlex.quote(sys.executable)[:40] in entry
    assert finished not in section


def test_other_sessions_process_not_listed(registry, compressor, tmp_path):
    mine = _spawn(SESSION, tmp_path)
    theirs = _spawn("sess-bg-other", tmp_path)

    section = _section(_summary(compressor))

    assert mine in section
    assert theirs not in section


def test_section_rebuilt_and_absent_once_nothing_runs(registry, compressor, tmp_path):
    proc = _spawn(SESSION, tmp_path)
    assert proc in _section(_summary(compressor))

    registry.kill_process(proc)

    assert HEADING not in _summary(compressor)


def test_process_from_compression_ancestor_listed(registry, compressor, tmp_path):
    class _Lineage:
        def get_compression_lineage(self, session_id):
            return ["sess-bg-root", session_id]

    compressor._session_db = _Lineage()
    proc = _spawn("sess-bg-root", tmp_path)

    assert proc in _section(_summary(compressor))


def test_section_is_bounded(registry, compressor):
    for i in range(25):
        proc = registry._new_session(f"run-job-{i:02d} " + "x" * 200, SESSION, SESSION, "", "/")
        registry._running[proc.id] = proc

    section = _section(_summary(compressor))
    entries = [line for line in section.splitlines() if line.startswith("- proc_")]

    assert len(entries) == 10
    assert all(len(line.split("): ", 1)[1]) <= 80 for line in entries)
    assert len(HEADING) + len(section) <= 1_500
    assert "+15 more" in section


def test_recovered_process_whose_pid_exited_not_listed(registry, compressor, tmp_path):
    live = _spawn(SESSION, tmp_path)
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    proc = registry._new_session("recovered-then-exited", SESSION, SESSION, "", "/",
                                 pid=gone.pid, detached=True, pid_scope="host")
    registry._running[proc.id] = proc  # checkpoint recovery state: stored exited=False

    section = _section(_summary(compressor))

    assert live in section
    assert proc.id not in section


def test_lineage_failure_keeps_current_session_scope(registry, compressor, tmp_path):
    class _BrokenLineage:
        def get_compression_lineage(self, session_id):
            raise RuntimeError("unavailable")

    compressor._session_db = _BrokenLineage()
    proc = _spawn(SESSION, tmp_path)

    assert proc in _section(_summary(compressor))


def test_registry_failure_omits_section_without_fallback(compressor, monkeypatch):
    import tools.process_registry as processes

    class _BrokenRegistry:
        def __getattr__(self, name):
            raise RuntimeError("unavailable")

    monkeypatch.setattr(processes, "process_registry", _BrokenRegistry())
    summary = _summary(compressor)

    assert "Synthetic work is recorded." in summary
    assert HEADING not in summary
