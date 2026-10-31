"""Compaction preserves live process handles without trusting a model's process snapshot."""

import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

import agent.context_compressor as cc
from agent.prepared_compaction import PreparedCompaction
from tests.waiting import wait_until
from tools.process_registry import ProcessRegistry

HEADING = "## Background Processes (still running)"
SESSION = "session-current"


@pytest.fixture
def registry(monkeypatch):
    import tools.process_registry as processes

    registry = ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    yield registry
    for session in list(registry._running.values()):
        if session.process is not None:
            _finish(registry, session.id)


@pytest.fixture
def compressor(monkeypatch):
    monkeypatch.setattr(cc, "get_model_context_length", lambda *a, **k: 100_000)
    monkeypatch.setattr(cc, "call_llm", lambda **kw: {"choices": [{
        "message": {"content": (
            f"## Goal\nKeep the inline mention {HEADING}.\n\n"
            f"{HEADING}\n- stale-model-handle\n\n## Next Steps\nContinue the work."
        )}, "finish_reason": "stop"}]})
    compressor = cc.ContextCompressor(
        model="test/model", config_context_length=100_000,
        protect_first_n=0, protect_last_n=3, quiet_mode=True, tail_mode="lean",
    )
    compressor.tail_token_budget = 50
    compressor.bind_session_state(session_id=SESSION)
    return compressor


def _spawn(registry, task_id, tmp_path, code="import sys; sys.stdin.read()"):
    argv = [sys.executable, "-c", code]
    proc = subprocess.Popen(argv, cwd=tmp_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    return registry.adopt_local(proc, command=shlex.join(argv), task_id=task_id, cwd=str(tmp_path)).id


def _finish(registry, session_id):
    proc = registry.get(session_id).process
    proc.stdin.close()
    proc.wait(timeout=5)
    wait_until(lambda: registry.poll(session_id)["status"] != "running", what="process exit")


def _messages():
    rows = [{"role": "system", "content": "Synthetic test conversation."}]
    for i in range(10):
        rows += [{"role": "user", "content": f"Record ordinary progress {i}."},
                 {"role": "assistant", "content": f"Progress {i} recorded."}]
    return rows


def _summary(compressor, rows=None, **kwargs):
    out = compressor.compress(rows or _messages(), **kwargs)
    assert not compressor._last_summary_fallback_used
    return next(row["content"] for row in out if row.get(cc.COMPRESSED_SUMMARY_METADATA_KEY))


def _section(summary):
    assert summary.splitlines().count(HEADING) == 1, summary
    return summary.split("\n" + HEADING, 1)[1].split("\n## ", 1)[0]


def test_inline_compaction_replaces_model_snapshot_with_live_owned_processes(registry, compressor, tmp_path):
    running = _spawn(registry, SESSION, tmp_path)
    finished = _spawn(registry, SESSION, tmp_path, code="pass")
    other = _spawn(registry, "session-other", tmp_path)
    wait_until(lambda: registry.poll(finished)["status"] != "running", what="process exit")

    summary = _summary(compressor, force=True)
    section = _section(summary)

    entry = next(line for line in section.splitlines() if running in line)
    assert entry.startswith(f"- {running} (started ")
    assert shlex.quote(sys.executable)[:40] in entry
    assert finished not in section and other not in section
    assert "stale-model-handle" not in summary
    assert f"Keep the inline mention {HEADING}." in summary
    assert "## Next Steps\nContinue the work." in summary

    _finish(registry, running)
    summary = _summary(compressor, force=True)
    assert HEADING not in summary.splitlines()
    assert "stale-model-handle" not in summary


def test_prepared_compaction_refreshes_processes_at_adoption(registry, compressor, tmp_path):
    compressor.prepared_compaction = PreparedCompaction()
    compressor.last_prompt_tokens = compressor.threshold_tokens
    rows = _messages()
    old = _spawn(registry, SESSION, tmp_path)
    agent = SimpleNamespace(context_compressor=compressor, compression_enabled=True, api_mode="chat_completions")
    thread = compressor.prepared_compaction.maybe_prepare(agent, compressor, rows, origin="tool-batch")
    assert thread is not None
    thread.join(5)
    prepared = compressor.prepared_compaction._entry
    assert prepared is not None

    _finish(registry, old)
    later = _spawn(registry, SESSION, tmp_path)
    ancestor = _spawn(registry, "session-ancestor", tmp_path)
    compressor._session_db = SimpleNamespace(get_compression_lineage=lambda sid: ["session-ancestor", sid])
    summary = _summary(compressor, rows, current_tokens=compressor.threshold_tokens + 1)

    assert compressor.prepared_compaction.spliced_attempt_id is not None
    for text in (summary, compressor._previous_summary):
        section = _section(text)
        assert later in section and ancestor in section
        assert old not in section and "stale-model-handle" not in text


def test_process_appendix_is_bounded(registry, compressor):
    for i in range(25):
        proc = registry._new_session(f"run-job-{i:02d} " + "x" * 200, SESSION, SESSION, "", "")
        registry._running[proc.id] = proc

    section = _section(_summary(compressor, force=True))
    entries = [line for line in section.splitlines() if line.startswith("- proc_")]

    assert len(entries) == 10
    assert all(len(line.split("): ", 1)[1]) <= 80 for line in entries)
    assert len(HEADING) + len(section) <= 1_500
    assert "+15 more" in section


def test_registry_failure_omits_process_snapshot(compressor, monkeypatch):
    import tools.process_registry as processes

    def unavailable():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(processes.process_registry, "list_sessions", unavailable)
    summary = _summary(compressor, force=True)

    assert "Continue the work." in summary
    assert HEADING not in summary.splitlines()
    assert "stale-model-handle" not in summary
