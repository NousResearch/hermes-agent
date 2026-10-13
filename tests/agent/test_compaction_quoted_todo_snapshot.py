"""A todo snapshot folded into a user row must not be quoted into the rebuilt lean appendix.

An earlier boundary folds the task list into the trailing real user row. When that row is later
compacted, quoting it verbatim copies the snapshot header into the summary; the next fold then takes
the summary for the live snapshot and cuts everything after the header (background processes,
recovery footer, end marker). Quotes keep the user's words and drop the scaffolding.
"""
import json
import os
import shlex
import sys
from unittest.mock import patch

import pytest

import agent.context_compressor as cc
import run_agent
from agent.context_compressor_background import _LEAN_BACKGROUND_HEADING
from gateway.compaction_todo_hydration import hydrate_todos_for_compaction
from tools.process_registry import ProcessRegistry
from tools.terminal_tool import terminal_tool
from tools.todo_tool import TODO_INJECTION_HEADER

_FOLDED = "Keep going\n\n" + TODO_INJECTION_HEADER + "\n- [>] t1. Prepare release (in_progress)"


@pytest.fixture
def registry(monkeypatch):
    import tools.process_registry as processes

    registry = ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    yield registry
    registry.kill_all()


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr(cc, "get_model_context_length", lambda *a, **k: 100_000)
    monkeypatch.setattr(cc, "call_llm", lambda **kw: {"choices": [{
        "message": {"content": "## Goal\nShip the release."}, "finish_reason": "stop"}]})
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        built = run_agent.AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                                  quiet_mode=True, skip_context_files=True, skip_memory=True)
    compressor = cc.ContextCompressor(model="test/model", config_context_length=100_000, protect_first_n=0,
                                      protect_last_n=3, quiet_mode=True, tail_mode="lean")
    compressor.tail_token_budget = 50
    compressor.bind_session_state(session_id=built.session_id)
    built.context_compressor = compressor
    built.compression_in_place = False
    return built


def _rows(first_user: str) -> list[dict]:
    rows = [{"role": "user", "content": first_user}]
    for i in range(8):
        rows += [{"role": "assistant", "content": f"Progress {i} recorded. " + "x" * 300},
                 {"role": "user", "content": f"Record progress {i}. " + "y" * 300}]
    return rows + [{"role": "assistant", "content": "Done."}]


def _spawn(task_id, tmp_path):
    command = shlex.quote(sys.executable) + " -c " + shlex.quote("import time; time.sleep(60)")
    return json.loads(terminal_tool(command, background=True, task_id=task_id, workdir=str(tmp_path)))["session_id"]


def _compact(agent, rows):
    # The gateway compaction paths hydrate the fresh agent's store first; that makes the store authoritative.
    hydrate_todos_for_compaction(agent, rows)
    assert agent._todo_store.read() == [{"id": "t1", "content": "Prepare release", "status": "in_progress"}]
    compressed, _ = agent._compress_context(rows, "sys", approx_tokens=120_000, force=True)
    return compressed


def _assert_appendix_intact(compressed, process_id):
    summary = next(m["content"] for m in compressed if cc._LEAN_USER_MESSAGES_HEADING in str(m.get("content") or ""))
    quotes = summary.split(cc._LEAN_USER_MESSAGES_HEADING, 1)[1].split("\n## ", 1)[0]
    assert "> Keep going" in quotes
    assert TODO_INJECTION_HEADER not in summary
    assert _LEAN_BACKGROUND_HEADING in summary and process_id in summary
    assert cc._LEAN_RECOVERY_HEADING in summary
    assert cc._SUMMARY_END_MARKER in summary
    # The live list is folded once, after the summary, and still names the task.
    text = "\n".join(str(m.get("content") or "") for m in compressed)
    assert text.count(TODO_INJECTION_HEADER) == 1
    assert "Prepare release" in text.split(TODO_INJECTION_HEADER, 1)[1]


def test_folded_snapshot_in_compacted_user_row_keeps_appendix(agent, registry, tmp_path):
    process_id = _spawn(agent.session_id, tmp_path)
    _assert_appendix_intact(_compact(agent, _rows(_FOLDED)), process_id)


def test_carried_quote_with_folded_snapshot_keeps_appendix(agent, registry, tmp_path):
    # The previous handoff already quoted the folded row with its scaffolding; carrying that quote forward
    # must drop the scaffolding too. A later snapshot row (the newest) keeps the store authoritative.
    process_id = _spawn(agent.session_id, tmp_path)
    handoff = (cc.SUMMARY_PREFIX + "\n## Goal\nShip the release.\n\n" + cc._LEAN_USER_MESSAGES_HEADING + "\n> "
               + _FOLDED.replace("\n", "\n> ") + "\n\n" + cc._LEAN_RECOVERY_HEADING + "\nSearch session history.")
    rows = [{"role": "user", "content": handoff}, *_rows("Plan the release.")[1:]]
    rows[-2] = {"role": "user", "content": "Carry on\n\n" + _FOLDED.split("\n\n", 1)[1]}
    _assert_appendix_intact(_compact(agent, rows), process_id)
