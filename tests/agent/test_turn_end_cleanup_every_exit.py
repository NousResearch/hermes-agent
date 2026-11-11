"""Every ``run_conversation()`` exit saves the trajectory and releases the turn's task resources.

Through a real ``AIAgent`` + ``SessionDB`` against the loopback model: with trajectory saving on, a
normal answer is appended to ``trajectory_samples.jsonl`` and every failed exit (a provider error no
fallback can fix, retry exhaustion, a content-policy refusal, a reply capped by the output limit on
every continuation, three calls to an unknown tool) to ``failed_trajectories.jsonl``, once. The
isolated terminal environment a rollout task opened (``batch_runner`` registers one per task) is torn
down when the turn then fails on a provider error.
"""

import json
import os
from pathlib import Path

import pytest

from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text, ToolCall, write_hermes_home

_TASK_ID = "task-turn-end"

_EXTRA_CONFIG = """updates:
  check: false
models_dev:
  url: http://127.0.0.1:9/api.json
skills:
  creation_nudge_interval: 0
memory:
  nudge_interval: 0
"""

# exit kind -> (script, expected ``completed``)
EXITS = {
    "text_answer": ([Text("all done")], True),
    "non_retryable_provider_error": ([Error(status=400, message="bad request")], False),
    "retries_exhausted": ([Error(status=500, message="upstream down")] * 2, False),
    "content_policy_refusal": ([Text("", finish_reason="content_filter")], False),
    "output_limit_on_every_continuation": ([Text("partial answer", finish_reason="length")] * 4, False),
    "unknown_tool_three_times": ([ToolCall("no_such_tool", {})] * 3, False),
}


@pytest.fixture
def run_turn(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.terminal_tool_lifecycle import cleanup_vm

    home = Path(os.environ["HERMES_HOME"])
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)  # trajectory files are written to the working directory
    opened = []

    def run(script):
        server = FakeLLMServer(script)
        server.__enter__()
        opened.append(lambda: server.__exit__(None, None, None))
        write_hermes_home(home, server.base_url, extra_config=_EXTRA_CONFIG)
        config_path = home / "config.yaml"
        # No transient-outage recovery cycles: the ladder would wait between attempts.
        config_path.write_text(config_path.read_text(encoding="utf-8").replace(
            "  api_max_retries: 1\n", "  api_max_retries: 1\n  auto_recovery_cycles: 0\n", 1,
        ), encoding="utf-8")
        db = SessionDB(db_path=tmp_path / "state.db")
        opened.append(db.close)
        agent = AIAgent(base_url=server.base_url, api_key="***", provider="custom", model="fake-model",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=True,
                        session_id="turn-end-exit", session_db=db)
        return agent.run_conversation("do the thing", task_id=_TASK_ID)

    yield run, workdir
    cleanup_vm(_TASK_ID)
    for close in opened:
        close()


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


@pytest.mark.parametrize("exit_kind", list(EXITS))
def test_every_turn_exit_saves_one_trajectory(run_turn, exit_kind):
    run, workdir = run_turn
    script, completed = EXITS[exit_kind]
    result = run(script)
    assert result["completed"] is completed
    saved = _records(workdir / ("trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"))
    other = _records(workdir / ("failed_trajectories.jsonl" if completed else "trajectory_samples.jsonl"))
    assert len(saved) == 1, f"{exit_kind}: {len(saved)} trajectory records; result={result.get('failure_reason')!r}"
    assert other == []
    [record] = saved
    assert record["completed"] is completed
    assert {"from": "human", "value": "do the thing"} in record["conversations"]


@pytest.fixture
def isolated_task_env():
    from tools.terminal_tool import clear_task_env_overrides, register_task_env_overrides

    register_task_env_overrides(_TASK_ID, {"env_type": "local"})
    yield
    clear_task_env_overrides(_TASK_ID)


def test_a_failed_turn_releases_its_terminal_environment(run_turn, isolated_task_env):
    from tools.terminal_tool_lifecycle import get_active_env

    run, _ = run_turn
    result = run([ToolCall("terminal", {"command": "echo ran-in-task-env"}), Error(status=400, message="bad request")])
    assert result["failed"] is True
    assert any(m["role"] == "tool" and "ran-in-task-env" in m["content"] for m in result["messages"])
    assert get_active_env(_TASK_ID) is None
