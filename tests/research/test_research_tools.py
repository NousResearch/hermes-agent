"""Regression tests for research tool integrations."""

from __future__ import annotations

import json
from pathlib import Path

from research.state_store import ResearchStateStore
from tools.dataset_tool import _get_auxiliary_client as get_dataset_aux_client
from tools.evaluation_tool import evaluation_tool
from tools.judge_tool import _get_auxiliary_client as get_judge_aux_client
from tools.research_loop_tool import research_loop_tool
from tools.tinker_posttrain_tool import tinker_posttrain_tool


def test_dataset_tool_uses_agent_auxiliary_client(monkeypatch):
    sentinel = object()

    def fake_get_text_auxiliary_client(task: str = ""):
        assert task == "dataset_synthesis"
        return sentinel, "fake-model"

    monkeypatch.setattr("agent.auxiliary_client.get_text_auxiliary_client", fake_get_text_auxiliary_client)

    assert get_dataset_aux_client("dataset_synthesis") is sentinel


def test_judge_tool_uses_agent_auxiliary_client(monkeypatch):
    sentinel = object()

    def fake_get_text_auxiliary_client(task: str = ""):
        assert task == "judge"
        return sentinel, "fake-model"

    monkeypatch.setattr("agent.auxiliary_client.get_text_auxiliary_client", fake_get_text_auxiliary_client)

    assert get_judge_aux_client("judge") is sentinel


def test_evaluation_tool_honors_backend_and_persists_experiment_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Eval Project", brief="evaluate a checkpoint")
    runtime_dir = store.layout(project.project_id)["runtime"]
    captured = {}

    class FakeProc:
        returncode = 0

        def communicate(self, timeout=None):
            payload = {
                "results": {
                    "hellaswag": {
                        "acc,none": 0.42,
                    }
                }
            }
            return json.dumps(payload), ""

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr("tools.evaluation_tool.subprocess.Popen", fake_popen)

    result = json.loads(
        evaluation_tool(
            action="run_eval",
            cwd=str(tmp_path),
            project_id=project.project_id,
            run_id="run_eval_test",
            experiment_id="exp_eval_test",
            checkpoint_path="/tmp/checkpoint",
            benchmarks=["hellaswag"],
            model_backend="vllm",
        )
    )

    assert result["success"] is True
    assert "--model" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--model") + 1] == "vllm"
    assert str(runtime_dir / f"eval_{result['results'][0]['eval_id']}") in captured["cmd"]

    saved = store.load_evaluations(project.project_id, run_id="run_eval_test")
    assert saved[0]["experiment_id"] == "exp_eval_test"


def test_hyperparam_queue_feeds_start_run_and_monitor_cleanup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"prompt":"hi","completion":"hello"}\n', encoding="utf-8")
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Search Project", brief="run a search")

    search = json.loads(
        research_loop_tool(
            action="hyperparam_search",
            cwd=str(tmp_path),
            project_id=project.project_id,
            strategy="grid",
            param_grid={"learning_rate": [1e-4, 2e-4]},
            base_config={
                "method": "sft",
                "base_model": "meta-llama/Llama-3.1-8B",
                "dataset_path": str(dataset),
            },
        )
    )

    assert search["queued"] == 2

    class FakeProc:
        pid = 34567

    def fake_launch(paths, cwd):
        paths.status_path.write_text(
            json.dumps({"status": "running", "step": 0, "run_id": Path(paths.runtime_dir).name}),
            encoding="utf-8",
        )
        return FakeProc()

    monkeypatch.setattr("tools.tinker_posttrain_tool.launch_background_run", fake_launch)
    monkeypatch.setattr("tools.tinker_posttrain_tool._schedule_monitor", lambda *args, **kwargs: {"id": "job_monitor"})
    monkeypatch.setattr("tools.tinker_posttrain_tool._pid_is_alive", lambda pid: True)

    started = json.loads(
        tinker_posttrain_tool(
            action="start_run",
            cwd=str(tmp_path),
            project_id=project.project_id,
            mode="auto",
        )
    )

    assert started["success"] is True
    assert started["run"]["metadata"]["search_index"] == 0

    loop_state = store.load_loop_state(project.project_id)
    assert len(loop_state["hyperparam_queue"]) == 1
    assert loop_state["hyperparam_active"] == [{"run_id": started["run"]["run_id"], "search_index": 0}]

    runtime_status = Path(started["run"]["status_path"])
    runtime_status.write_text(json.dumps({"status": "completed", "step": 1}), encoding="utf-8")
    monkeypatch.setattr("tools.research_loop_tool._pid_is_alive", lambda pid: False)
    monkeypatch.setattr("tools.research_loop_tool._schedule_prompt", lambda *args, **kwargs: {"id": "job_followup"})

    monitored = json.loads(
        research_loop_tool(
            action="monitor_run",
            cwd=str(tmp_path),
            project_id=project.project_id,
            run_id=started["run"]["run_id"],
        )
    )

    assert monitored["success"] is True
    assert monitored["job"]["id"] == "job_followup"

    loop_state = store.load_loop_state(project.project_id)
    assert loop_state["hyperparam_active"] == []
    assert len(loop_state["hyperparam_queue"]) == 1
    assert started["run"]["run_id"] not in store.load_project(project.project_id).active_run_ids
