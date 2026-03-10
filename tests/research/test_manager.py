"""Tests for research.manager and research_manager_tool."""

from __future__ import annotations

import json

from research.manager import assess_dataset_rows, rank_runs, suggest_next_step, triage_literature
from research.models import ResearchProject
from research.state_store import ResearchStateStore
from tools.research_manager_tool import research_manager_tool


def test_triage_literature_scores_and_focuses():
    summary = triage_literature(
        [
            {
                "paper_id": "p1",
                "title": "Reasoning post-training with verifiable rewards",
                "abstract": "Train reasoning models with verifiable rewards and strong evaluation.",
                "year": 2025,
                "citation_count": 20,
            },
            {
                "paper_id": "p2",
                "title": "Classic instruction tuning baseline",
                "abstract": "A strong baseline for instruction tuning.",
                "year": 2021,
                "citation_count": 1200,
            },
        ],
        objective="Improve reasoning post-training with verifiable reward signals",
    )

    assert summary["paper_count"] == 2
    assert summary["ranked_papers"][0]["paper_id"] == "p1"
    assert summary["focus_areas"]


def test_assess_dataset_rows_flags_duplicates():
    report = assess_dataset_rows(
        [
            {"prompt": "a", "completion": "b"},
            {"prompt": "a", "completion": "b"},
            {"prompt": "c", "completion": "d"},
        ],
        method="sft",
    )
    assert report["num_rows"] == 3
    assert report["duplicate_rows"] == 1
    assert report["quality_score"] > 0


def test_rank_runs_detects_regressions():
    rankings = rank_runs(
        runs=[
            {"run_id": "run_base", "status": "completed", "created_at": "2026-01-01T00:00:00+00:00"},
            {"run_id": "run_new", "status": "completed", "created_at": "2026-01-02T00:00:00+00:00"},
        ],
        eval_results=[
            {"run_id": "run_base", "metrics": {"hellaswag/acc,none": 0.70, "eval/loss": 1.1}},
            {"run_id": "run_new", "metrics": {"hellaswag/acc,none": 0.69, "eval/loss": 1.0}},
        ],
    )
    by_run = {row["run_id"]: row for row in rankings["rankings"]}
    assert rankings["baseline_run_id"] == "run_base"
    assert any(item["metric"] == "hellaswag/acc,none" for item in by_run["run_new"]["regressions"])


def test_suggest_next_step_recommends_eval_when_completed_runs_exist():
    project = ResearchProject(project_id="proj_test", name="Test", cwd="/tmp", workspace_dir="/tmp/ws")
    decision = suggest_next_step(
        project=project,
        status_summary={
            "literature_notes": 3,
            "hypotheses": 1,
            "experiments": 1,
            "active_run_ids": [],
            "loop_state": {},
        },
        runs=[{"run_id": "run_done", "status": "completed"}],
        eval_results=[],
        datasets=[],
    )
    assert decision["next_action"] == "evaluate_completed_runs"


def test_research_manager_tool_prunes_queue_and_writes_memo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Manager Project", brief="improve reasoning with rewards")
    store.append_jsonl(
        project.project_id,
        "literature",
        {
            "paper_id": "p1",
            "title": "Reasoning post-training with verifiable rewards",
            "abstract": "Train reasoning models with verifiable rewards.",
            "year": 2025,
            "citation_count": 15,
        },
    )
    store.save_run(
        {
            "project_id": project.project_id,
            "run_id": "run_good",
            "status": "completed",
            "method": "sft",
            "base_model": "llama",
            "dataset_path": "/tmp/data.jsonl",
            "learning_rate": 1e-4,
            "batch_size": 8,
            "lora_rank": 16,
        }
    )
    store.append_jsonl(
        project.project_id,
        "evaluations",
        {
            "eval_id": "eval_1",
            "project_id": project.project_id,
            "run_id": "run_good",
            "experiment_id": "exp_1",
            "benchmark": "hellaswag",
            "metrics": {"hellaswag/acc,none": 0.74},
            "status": "completed",
        },
    )
    loop_state = store.load_loop_state(project.project_id)
    loop_state["hyperparam_queue"] = [
        {"method": "sft", "base_model": "llama", "dataset_path": "/tmp/data.jsonl", "learning_rate": 1e-4, "batch_size": 8, "lora_rank": 16},
        {"method": "sft", "base_model": "llama", "dataset_path": "/tmp/data.jsonl", "learning_rate": 2e-4, "batch_size": 4, "lora_rank": 16},
    ]
    store.save_loop_state(project.project_id, loop_state)

    pruned = json.loads(
        research_manager_tool(
            action="prune_search",
            cwd=str(tmp_path),
            project_id=project.project_id,
            max_queue=1,
        )
    )
    assert pruned["success"] is True
    assert pruned["kept"] == 1
    assert pruned["dropped"] == 1

    memo = json.loads(
        research_manager_tool(
            action="write_research_memo",
            cwd=str(tmp_path),
            project_id=project.project_id,
        )
    )
    assert memo["success"] is True
    assert "path" in memo
