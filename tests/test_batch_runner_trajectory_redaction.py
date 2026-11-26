"""batch_runner writes trajectories to the exported dataset; secrets must be redacted (#108495)."""

import json

from batch_runner import BatchRunner, _process_batch_worker

SECRET = "sk-" + "proj-batchrunner-secret-1234567890"


def test_batch_worker_redacts_trajectory_before_append(tmp_path, monkeypatch):
    prompt = f"use api_key={SECRET}"
    result = {
        "success": True,
        "trajectory": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": '<tool_call>{"name":"terminal","arguments":{"command":"curl -H \'Authorization: Bearer ' + SECRET + '\'"}}</tool_call>'},
        ],
        "reasoning_stats": {"has_any_reasoning": True},
        "tool_stats": {},
        "metadata": {},
        "completed": True,
        "api_calls": 1,
        "toolsets_used": [],
    }
    monkeypatch.setattr("batch_runner._process_single_prompt", lambda *a, **kw: result)

    _process_batch_worker((1, [(0, {"prompt": prompt})], tmp_path, set(), {"verbose": False}))

    raw = (tmp_path / "batch_1.jsonl").read_text(encoding="utf-8")
    assert SECRET not in raw
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert [t["from"] for t in rows[0]["conversations"]] == ["human", "gpt"]


def test_resume_still_skips_prompt_saved_in_redacted_form(tmp_path, monkeypatch):
    prompt = f"use api_key={SECRET}"
    runner = BatchRunner.__new__(BatchRunner)
    runner.output_dir = tmp_path
    runner.dataset = [{"prompt": prompt}, {"prompt": "other"}]
    result = {
        "success": True,
        "trajectory": [{"from": "human", "value": prompt}],
        "reasoning_stats": {"has_any_reasoning": True},
        "tool_stats": {}, "metadata": {}, "completed": True, "api_calls": 1, "toolsets_used": [],
    }
    monkeypatch.setattr("batch_runner._process_single_prompt", lambda *a, **kw: result)
    _process_batch_worker((1, [(0, runner.dataset[0])], tmp_path, set(), {"verbose": False}))

    remaining, skipped = runner._filter_dataset_by_completed(runner._scan_completed_prompts_by_content())

    assert skipped == [0]
    assert [idx for idx, _ in remaining] == [1]
