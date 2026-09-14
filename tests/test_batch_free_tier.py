"""A refused dataset row remains pending, never a completed/discarded trajectory."""
import json

import batch_runner


def test_refusal_preserves_input_without_checkpoint_or_automatic_retry(tmp_path, monkeypatch):
    notice = "Welcome allowance reached. Configure a provider, then resume this batch."
    calls = []

    class Agent:
        def __init__(self, **kwargs):
            pass

        def run_conversation(self, prompt, **kwargs):
            calls.append(prompt)
            return {"messages": [], "final_response": notice, "completed": False,
                    "failed": True, "partial": False, "error": "free_tier_limit",
                    "failure_reason": "free_tier_limit", "code": "free_tier_limit",
                    "retryable": True, "failure_retryable": False, "api_calls": 0}

        def _convert_to_trajectory_format(self, *args):
            raise AssertionError("A refusal is not a trajectory")

    monkeypatch.setattr(batch_runner, "AIAgent", Agent)
    monkeypatch.setattr(batch_runner, "sample_toolsets_from_distribution", lambda _: [])
    config = {"distribution": {}, "model": "welcome", "max_iterations": 2}
    prompt = {"prompt": "do work", "custom_input": {"keep": "exactly"}}
    result = batch_runner._process_single_prompt(0, prompt, 1, config)
    assert result["success"] is False
    assert result["failure_reason"] == "free_tier_limit"
    assert result["final_response"] == notice
    assert result["retryable"] is False
    calls.clear()
    summary = batch_runner._process_batch_worker((1, [(0, prompt), (1, {"prompt": "next"})], tmp_path, set(), config))
    assert calls == ["do work"]
    assert summary["completed_prompts"] == []
    assert summary["discarded_no_reasoning"] == 0
    assert summary["pending_prompts"] == [0, 1]
    assert not (tmp_path / "batch_1.jsonl").exists()
    pending = [json.loads(line) for line in (tmp_path / "pending_1.jsonl").read_text().splitlines()]
    assert pending[0]["prompt_data"] == prompt
    assert all(row["retryable_after_recovery"] and not row["retryable"] for row in pending)
    # The parent summary must not turn a paused worker into a successful finished run.
    dataset = tmp_path / "input.jsonl"
    dataset.write_text(json.dumps(prompt) + "\n")
    monkeypatch.chdir(tmp_path)
    runner = batch_runner.BatchRunner(dataset_file=str(dataset), batch_size=1, run_name="gate", num_workers=1)
    monkeypatch.setattr(runner, "_run_pool", lambda *args: [summary])
    runner.run()
    stats = json.loads(runner.stats_file.read_text())
    assert stats["status"] == "pending_recovery"
    assert stats["completed_at"] is None
    assert stats["pending_prompts"] == [0, 1]
    assert not runner._scan_completed_prompts_by_content()


def test_crossing_result_stays_successful_with_continuation_metadata(monkeypatch):
    class Agent:
        def __init__(self, **kwargs):
            pass

        def run_conversation(self, *args, **kwargs):
            return {"messages": [{"role": "assistant", "content": "finished"}],
                    "final_response": "finished\n\nConfigure a provider.", "completed": True,
                    "api_calls": 1, "free_tier": {"capped": True}, "free_tier_notice": "Configure a provider."}

        def _convert_to_trajectory_format(self, messages, *args):
            return messages

    monkeypatch.setattr(batch_runner, "AIAgent", Agent)
    monkeypatch.setattr(batch_runner, "sample_toolsets_from_distribution", lambda _: [])
    result = batch_runner._process_single_prompt(0, {"prompt": "finish"}, 1,
        {"distribution": {}, "model": "welcome", "max_iterations": 2})
    assert result["success"] and result["completed"]
    assert result["free_tier"]["capped"]
    assert result["final_response"].endswith(result["free_tier_notice"])
    assert result["trajectory"][-1]["content"] == "finished"
