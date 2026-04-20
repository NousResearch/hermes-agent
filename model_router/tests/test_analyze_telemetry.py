import json
import subprocess
import sys
from pathlib import Path

from analyze_telemetry import summarize


ROOT = Path(__file__).resolve().parent.parent


def test_summarize_counts_decisions_feedback_and_joined_stats():
    events = [
        {
            "timestamp": "2026-04-20T10:00:00+00:00",
            "event_type": "decision",
            "request_id": "r1",
            "input": {"task_type": "coding", "priority": "high", "quota": "normal"},
            "decision": {
                "primary_model": "gpt-5.4",
                "reviewer": "claude-sonnet-4.6",
                "trace": ["normalize: has_code/has_logs -> coding"],
            },
        },
        {
            "timestamp": "2026-04-20T10:01:00+00:00",
            "event_type": "feedback",
            "request_id": "r1",
            "outcome": "success",
            "fallback_used": False,
            "actual_model_used": "gpt-5.4",
            "user_rating": 5,
        },
        {
            "timestamp": "2026-04-20T10:02:00+00:00",
            "event_type": "decision",
            "request_id": "r2",
            "input": {"task_type": "chat", "priority": "low", "quota": "critical"},
            "decision": {
                "primary_model": "deepseek",
                "reviewer": None,
                "trace": ["override: quota=critical + priority=low -> deepseek"],
            },
        },
        {
            "timestamp": "2026-04-20T10:03:00+00:00",
            "event_type": "feedback",
            "request_id": "r2",
            "outcome": "fallback_used",
            "fallback_used": True,
            "actual_model_used": "claude-sonnet-4.6",
            "user_rating": 3,
        },
    ]

    summary = summarize(events)

    assert summary["total_decisions"] == 2
    assert summary["total_feedback_events"] == 2
    assert summary["primary_models"]["gpt-5.4"] == 1
    assert summary["primary_models"]["deepseek"] == 1
    assert summary["feedback"]["outcomes"]["success"] == 1
    assert summary["feedback"]["outcomes"]["fallback_used"] == 1

    gpt_stats = summary["joined"]["by_primary_model"]["gpt-5.4"]
    assert gpt_stats["success_rate"] == 1.0
    assert gpt_stats["fallback_rate"] == 0.0
    assert gpt_stats["average_rating"] == 5.0

    deepseek_stats = summary["joined"]["by_primary_model"]["deepseek"]
    assert deepseek_stats["fallback_rate"] == 1.0
    assert deepseek_stats["mismatch_actual_model"] == 1


def test_analyze_cli_outputs_json(tmp_path: Path):
    log_path = tmp_path / "router.jsonl"
    rows = [
        {
            "timestamp": "2026-04-20T10:00:00+00:00",
            "event_type": "decision",
            "request_id": "1",
            "input": {"task_type": "coding", "priority": "high", "quota": "normal"},
            "decision": {"primary_model": "gpt-5.4", "reviewer": None, "trace": []},
        }
    ]
    log_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(ROOT / "analyze_telemetry.py"), str(log_path), "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["total_decisions"] == 1
    assert payload["primary_models"]["gpt-5.4"] == 1
