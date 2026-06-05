from __future__ import annotations

import json

from scripts.self_improvement import summarize as summary


def test_summarize_reports_latest_task_and_context_contributors(tmp_path):
    root = tmp_path / "self-improvement-log"
    root.mkdir()
    (root / "task_runs.jsonl").write_text(
        json.dumps(
            {
                "kind": "task_run_telemetry",
                "session_id": "s1",
                "input_tokens": 250000,
                "output_tokens": 1000,
                "cache_read_tokens": 3000000,
                "api_call_count": 41,
                "tool_stats": ["skill_view:2", "cronjob:3", "terminal:1"],
                "largest_context_items": [
                    "tool:skill_view:48165",
                    "tool:skill_view:48165",
                    "tool:cronjob:13463",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "memory_context_audit.jsonl").write_text(
        json.dumps(
            {
                "kind": "memory_context_audit",
                "candidate_count": 3,
                "candidates": [
                    {"reasons": ["raw_user_fragment"], "content": "[USER] proceed"},
                    {"reasons": ["background_process_fragment"], "content": "[USER] startup"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = summary.build_summary(root)

    assert result["task_runs"] == 1
    assert result["latest_task_session"] == "s1"
    assert result["telemetry_totals"]["input_tokens"] == 250000
    assert result["review_flags"] == [
        "high_input_tokens",
        "high_cache_read_tokens",
        "high_api_calls",
        "duplicate_skill_view",
        "repeated_cronjob_list",
        "memory_context_noise",
    ]
    assert result["largest_context_items"][0] == "tool:skill_view:48165"
    assert result["memory_context"]["candidate_count"] == 3


def test_main_prints_json_summary(tmp_path, capsys):
    root = tmp_path / "empty-log"
    root.mkdir()

    rc = summary.main(["--root", str(root)])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task_runs"] == 0
    assert payload["review_flags"] == []


def test_read_jsonl_ignores_bad_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"ok": true}\nnot json\n{"ok": false}\n', encoding="utf-8")

    assert summary.read_jsonl(path) == [{"ok": True}, {"ok": False}]
