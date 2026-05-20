"""Tests for JSONL replay benchmark for tool result compaction."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts.replay_tool_result_compaction import run_replay


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_replay_compacts_large_string_results(tmp_path):
    input_path = tmp_path / "tool_results.jsonl"
    _write_jsonl(
        input_path,
        [
            {"tool_name": "terminal", "tool_call_id": "call_1", "content": "A" * 20_000},
            {"tool_name": "read_file", "tool_call_id": "call_2", "content": "small"},
            {"tool_name": "vision_analyze", "tool_call_id": "call_3", "content": ["not", "string"]},
        ],
    )

    result = run_replay(
        input_path=input_path,
        content_field="content",
        threshold_tokens=1_000,
        preview_chars=100,
        max_disk_mb=500,
        keep_dir=False,
    )

    assert result["total_json_objects"] == 3
    assert result["string_results"] == 2
    assert result["compacted_count"] == 1
    assert result["raw_file_count"] == 1
    assert result["before_tokens_estimate"] > result["after_tokens_estimate"]
    assert result["saved_percent_estimate"] > 0
    assert not Path(result["raw_result_dir"]).exists()


def test_replay_supports_custom_content_field(tmp_path):
    input_path = tmp_path / "tool_results.jsonl"
    _write_jsonl(
        input_path,
        [
            {"tool_name": "terminal", "tool_call_id": "call_1", "result": "A" * 20_000},
        ],
    )

    result = run_replay(
        input_path=input_path,
        content_field="result",
        threshold_tokens=1_000,
        preview_chars=100,
        max_disk_mb=500,
        keep_dir=False,
    )

    assert result["string_results"] == 1
    assert result["compacted_count"] == 1


def test_replay_keep_dir_preserves_raw_result_dir(tmp_path):
    input_path = tmp_path / "tool_results.jsonl"
    _write_jsonl(
        input_path,
        [
            {"tool_name": "terminal", "tool_call_id": "call_1", "content": "A" * 20_000},
        ],
    )

    result = run_replay(
        input_path=input_path,
        content_field="content",
        threshold_tokens=1_000,
        preview_chars=100,
        max_disk_mb=500,
        keep_dir=True,
    )

    raw_result_dir = Path(result["raw_result_dir"])
    try:
        assert raw_result_dir.exists()
        assert result["raw_file_count"] == 1
        assert list(raw_result_dir.rglob("*.json"))
    finally:
        shutil.rmtree(raw_result_dir, ignore_errors=True)
