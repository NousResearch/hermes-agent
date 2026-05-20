#!/usr/bin/env python3
"""Replay a JSONL file through tool result compaction.

This is a local, LLM-free benchmark for recorded tool-heavy sessions. Each JSONL
line should contain a string tool result in a configurable field (default:
`content`). Lines with missing or non-string content are skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

from agent.tool_result_compaction import compact_tool_content_if_needed, token_estimate


@dataclass
class ReplayAgent:
    session_id: str = "replay-session"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                continue
            yield item


def run_replay(
    *,
    input_path: Path,
    content_field: str,
    threshold_tokens: int,
    preview_chars: int,
    max_disk_mb: int,
    keep_dir: bool,
) -> dict[str, Any]:
    temp_dir = Path(tempfile.mkdtemp(prefix="hermes-tool-result-replay-"))
    cfg = {
        "enabled": True,
        "threshold_tokens": threshold_tokens,
        "preview_chars": preview_chars,
        "raw_result_dir": str(temp_dir),
        "max_disk_mb": max_disk_mb,
    }

    total_lines = 0
    string_results = 0
    compacted_count = 0
    before_tokens = 0
    after_tokens = 0

    try:
        with patch("agent.tool_result_compaction.get_config", return_value=cfg):
            for index, item in enumerate(_iter_jsonl(input_path)):
                total_lines += 1
                content = item.get(content_field)
                if not isinstance(content, str):
                    continue
                string_results += 1
                before_tokens += token_estimate(content)
                tool_name = str(item.get("tool_name") or item.get("name") or "tool")
                tool_call_id = str(item.get("tool_call_id") or f"line_{index}")
                compacted_content, info = compact_tool_content_if_needed(
                    tool_name,
                    tool_call_id,
                    content,
                    ReplayAgent(),
                )
                after_tokens += token_estimate(compacted_content)
                if info is not None:
                    compacted_count += 1

        raw_files = list(temp_dir.rglob("*.json"))
        raw_bytes = sum(path.stat().st_size for path in raw_files)
        saved_tokens = before_tokens - after_tokens
        saved_pct = (saved_tokens / before_tokens * 100.0) if before_tokens else 0.0
        return {
            "input_path": str(input_path),
            "content_field": content_field,
            "total_json_objects": total_lines,
            "string_results": string_results,
            "compacted_count": compacted_count,
            "before_tokens_estimate": before_tokens,
            "after_tokens_estimate": after_tokens,
            "saved_tokens_estimate": saved_tokens,
            "saved_percent_estimate": round(saved_pct, 2),
            "raw_file_count": len(raw_files),
            "raw_bytes": raw_bytes,
            "raw_result_dir": str(temp_dir),
        }
    finally:
        if not keep_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--content-field", default="content")
    parser.add_argument("--threshold-tokens", type=int, default=5_000)
    parser.add_argument("--preview-chars", type=int, default=1_000)
    parser.add_argument("--max-disk-mb", type=int, default=500)
    parser.add_argument("--keep-dir", action="store_true")
    args = parser.parse_args()

    result = run_replay(
        input_path=args.input_path,
        content_field=args.content_field,
        threshold_tokens=args.threshold_tokens,
        preview_chars=args.preview_chars,
        max_disk_mb=args.max_disk_mb,
        keep_dir=args.keep_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
