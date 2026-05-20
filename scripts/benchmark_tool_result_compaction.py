#!/usr/bin/env python3
"""Synthetic benchmark for tool result compaction.

This script does not call an LLM. It estimates before/after token pressure for
large text tool results using the same chars/4 heuristic as the compaction
module, and it exercises the real compaction path against a temporary raw result
directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from agent.tool_result_compaction import compact_tool_content_if_needed, token_estimate


@dataclass
class DummyAgent:
    session_id: str = "benchmark-session"


def _build_payload(size_chars: int, index: int) -> str:
    header = f"tool_result[{index}]\n"
    footer = f"\nend_tool_result[{index}]"
    body_size = max(0, size_chars - len(header) - len(footer))
    line = f"line {index}: synthetic terminal/read_file output for compaction benchmark\n"
    repeats = (body_size // len(line)) + 1
    body = (line * repeats)[:body_size]
    return header + body + footer


def run_benchmark(
    *,
    tool_results: int,
    result_chars: int,
    threshold_tokens: int,
    preview_chars: int,
    max_disk_mb: int,
    keep_dir: bool,
) -> dict[str, Any]:
    temp_dir = Path(tempfile.mkdtemp(prefix="hermes-tool-result-compaction-"))
    cfg = {
        "enabled": True,
        "threshold_tokens": threshold_tokens,
        "preview_chars": preview_chars,
        "raw_result_dir": str(temp_dir),
        "max_disk_mb": max_disk_mb,
    }

    before_tokens = 0
    after_tokens = 0
    compacted_count = 0
    raw_paths: list[str] = []

    try:
        with patch("agent.tool_result_compaction.get_config", return_value=cfg):
            for index in range(tool_results):
                content = _build_payload(result_chars, index)
                before_tokens += token_estimate(content)
                compacted_content, info = compact_tool_content_if_needed(
                    "benchmark_tool",
                    f"call_{index}",
                    content,
                    DummyAgent(),
                )
                after_tokens += token_estimate(compacted_content)
                if info is not None:
                    compacted_count += 1
                    raw_paths.append(info["raw_result_path"])

        raw_files = list(temp_dir.rglob("*.json"))
        raw_bytes = sum(path.stat().st_size for path in raw_files)
        saved_tokens = before_tokens - after_tokens
        saved_pct = (saved_tokens / before_tokens * 100.0) if before_tokens else 0.0
        return {
            "tool_results": tool_results,
            "result_chars_each": result_chars,
            "threshold_tokens": threshold_tokens,
            "preview_chars": preview_chars,
            "before_tokens_estimate": before_tokens,
            "after_tokens_estimate": after_tokens,
            "saved_tokens_estimate": saved_tokens,
            "saved_percent_estimate": round(saved_pct, 2),
            "compacted_count": compacted_count,
            "raw_file_count": len(raw_files),
            "raw_bytes": raw_bytes,
            "raw_result_dir": str(temp_dir),
            "sample_raw_path": raw_paths[0] if raw_paths else None,
        }
    finally:
        if not keep_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-results", type=int, default=10)
    parser.add_argument("--result-chars", type=int, default=50_000)
    parser.add_argument("--threshold-tokens", type=int, default=5_000)
    parser.add_argument("--preview-chars", type=int, default=1_000)
    parser.add_argument("--max-disk-mb", type=int, default=500)
    parser.add_argument("--keep-dir", action="store_true")
    args = parser.parse_args()

    result = run_benchmark(
        tool_results=args.tool_results,
        result_chars=args.result_chars,
        threshold_tokens=args.threshold_tokens,
        preview_chars=args.preview_chars,
        max_disk_mb=args.max_disk_mb,
        keep_dir=args.keep_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
