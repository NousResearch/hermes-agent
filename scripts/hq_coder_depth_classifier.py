"""HQ Coder Depth Classifier (MVP)

Heuristic MVP only. Not live enforcement. Intended for dry-run/build-packet use.
"""
from __future__ import annotations
import json
import os
import tempfile
import textwrap
from pathlib import Path


def classify(tool_calls: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for call in tool_calls:
        name = call.get("name") or call.get("tool_name") or "unknown"
        counts[name] = counts.get(name, 0) + 1
    tool_count = sum(counts.values())
    edit_run_test = bool(
        counts.get("file_write") or counts.get("edit")
    ) and bool(
        counts.get("run_command") or counts.get("execute_command") or counts.get("terminal")
    )
    multi_shell = counts.get("terminal", 0) + counts.get("run_command", 0) + counts.get("execute_command", 0)
    result = {
        "tool_count": tool_count,
        "edit_run_test": edit_run_test,
        "multi_shell": multi_shell,
        "tool_breakdown": counts,
        "depth": "low"
        if tool_count <= 3
        else ("medium" if tool_count <= 12 else "high"),
    }
    return result


def main() -> int:
    payload_path = Path(
        os.environ.get("HQ_CODER_DEPTH_CLASSIFIER_INPUT")
        or os.path.join(tempfile.gettempdir(), "hq_coder_depth_input.json")
    )
    out_path = Path(
        os.environ.get("HQ_CODER_DEPTH_CLASSIFIER_OUTPUT")
        or os.path.join(tempfile.gettempdir(), "hq_coder_depth_output.json")
    )
    if not payload_path.exists():
        raise SystemExit(f"missing input: {payload_path}")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    calls = payload.get("tool_calls", []) if isinstance(payload, dict) else []
    out_path.write_text(json.dumps(classify(calls), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
