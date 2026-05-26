#!/usr/bin/env python3
"""Reproducible Phase 2A result-shaping benchmark.

Creates a disposable fixture, runs read_file/search_files through the normal
dispatcher with opt-in metrics enabled only for this process, and compares
default auto-shaped output to explicit result_mode='full' output.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _char_len(result: str) -> int:
    return len(result if isinstance(result, str) else str(result))


def _call(tool: str, args: dict, *, task_id: str, session_id: str) -> str:
    from model_tools import handle_function_call

    return handle_function_call(
        tool,
        args,
        task_id=task_id,
        session_id=session_id,
        tool_call_id=f"{tool}-{args.get('result_mode', 'auto')}-{args.get('offset', 0)}",
    )


def _build_fixture(root: Path) -> tuple[Path, Path]:
    large = root / "large_module.py"
    lines = []
    for i in range(1, 701):
        marker = "PHASE2A_NEEDLE" if i % 4 == 0 else "ordinary"
        lines.append(
            f"def function_{i:04d}():  # {marker}\n"
            f"    return 'line {i:04d} {'x' * 120}'\n"
        )
    large.write_text("".join(lines), encoding="utf-8")

    for file_idx in range(18):
        target = root / f"module_{file_idx:02d}.py"
        target.write_text(
            "\n".join(
                f"value_{line_idx} = 'PHASE2A_NEEDLE {'y' * 160}'"
                for line_idx in range(18)
            ),
            encoding="utf-8",
        )

    return large, root


def _audit_metrics(path: Path) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, ["metrics file was not created"]

    raw = path.read_text(encoding="utf-8")
    forbidden = [
        "PHASE2A_NEEDLE",
        "value_1 =",
        "function_0001",
        '"args"',
        '"arguments"',
        '"raw_args"',
        '"result"',
        '"content"',
        '"output"',
    ]
    failures = [token for token in forbidden if token in raw]

    for line_no, line in enumerate(raw.splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            failures.append(f"invalid json line {line_no}")
            continue
        if "sha256" not in row or "result_char_count" not in row:
            failures.append(f"missing metadata line {line_no}")

    return not failures, failures


def main() -> int:
    fixture_root = Path(tempfile.mkdtemp(prefix="hermes-phase2a-fixture-"))
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-phase2a-home-"))
    old_home = os.environ.get("HERMES_HOME")
    old_metrics = os.environ.get("HERMES_TOKEN_OPTIMIZATION_METRICS")
    old_disable = os.environ.get("HERMES_DISABLE_RESULT_COMPACTION")

    try:
        os.environ["HERMES_HOME"] = str(hermes_home)
        os.environ["HERMES_TOKEN_OPTIMIZATION_METRICS"] = "1"
        os.environ.pop("HERMES_DISABLE_RESULT_COMPACTION", None)
        large_file, search_root = _build_fixture(fixture_root)

        calls = [
            (
                "read_file",
                {"path": str(large_file), "limit": 500},
                {"path": str(large_file), "limit": 500, "result_mode": "full"},
            ),
            (
                "search_files",
                {
                    "pattern": "PHASE2A_NEEDLE",
                    "path": str(search_root),
                    "limit": 50,
                    "output_mode": "content",
                },
                {
                    "pattern": "PHASE2A_NEEDLE",
                    "path": str(search_root),
                    "limit": 50,
                    "output_mode": "content",
                    "result_mode": "full",
                },
            ),
        ]

        shaped_total = 0
        full_total = 0
        details = []
        for index, (tool, shaped_args, full_args) in enumerate(calls, start=1):
            shaped = _call(tool, shaped_args, task_id=f"phase2a-shaped-{index}", session_id="phase2a")
            full = _call(tool, full_args, task_id=f"phase2a-full-{index}", session_id="phase2a")
            shaped_len = _char_len(shaped)
            full_len = _char_len(full)
            shaped_total += shaped_len
            full_total += full_len
            details.append((tool, shaped_len, full_len))

        reduction = 0.0 if full_total == 0 else (full_total - shaped_total) / full_total * 100
        metrics_file = hermes_home / "token-optimization" / "metrics.jsonl"
        audit_ok, audit_failures = _audit_metrics(metrics_file)

        print("Phase 2A result-shaping benchmark")
        for tool, shaped_len, full_len in details:
            print(f"- {tool}: auto={shaped_len:,} chars full={full_len:,} chars")
        print(f"- combined: auto={shaped_total:,} chars full={full_total:,} chars")
        print(f"- reduction: {reduction:.2f}%")
        print(f"- metrics_file: {metrics_file}")
        print(f"- metrics_leakage_audit: {'PASS' if audit_ok else 'FAIL'}")
        if audit_failures:
            print("- audit_failures: " + ", ".join(audit_failures))

        return 0 if audit_ok and reduction >= 30.0 else 1
    finally:
        if old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = old_home
        if old_metrics is None:
            os.environ.pop("HERMES_TOKEN_OPTIMIZATION_METRICS", None)
        else:
            os.environ["HERMES_TOKEN_OPTIMIZATION_METRICS"] = old_metrics
        if old_disable is None:
            os.environ.pop("HERMES_DISABLE_RESULT_COMPACTION", None)
        else:
            os.environ["HERMES_DISABLE_RESULT_COMPACTION"] = old_disable
        shutil.rmtree(fixture_root, ignore_errors=True)
        shutil.rmtree(hermes_home, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
