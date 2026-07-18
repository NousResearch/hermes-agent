#!/usr/bin/env python3
"""Deterministic synthetic benchmark for Kanban worker-context profiles."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def run_benchmark() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hermes-kanban-context-bench-") as tmp:
        home = Path(tmp) / ".hermes"
        os.environ["HERMES_HOME"] = str(home)
        os.environ["HERMES_KANBAN_BOARD"] = "context-benchmark"

        from hermes_cli import kanban_db as kb

        with kb.connect() as conn:
            task_id = kb.create_task(
                conn,
                title="Synthetic broad lifecycle benchmark",
                body="BODY_MARKER\n" + ("B" * 8_000),
                assignee="benchmark-worker",
                granularity_policy="allow",
            )

            for index in range(20):
                parent_id = kb.create_task(
                    conn,
                    title=f"Synthetic parent {index}",
                    body="PARENT_BODY\n" + ("P" * 2_000),
                    assignee="benchmark-worker",
                    granularity_policy="allow",
                )
                conn.execute(
                    "UPDATE tasks SET status = 'done', result = ? WHERE id = ?",
                    (f"parent-result-{index}-" + ("R" * 5_000), parent_id),
                )
                conn.execute(
                    """INSERT INTO task_runs
                       (task_id, profile, status, started_at, ended_at, outcome,
                        summary, metadata, error)
                       VALUES (?, 'benchmark-worker', 'done', 1, 2, 'completed',
                               ?, ?, NULL)""",
                    (
                        parent_id,
                        f"parent-summary-{index}-" + ("S" * 6_000),
                        '{"blob":"' + ("M" * 6_000) + '"}',
                    ),
                )
                conn.execute(
                    "INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)",
                    (parent_id, task_id),
                )

            for index in range(8):
                conn.execute(
                    """INSERT INTO task_runs
                       (task_id, profile, status, started_at, ended_at, outcome,
                        summary, metadata, error)
                       VALUES (?, 'benchmark-worker', 'failed', ?, ?, 'timed_out',
                               ?, ?, ?)""",
                    (
                        task_id,
                        10 + index,
                        20 + index,
                        f"attempt-summary-{index}-" + ("A" * 7_000),
                        '{"blob":"' + ("D" * 7_000) + '"}',
                        f"attempt-error-{index}-" + ("E" * 7_000),
                    ),
                )

            for index in range(40):
                kb.add_comment(
                    conn,
                    task_id,
                    author="benchmark-worker",
                    body=f"comment-{index}-" + ("C" * 3_000),
                )
            conn.commit()

            full = kb.build_worker_context(conn, task_id, compact=False)
            compact = kb.build_worker_context(conn, task_id, compact=True)

        full_bytes = len(full.encode("utf-8"))
        compact_bytes = len(compact.encode("utf-8"))
        reduced_bytes = full_bytes - compact_bytes
        reduction_pct = round((reduced_bytes / full_bytes) * 100, 2)
        omissions = {
            "comments": 40 - kb._CTX_COMPACT_MAX_COMMENTS,
            "parents": 20 - kb._CTX_COMPACT_MAX_PARENTS,
            "prior_attempts": 8 - kb._CTX_COMPACT_MAX_PRIOR_ATTEMPTS,
        }
        recovery = {
            "cli": (
                "hermes kanban context <task_id> --run-id <run_id> "
                "--field partial_summary_full"
            ),
            "model_tool": "kanban_show(task_id=<task_id>, full_context=true)",
        }
        return {
            "benchmark": "kanban_worker_context_synthetic_v1",
            "fixture": {
                "parents": 20,
                "prior_attempts": 8,
                "comments": 40,
                "task_body_chars": 8_012,
            },
            "full": {
                "utf8_bytes": full_bytes,
                "approx_tokens_at_4_bytes_per_token": math.ceil(full_bytes / 4),
            },
            "compact": {
                "utf8_bytes": compact_bytes,
                "approx_tokens_at_4_bytes_per_token": math.ceil(compact_bytes / 4),
            },
            "savings": {
                "utf8_bytes": reduced_bytes,
                "approx_tokens_at_4_bytes_per_token": math.ceil(reduced_bytes / 4),
                "percent": reduction_pct,
            },
            "omissions": omissions,
            "recovery": recovery,
            "contracts": {
                "compact_under_48k_utf8": compact_bytes <= 48 * 1024,
                "compact_under_half_of_full": compact_bytes < full_bytes / 2,
                "cli_recovery_marker_present": "hermes kanban context" in compact,
                "model_tool_recovery_marker_present": "kanban_show" in compact,
                "parent_omission_marker_present": "12 earlier parents" in compact,
                "attempt_omission_marker_present": "7 earlier attempts" in compact,
                "comment_omission_marker_present": "34 earlier comments" in compact,
                "task_body_preserved": "BODY_MARKER" in compact,
            },
            "caveat": (
                "Token counts are a transparent 4-UTF-8-bytes/token approximation; "
                "provider tokenizers and cache behavior vary."
            ),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if all(report["contracts"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
