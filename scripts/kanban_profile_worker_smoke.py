#!/usr/bin/env python3
"""Smoke-test that a Hermes profile can run through the Kanban dispatcher path.

This creates an isolated temporary Kanban board, dispatches one scratch task to
`hermes -p <profile>`, waits for the worker to complete it, and prints a
non-secret summary. It intentionally does not read or print auth.json contents.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="blue", help="Hermes profile to spawn (default: blue)")
    parser.add_argument("--timeout", type=int, default=180, help="Seconds to wait for worker completion")
    parser.add_argument("--keep", action="store_true", help="Keep the temporary Kanban home for log inspection")
    parser.add_argument("--board", default="smoke", help="Temporary board slug (default: smoke)")
    return parser.parse_args()


def _prepare_kanban_env(tmp_home: Path, board: str) -> None:
    # Keep HERMES_HOME/profile config untouched so `hermes -p <profile>` exercises
    # the real profile/auth resolution path, but isolate the Kanban DB/logs/workspace.
    os.environ["HERMES_KANBAN_HOME"] = str(tmp_home)
    os.environ["HERMES_KANBAN_BOARD"] = board
    for key in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_KANBAN_WORKSPACE",
        "HERMES_TENANT",
    ):
        os.environ.pop(key, None)


def main() -> int:
    args = _parse_args()
    tmp_home = Path(tempfile.mkdtemp(prefix=f"hermes-kanban-{args.profile}-smoke."))
    _prepare_kanban_env(tmp_home, args.board)

    # Import after env isolation so module-level defaults cannot capture live board state.
    from hermes_cli import kanban_db as kb

    task_id = ""
    try:
        kb.init_db(board=args.board)
        with kb.connect() as conn:
            task_id = kb.create_task(
                conn,
                title=f"{args.profile} spawned worker credential smoke",
                body=(
                    "Smoke test only: call kanban_show, then kanban_complete "
                    "with summary 'profile worker smoke ok' and metadata "
                    "{'smoke': true}. Do not modify files."
                ),
                assignee=args.profile,
                created_by="smoke",
                workspace_kind="scratch",
                max_runtime_seconds=args.timeout,
                skills=["hermes-agent"],
            )
            result = kb.dispatch_once(conn, max_spawn=1, board=args.board)

        print(json.dumps({
            "event": "dispatched",
            "profile": args.profile,
            "task_id": task_id,
            "spawned": result.spawned,
            "crashed": result.crashed,
            "skipped_nonspawnable": result.skipped_nonspawnable,
            "kanban_home": str(tmp_home),
        }, ensure_ascii=False))

        deadline = time.time() + max(1, args.timeout)
        last = None
        while time.time() < deadline:
            time.sleep(5)
            with kb.connect() as conn:
                task = kb.get_task(conn, task_id)
                runs = kb.list_runs(conn, task_id)
            run = runs[-1] if runs else None
            last = {
                "status": task.status if task else None,
                "outcome": run.outcome if run else None,
                "summary": run.summary if run else None,
                "error": (run.error[:500] if run and run.error else None),
            }
            print(json.dumps({"event": "poll", **last}, ensure_ascii=False))
            if last["status"] in {"done", "blocked"} or last["outcome"] in {
                "completed", "blocked", "crashed", "timed_out", "spawn_failed", "gave_up"
            }:
                break

        log_path = kb.worker_log_path(task_id, board=args.board)
        log_tail = None
        if log_path.exists():
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        final = {
            "event": "final",
            "profile": args.profile,
            "task_id": task_id,
            "result": last,
            "log_path": str(log_path),
            "kept_kanban_home": str(tmp_home) if args.keep else None,
        }
        print(json.dumps(final, ensure_ascii=False))
        if log_tail:
            print("--- worker log tail (redacted by omission; no auth files printed) ---")
            print(log_tail)
            print("--- end worker log tail ---")

        ok = bool(last and last.get("status") == "done" and last.get("outcome") == "completed")
        return 0 if ok else 1
    finally:
        if args.keep:
            print(f"Kept temporary Kanban home: {tmp_home}")
        else:
            shutil.rmtree(tmp_home, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
