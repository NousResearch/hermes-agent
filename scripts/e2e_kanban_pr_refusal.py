"""Observe a real GitHub PR refusal through an isolated native task lifecycle.

No worker or model is started. The existing gh identity makes read-only requests;
the caller must separately authorize and account for each diagnostic attempt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-url", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--expected-stage", required=True, choices=[
        "pr_policy", "branch_rules", "check_runs", "commit_statuses", "head_recheck",
    ])
    parser.add_argument("--expected-http-status", required=True, type=int, choices=[401, 403, 404])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    checks = {}
    report = {"journey": "native-pr-read-refusal", "status": "failed", "checks": checks,
              "source_sha256": hashlib.sha256((repo / "hermes_cli/kanban_pr_acceptance.py").read_bytes()).hexdigest(),
              "model_calls": 0, "worker_processes": 0, "provider_writes": 0}
    private = tempfile.TemporaryDirectory(prefix="hermes-pr-refusal-e2e-")
    root = Path(private.name)
    connection = None
    try:
        # Scope only this explicitly authorized disposable board, never the active task.
        for key in list(os.environ):
            if key.startswith("HERMES_KANBAN_"):
                os.environ.pop(key)
        os.environ.update(HERMES_HOME=str(root), HERMES_KANBAN_HOME=str(root),
                          HERMES_MANAGED_DIR=str(root / "managed"))
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        connection = kbc.connect()
        assert kb.kanban_db_path().resolve().is_relative_to(root)
        task_id = kb.create_task(connection, title="Observe GitHub acceptance read prerequisite",
            created_by="authorized-isolated-e2e", assignee=None, workspace_kind="scratch",
            project_id="", initial_status="running", completion_contract=args.pr_url,
            max_runtime_seconds=60)
        task = kb.claim_task(connection, task_id, ttl_seconds=60, claimer="isolated-e2e-owner")
        assert task is not None and task.current_run_id is not None
        select = "SELECT status,current_run_id,max_runtime_seconds,worker_pid,completion_contract FROM tasks WHERE id=?"
        before = tuple(connection.execute(select, (task_id,)).fetchone())
        completed = kb.complete_task(connection, task_id, summary="Read-only PR prerequisite observation",
            metadata={"published_pr": args.pr_url}, expected_run_id=task.current_run_id,
            fire_lifecycle_hook=False)
        row = connection.execute("SELECT run_id,substr(payload,1,65537) AS payload FROM task_events "
            "WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
        assert row is not None and row["run_id"] == task.current_run_id
        assert isinstance(row["payload"], str) and len(row["payload"]) <= 65536
        receipt = json.loads(row["payload"])
        assert not completed and receipt["ok"] is False and receipt["classification"] == "auth"
        assert receipt["pr_url"] == args.pr_url and receipt["head_sha"] == args.expected_sha
        checks["real_required_evidence_refusal_retains_unaccepted_task"] = "passed"
        provenance = receipt.get("failure_provenance")
        assert provenance == {"stage": args.expected_stage, "http_status": args.expected_http_status,
                              "credential_scope": "ambient"}
        report["failure_provenance"] = provenance
        checks["native_receipt_identifies_exact_denied_read_without_raw_text"] = "passed"
        assert tuple(connection.execute(select, (task_id,)).fetchone()) == before
        assert before[1] == task.current_run_id and before[2] == 60 and before[3] is None
        run = connection.execute("SELECT max_runtime_seconds,ended_at FROM task_runs WHERE id=?",
                                 (task.current_run_id,)).fetchone()
        assert run["max_runtime_seconds"] == 60 and run["ended_at"] is None
        checks["original_run_identity_and_limit_unchanged_no_worker"] = "passed"
        assert kb.archive_task(connection, task_id)
        assert kb.get_task(connection, task_id).status == "archived"
        checks["native_archive_cancels_disposable_task"] = "passed"
        report["status"] = "passed"
        return 0
    finally:
        if connection is not None:
            connection.close()
        private.cleanup()
        report["cleanup"] = {"private_native_store_removed": not root.exists()}
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
