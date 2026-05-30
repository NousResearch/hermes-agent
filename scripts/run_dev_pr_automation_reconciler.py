#!/usr/bin/env python3.11
"""Run one polling pass of Hermes Dev GitHub PR automation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.dev_control.github_pr_automation import (  # noqa: E402
    DevGitHubPRAutomationStore,
    managed_repos,
    reconcile_github_pr_automation,
)
from gateway.dev_control.scm_lifecycle import DevSCMLifecycleStore  # noqa: E402
from hermes_state import DEFAULT_DB_PATH  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll GitHub PRs and run trusted-label Hermes PR automation.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument(
        "--repos",
        default=os.getenv("HERMES_DEV_PR_AUTOMATION_REPOS", ",".join(sorted(managed_repos()))),
        help="Comma-separated owner/name repos to reconcile.",
    )
    parser.add_argument("--limit", type=int, default=int(os.getenv("HERMES_DEV_PR_AUTOMATION_PR_LIMIT", "50")))
    parser.add_argument(
        "--lock-path",
        default=os.getenv("HERMES_DEV_PR_AUTOMATION_LOCK", "/tmp/hermes_dev_pr_automation_reconciler.lock"),
    )
    args = parser.parse_args()

    lock_path = Path(args.lock_path)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        print(json.dumps({"ok": False, "status": "already_running", "lock_path": str(lock_path)}))
        return 0

    try:
        os.write(fd, str(time.time()).encode("utf-8"))
        os.close(fd)
        db_path = Path(args.db_path).expanduser()
        repos = [repo.strip() for repo in str(args.repos or "").split(",") if repo.strip()]
        store = DevGitHubPRAutomationStore(db_path)
        scm_store = DevSCMLifecycleStore(db_path)
        result = reconcile_github_pr_automation(
            store=store,
            scm_store=scm_store,
            repos=repos,
            limit=args.limit,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result.get("ok") else 1
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
