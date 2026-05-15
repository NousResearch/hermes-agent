#!/usr/bin/env python3
"""Oflow tracked-file handoff gate.

This gate is intentionally read-only. It reports changed files and verifies that
handoff evidence explicitly records the local-CI hard stops required before a PR
is created or updated.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

REQUIRED_HARD_STOPS = (
    "no_merge",
    "no_deploy",
    "no_restart",
    "no_env_inspection",
    "no_secret_inspection",
    "no_runtime_monitoring",
    "no_production_probes",
    "no_provider_api_calls",
    "no_ssh",
    "no_db_access",
    "no_trading_or_order_impact",
)


def run_git(repo: Path, args: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


def changed_files(repo: Path, upstream: str) -> list[str]:
    merge_base = run_git(repo, ["merge-base", upstream, "HEAD"]).strip()
    committed = run_git(repo, ["diff", "--name-only", f"{merge_base}..HEAD"])
    unstaged = run_git(repo, ["diff", "--name-only"])
    staged = run_git(repo, ["diff", "--cached", "--name-only"])
    untracked = run_git(repo, ["ls-files", "--others", "--exclude-standard"])
    files = [line for line in (committed + "\n" + unstaged + "\n" + staged + "\n" + untracked).splitlines() if line]
    return sorted(dict.fromkeys(files))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--upstream", default="origin/main")
    parser.add_argument("--summary", type=Path, default=Path("artifacts/oflow-local-ci-summary.json"))
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    files = changed_files(repo, args.upstream)
    summary_path = args.summary if args.summary.is_absolute() else repo / args.summary

    failures: list[str] = []
    hard_stops: dict[str, bool] = {}
    if not summary_path.exists():
        failures.append(f"missing summary artifact: {summary_path.relative_to(repo)}")
    else:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        hard_stops = data.get("hard_stops", {})
        missing = [key for key in REQUIRED_HARD_STOPS if hard_stops.get(key) is not True]
        if missing:
            failures.append("hard-stop evidence missing or false: " + ", ".join(missing))

    payload = {
        "passed": not failures,
        "changed_files": files,
        "required_hard_stops": list(REQUIRED_HARD_STOPS),
        "hard_stops": hard_stops,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
