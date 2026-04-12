#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_head_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def ensure_repo_root() -> Path:
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    return Path(root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write .hermes-gate/gate.json for the current HEAD")
    p.add_argument("--gate-dir", default=".hermes-gate", help="Directory for gate artifacts (default: .hermes-gate)")

    p.add_argument("--review-status", required=True, choices=["PASS", "FAIL"])
    p.add_argument("--reviewer", required=True)
    p.add_argument("--review-summary", required=True)
    p.add_argument("--review-report", required=True)

    p.add_argument("--test-status", required=True, choices=["PASS", "FAIL"])
    p.add_argument("--tester", required=True)
    p.add_argument("--test-summary", required=True)
    p.add_argument("--test-report", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = ensure_repo_root()
    head_sha = git_head_sha()

    gate_dir = repo_root / args.gate_dir
    gate_dir.mkdir(parents=True, exist_ok=True)

    review_report = Path(args.review_report)
    test_report = Path(args.test_report)
    if not review_report.is_absolute():
        review_report = repo_root / review_report
    if not test_report.is_absolute():
        test_report = repo_root / test_report

    if not review_report.exists():
        print(f"review report not found: {review_report}", file=sys.stderr)
        return 1
    if not test_report.exists():
        print(f"test report not found: {test_report}", file=sys.stderr)
        return 1

    payload = {
        "head_sha": head_sha,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "review": {
            "status": args.review_status,
            "head_sha": head_sha,
            "reviewer": args.reviewer,
            "summary": args.review_summary,
            "report_path": str(review_report.relative_to(repo_root)),
        },
        "test": {
            "status": args.test_status,
            "head_sha": head_sha,
            "tester": args.tester,
            "summary": args.test_summary,
            "report_path": str(test_report.relative_to(repo_root)),
        },
    }

    gate_path = gate_dir / "gate.json"
    gate_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(gate_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
