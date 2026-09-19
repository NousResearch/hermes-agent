#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
from pathlib import Path

SHA_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        check=False,
    )


def verify_commit(repo: Path, value: str) -> None:
    if not SHA_PATTERN.fullmatch(value):
        raise ValueError("invalid commit identifier")
    result = git(repo, "rev-parse", "--verify", f"{value}^{{commit}}")
    if result.returncode != 0:
        raise ValueError("commit is unavailable")


def safe_relative_path(repo: Path, raw: object) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError("finding path is invalid")
    candidate = Path(raw)
    if candidate.is_absolute():
        candidate = candidate.resolve().relative_to(repo.resolve())
    if ".." in candidate.parts:
        raise ValueError("finding path escapes repository")
    return candidate.as_posix()


def main() -> int:
    args = parse_args()
    try:
        repo = args.repo.resolve(strict=True)
        binary = args.binary.resolve(strict=True)
        if not repo.is_dir() or not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("runtime path is invalid")
        verify_commit(repo, args.head)
        initial_push = bool(SHA_PATTERN.fullmatch(args.base)) and set(args.base) == {"0"}
        if not initial_push:
            verify_commit(repo, args.base)
            if git(repo, "merge-base", "--is-ancestor", args.base, args.head).returncode != 0:
                raise ValueError("base is not an ancestor of head")
        log_options = args.head if initial_push else f"{args.base}..{args.head}"

        args.report.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        args.report.touch(mode=0o600, exist_ok=True)
        args.report.chmod(0o600)
        command = [
            str(binary),
            "git",
            "--redact=100",
            "--no-banner",
            "--no-color",
            "--exit-code=2",
            "--report-format=json",
            f"--report-path={args.report}",
            f"--log-opts={log_options}",
            str(repo),
        ]
        completed = subprocess.run(command, cwd=repo, capture_output=True, check=False)
        if args.report.exists():
            args.report.chmod(0o600)
        if completed.returncode not in (0, 2):
            raise RuntimeError(f"scanner failed with exit code {completed.returncode}")

        findings = json.loads(args.report.read_text() or "[]")
        if not isinstance(findings, list):
            raise ValueError("scanner report is invalid")
        for finding in findings:
            if finding.get("Secret") != "REDACTED":
                raise ValueError("scanner report is not fully redacted")
        expected = 2 if findings else 0
        if completed.returncode != expected:
            raise ValueError("scanner exit and report disagree")

        payload = {
            "schema_version": "secret-guard-ci-result-v1",
            "status": "FINDINGS" if findings else "CLEAN",
            "finding_count": len(findings),
            "rule_ids": sorted({str(item.get("RuleID", "unknown")) for item in findings}),
            "paths": sorted({safe_relative_path(repo, item.get("File")) for item in findings}),
            "report_id": args.report.name,
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return expected
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "secret-guard-ci-result-v1",
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
