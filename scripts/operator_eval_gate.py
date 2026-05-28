#!/usr/bin/env python3
"""Run a profile trajectory eval batch and write a compact receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def newest_batch(profile_root: Path) -> Path:
    batches = sorted((profile_root / "evals" / "batches").glob("*.json"))
    if not batches:
        raise FileNotFoundError(f"No eval batch specs found under {profile_root / 'evals' / 'batches'}")
    return max(batches, key=lambda path: path.stat().st_mtime_ns)


def default_report_path(profile_root: Path, spec: Path) -> Path:
    return profile_root / "evals" / "runs" / f"{spec.stem}.md"


def run_eval(profile_root: Path, spec: Path, report: Path) -> dict:
    runner = profile_root / "evals" / "run_relative_trajectory_batch.py"
    if not runner.exists():
        raise FileNotFoundError(f"Eval runner not found: {runner}")
    report.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(runner), str(spec), "--output", str(report)]
    proc = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def write_receipt(profile_root: Path, spec: Path, report: Path, result: dict) -> tuple[Path, Path]:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt_dir = profile_root / "evals" / "runs" / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    json_path = receipt_dir / f"operator-eval-gate-{now}.json"
    md_path = receipt_dir / f"operator-eval-gate-{now}.md"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile_root": str(profile_root),
        "spec": str(spec),
        "report": str(report),
        "report_sha256": sha256_file(report) if report.exists() else None,
        "passed": result["returncode"] == 0,
        **result,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    md = "\n".join(
        [
            "# Hermes Operator Eval Gate",
            "",
            f"- Passed: `{payload['passed']}`",
            f"- Spec: `{spec}`",
            f"- Report: `{report}`",
            f"- Report SHA256: `{payload['report_sha256']}`",
            f"- Return code: `{payload['returncode']}`",
            "",
            "## Command",
            "",
            "```text",
            " ".join(result["command"]),
            "```",
        ]
    )
    md_path.write_text(md + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=Path.home() / ".hermes" / "profiles" / "sawyer",
    )
    parser.add_argument("--spec", type=Path, help="Eval batch spec JSON. Defaults to newest profile batch.")
    parser.add_argument("--report", type=Path, help="Rendered markdown report path.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    profile_root = args.profile_root.expanduser().resolve()
    spec = (args.spec or newest_batch(profile_root)).expanduser().resolve()
    report = (args.report or default_report_path(profile_root, spec)).expanduser().resolve()
    result = run_eval(profile_root, spec, report)
    json_path, md_path = write_receipt(profile_root, spec, report, result)
    print(f"receipt_json={json_path}")
    print(f"receipt_md={md_path}")
    return result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
