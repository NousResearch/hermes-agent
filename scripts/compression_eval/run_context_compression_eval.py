"""Run and validate the external hermes-compression-eval harness."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__:
    from .report_contract import validate_report
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.compression_eval.report_contract import validate_report


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    import psutil

    parent = psutil.Process(process.pid)
    children = parent.children(recursive=True)
    for child in children:
        child.terminate()
    parent.terminate()
    _, alive = psutil.wait_procs([*children, parent], timeout=10)
    for child in alive:
        child.kill()
    process.wait()


def _resolve_clean_source(path: Path) -> tuple[Path, str]:
    source_root = path.expanduser().resolve()
    if not source_root.is_dir():
        raise SystemExit(f"--hermes-root is not a directory: {source_root}")
    status = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if status.returncode or status.stdout.strip():
        raise SystemExit(f"--hermes-root must be clean: {source_root}")
    try:
        source_sha = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit("unable to resolve --hermes-root source revision") from exc
    return source_root, source_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--hermes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    hermes_root = args.hermes_root.expanduser().resolve()
    harness = args.harness.expanduser().resolve()
    command = list(args.command)
    if command[:1] == ["--"]:
        command.pop(0)
    if not hermes_root.is_dir() or not harness.is_dir() or not command:
        raise SystemExit("--harness, --hermes-root, and a harness command are required")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be finite and positive")
    hermes_root, source_sha = _resolve_clean_source(hermes_root)
    output = args.output.expanduser().resolve()
    if output == hermes_root or hermes_root in output.parents or output == harness or harness in output.parents:
        raise SystemExit("--output must be outside the source and harness trees")
    args.output = output
    args.output.unlink(missing_ok=True)
    report_path = harness / "results" / "latest" / "report.json"
    if report_path.exists():
        stale = report_path.with_name(f"report.stale.{time.time_ns()}.json")
        report_path.replace(stale)
    process: subprocess.Popen[str] = subprocess.Popen(
        command,
        cwd=harness,
        env={**os.environ, "HERMES_AGENT_ROOT": str(hermes_root)},
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(process)
        process.communicate()
        raise SystemExit(f"compression harness timed out after {args.timeout_seconds:g}s") from exc
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if result.returncode:
        raise SystemExit(f"compression harness failed with exit {result.returncode}")
    if not report_path.exists():
        raise SystemExit(f"compression report missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("source_sha") != source_sha:
        raise SystemExit("invalid compression report: source_sha does not match --hermes-root")
    errors = validate_report(report)
    if errors:
        raise SystemExit("invalid compression report: " + ", ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending_output = args.output.with_name(args.output.name + ".pending")
    pending_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending_output.replace(args.output)
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
