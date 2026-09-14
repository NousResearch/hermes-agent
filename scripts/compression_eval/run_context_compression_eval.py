"""Run and validate the external hermes-compression-eval harness."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import shutil
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


def _command_runtime_fingerprint(command: list[str]) -> str:
    executable = command[0]
    resolved = shutil.which(executable) if not Path(executable).is_absolute() else executable
    if resolved is None and Path(executable).is_file():
        first = Path(executable).read_text(encoding="utf-8", errors="replace").splitlines()[:1]
        resolved = first[0][2:].strip() if first and first[0].startswith("#!") else executable
    if resolved:
        try:
            version = subprocess.run([resolved, "--version"], capture_output=True, text=True, timeout=5, check=False)
            return json.dumps([resolved, version.stdout, version.stderr], sort_keys=True)
        except (OSError, subprocess.SubprocessError):
            pass
    return json.dumps([executable, sys.version], sort_keys=True)


def _evaluator_digest(harness: Path, command: list[str]) -> str:
    digest = hashlib.sha256(_command_runtime_fingerprint(command).encode())
    for path in sorted(harness.rglob("*")):
        relative = path.relative_to(harness)
        if any(part in {".git", ".venv", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if relative.parts[0] == "results" or not path.is_file():
            continue
        digest.update(str(relative).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--hermes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--battery-definition", type=Path, help="Reviewed JSON list of expected probe names")
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
    if args.battery_definition is None:
        raise SystemExit("--battery-definition is required")
    battery_bytes = args.battery_definition.read_bytes()
    expected = json.loads(battery_bytes)
    if not isinstance(expected, list) or not expected or any(not isinstance(x, str) or not x.strip() for x in expected) or len(set(expected)) != len(expected):
        raise SystemExit("battery definition must contain unique nonempty probe names")
    battery_digest = hashlib.sha256(battery_bytes).hexdigest()
    evaluator_digest = _evaluator_digest(harness, command)
    args.output = output
    if output == args.battery_definition.resolve():
        raise SystemExit("--output must not overwrite --battery-definition")
    args.output.unlink(missing_ok=True)
    report_path = harness / "results" / "latest" / "report.json"
    if report_path.exists():
        stale = report_path.with_name(f"report.stale.{time.time_ns()}.json")
        report_path.replace(stale)
    process: subprocess.Popen[str] = subprocess.Popen(
        command,
        cwd=harness,
        env={**os.environ, "HERMES_AGENT_ROOT": str(hermes_root),
             "HERMES_EVALUATOR_DIGEST": evaluator_digest, "HERMES_BATTERY_DIGEST": battery_digest},
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
    if _resolve_clean_source(hermes_root)[1] != source_sha or _evaluator_digest(harness, command) != evaluator_digest:
        raise SystemExit("source or evaluator changed during compression evaluation")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("evaluator_digest") != evaluator_digest or report.get("battery_digest") != battery_digest:
        raise SystemExit("invalid compression evaluator or battery provenance")
    if report.get("probe_manifest") != expected or set(report.get("probe_scores", {})) != set(expected):
        raise SystemExit("compression report does not match the independent battery definition")
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
