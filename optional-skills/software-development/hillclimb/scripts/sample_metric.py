#!/usr/bin/env python3
"""Noise-resistant metric sampler for the hillclimb optimization loop.

Runs a harness command N times, extracts one number from each run's stdout,
and reports the median. The baseline is frozen in `.hillclimb/baseline.json`
(override with --dir or HILLCLIMB_DIR) together with a `harness_id`: the first
12 hex characters of the SHA-256 of the whitespace-normalized harness command.

A changed harness command changes the `harness_id`. `compare` refuses to
compare across two different measurement methods (exit 3), because every
earlier number becomes incomparable when the harness changes.

FAILURE RULE: if the harness exits non-zero, or no number can be extracted
from its stdout, that sample FAILS. If ANY sample failed, no metric is
emitted at all and the command exits 2. There is deliberately no flag to
average over failed runs - a metric built from failed samples is not a metric.

Subcommands:
  baseline --harness CMD [--samples N] [--extract SPEC] [--name S]
           [--direction minimize|maximize] [--unit S] [--timeout S]
      Sample, then write baseline.json with frozen: true. The extraction spec
      is recorded in the baseline so `compare` reuses it. Exit 0.
  run --harness CMD [--samples N] [--extract SPEC] [--timeout S]
      Sample without touching the baseline. Exit 0.
  compare [--harness CMD | --value F] [--samples N] [--extract SPEC]
      Compare against the frozen baseline. When --extract is omitted, the spec
      recorded in the baseline is used, so a comparison can never silently
      measure a different way than the baseline did. Exit 0 for a direction-aware
      improvement, 1 when the metric did not improve, 2 when there is no
      baseline or the value is unusable, 3 when the harness no longer matches
      the frozen baseline.

--extract SPEC grammar (default: auto):
  auto             the LAST number printed (optional trailing unit)
  regex:PATTERN    the first capture group of the first match
  json:dotted.path the number at that path in stdout parsed as JSON
  line:PREFIX      the first number on the first line starting with PREFIX

Exit codes:
  0 - success (for `compare`: the metric improved)
  1 - `compare` only: the metric did not improve
  2 - a sample failed, or the invocation is unusable (no baseline to compare
      against, unparseable --value)
  3 - `compare` only: the harness command no longer matches the baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

DEFAULT_SAMPLES = 5
DEFAULT_TIMEOUT = 300
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def die(message: str, code: int = 2) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


def resolve_dir(value: str | None) -> Path:
    return Path(value or os.environ.get("HILLCLIMB_DIR") or ".hillclimb")


def ensure_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        die(f"{path} exists but is not a directory")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        die(f"cannot create {path}: {exc}")
    if not path.is_dir():
        die(f"{path} could not be created as a directory")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def harness_id(command: str) -> str:
    """Fingerprint of the measurement method: whitespace-insensitive."""
    normalized = " ".join(command.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def extract_metric(text: str, spec: str) -> tuple[float | None, str | None]:
    if spec == "auto" or spec == "":
        matches = NUMBER_RE.findall(text)
        if not matches:
            return None, "auto: no numeric token found in stdout"
        try:
            return float(matches[-1]), None
        except ValueError:
            return None, f"auto: not a number: {matches[-1]!r}"

    if spec.startswith("regex:"):
        pattern = spec[len("regex:") :]
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return None, f"regex: invalid pattern: {exc}"
        match = compiled.search(text)
        if not match:
            return None, "regex: no match in stdout"
        captured = match.group(1) if match.lastindex else match.group(0)
        try:
            return float(captured), None
        except ValueError:
            return None, f"regex: capture is not a number: {captured!r}"

    if spec.startswith("json:"):
        dotted = spec[len("json:") :]
        try:
            data = json.loads(text)
        except ValueError as exc:
            return None, f"json: stdout is not valid JSON: {exc}"
        current = data
        for key in dotted.split("."):
            if isinstance(current, list):
                try:
                    current = current[int(key)]
                except (ValueError, IndexError):
                    return None, f"json: no element {key!r}"
            elif isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None, f"json: path {dotted!r} not found"
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            return None, f"json: value at {dotted!r} is not a number"
        return float(current), None

    if spec.startswith("line:"):
        prefix = spec[len("line:") :]
        for line in text.splitlines():
            if line.startswith(prefix):
                matches = NUMBER_RE.findall(line)
                if not matches:
                    return None, f"line: no number on line starting with {prefix!r}"
                return float(matches[0]), None
        return None, f"line: no line starts with {prefix!r}"

    return None, f"unknown --extract spec: {spec!r}"


def run_harness(command: str, timeout: int) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"harness timed out after {timeout}s"
    except OSError as exc:
        return 127, "", f"harness could not start: {exc}"
    return completed.returncode, completed.stdout or "", completed.stderr or ""


def sample(command: str, count: int, extract: str, timeout: int) -> tuple[dict | None, list[dict]]:
    """Return (summary, failures). summary is None when any sample failed."""
    if count < 1:
        die("--samples must be at least 1")
    values: list[float] = []
    failures: list[dict] = []
    for index in range(1, count + 1):
        code, stdout, stderr = run_harness(command, timeout)
        if code != 0:
            failures.append(
                {
                    "sample": index,
                    "exit_code": code,
                    "stdout": stdout[-400:],
                    "stderr": stderr[-400:],
                }
            )
            continue
        value, error = extract_metric(stdout, extract)
        if value is None:
            failures.append(
                {
                    "sample": index,
                    "exit_code": code,
                    "extract_error": error,
                    "stdout": stdout[-400:],
                }
            )
            continue
        values.append(value)

    if failures:
        return None, failures

    return {
        "samples": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
        "values": values,
    }, []


def report_failures(command: str, failures: list[dict]) -> NoReturn:
    print(
        json.dumps(
            {
                "error": "sample failed; no metric emitted",
                "harness": command,
                "failed_samples": len(failures),
                "failures": failures,
            },
            indent=2,
        )
    )
    sys.exit(2)


def load_baseline(path: Path) -> dict:
    baseline = path / "baseline.json"
    if not baseline.is_file():
        return {}
    try:
        return json.loads(baseline.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        die(f"baseline.json is unreadable: {exc}")


def cmd_baseline(args: argparse.Namespace) -> int:
    path = resolve_dir(args.dir)
    extract = args.extract or "auto"
    summary, failures = sample(args.harness, args.samples, extract, args.timeout)
    if summary is None:
        report_failures(args.harness, failures)
    ensure_dir(path)
    stamp = now_iso()
    baseline = {
        "metric_name": args.name,
        "direction": args.direction,
        "harness": args.harness,
        "harness_id": harness_id(args.harness),
        "extract": extract,
        "samples": summary["samples"],
        "median": summary["median"],
        "min": summary["min"],
        "max": summary["max"],
        "spread": summary["spread"],
        "unit": args.unit,
        "frozen": True,
        "created_at": stamp,
        "frozen_at": stamp,
        "values": summary["values"],
    }
    (path / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(baseline, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    summary, failures = sample(args.harness, args.samples, args.extract or "auto", args.timeout)
    if summary is None:
        report_failures(args.harness, failures)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    path = resolve_dir(args.dir)
    baseline = load_baseline(path)
    if not baseline or "median" not in baseline:
        die(f"no frozen baseline in {path}; run `baseline` first")

    direction = baseline.get("direction", "minimize")
    before = float(baseline["median"])

    if args.harness:
        current_id = harness_id(args.harness)
        if current_id != baseline.get("harness_id"):
            print(
                json.dumps(
                    {
                        "error": "harness-id mismatch; baseline invalidated",
                        "baseline_harness_id": baseline.get("harness_id"),
                        "current_harness_id": current_id,
                    },
                    indent=2,
                )
            )
            return 3
        # Reuse the extraction spec the baseline was recorded with, so `compare`
        # can never silently measure a different way than the baseline did.
        extract = args.extract or baseline.get("extract") or "auto"
        summary, failures = sample(args.harness, args.samples, extract, args.timeout)
        if summary is None:
            report_failures(args.harness, failures)
        after = summary["median"]
        samples = summary["samples"]
    elif args.value is not None:
        try:
            after = float(args.value)
        except ValueError:
            die(f"--value must be a number: {args.value!r}")
        samples = 0
    else:
        die("compare needs --harness CMD or --value F")

    delta = after - before
    improved = delta < 0 if direction == "minimize" else delta > 0
    print(
        json.dumps(
            {
                "metric_name": baseline.get("metric_name"),
                "direction": direction,
                "before": before,
                "after": after,
                "delta": delta,
                "samples": samples,
                "improved": improved,
            },
            indent=2,
        )
    )
    return 0 if improved else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hillclimb metric sampler")
    parser.add_argument(
        "--dir",
        help="Directory holding baseline.json (default: HILLCLIMB_DIR env var, else .hillclimb)",
    )
    parser.add_argument(
        "--extract",
        default=None,
        help=(
            "auto | regex:PATTERN | json:dotted.path | line:PREFIX. Defaults to auto, "
            "except in `compare`, which defaults to the spec recorded in the baseline."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        # SUPPRESS keeps the root-level value when the flag is given before the
        # subcommand, while still allowing it after it.
        sub.add_argument("--dir", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        sub.add_argument("--extract", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    baseline = subparsers.add_parser("baseline", help="Sample and freeze a baseline")
    baseline.add_argument("--harness", required=True, help="command that prints the metric")
    baseline.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    baseline.add_argument("--name", default="metric")
    baseline.add_argument("--direction", choices=["minimize", "maximize"], default="minimize")
    baseline.add_argument("--unit", default=None)
    baseline.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    add_common(baseline)
    baseline.set_defaults(func=cmd_baseline)

    run = subparsers.add_parser("run", help="Sample without touching the baseline")
    run.add_argument("--harness", required=True)
    run.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    run.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    add_common(run)
    run.set_defaults(func=cmd_run)

    compare = subparsers.add_parser("compare", help="Compare against the frozen baseline")
    compare.add_argument("--harness", default=None)
    compare.add_argument("--value", default=None)
    compare.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    compare.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    add_common(compare)
    compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
