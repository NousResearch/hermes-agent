#!/usr/bin/env python3
"""Run bounded Hermes/dflash stability canaries and write JSONL evidence.

This is intentionally not an endless loop.  It runs a small set of workflow
prompts through ``hermes -z`` and requires each turn to return an exact success
marker.  Failures are recorded with enough local evidence to debug the next
root cause without relying on phone screenshots alone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence


DEFAULT_LOG_DIR = Path.home() / ".hermes" / "logs" / "dflash-stability-canary"
DEFAULT_PROVIDER = "taro"
DEFAULT_MODEL = "dflash"
DEFAULT_TOOLSETS = "terminal,file"
MAX_OUTPUT_CHARS = 6000


@dataclass(frozen=True)
class CanaryCase:
    name: str
    marker: str
    prompt: str


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False


DEFAULT_CASES: tuple[CanaryCase, ...] = (
    CanaryCase(
        name="meshboard-onboard",
        marker="CANARY_ONBOARD_OK",
        prompt=(
            "From the current working directory, run "
            "`python3 .mesh/tools/meshctl.py onboard`. If the command completes "
            "without a Python traceback, reply exactly CANARY_ONBOARD_OK and "
            "nothing else. If it fails, do not use the marker; summarize the "
            "failure briefly."
        ),
    ),
    CanaryCase(
        name="meshboard-status-read",
        marker="CANARY_STATUS_OK",
        prompt=(
            "From the current working directory, inspect STATUS.md just enough "
            "to confirm these task ids appear: "
            "hermes-dflash-phone-cli-sudden-death-root-cause-20260531 and "
            "hermes-tui-phone-status-bar-readable. Use shell tools if helpful. "
            "If both appear, reply exactly CANARY_STATUS_OK and nothing else. "
            "If either is missing, do not use the marker; say which is missing."
        ),
    ),
    CanaryCase(
        name="short-fragment-detector",
        marker="CANARY_FRAGMENT_OK",
        prompt=(
            "Run a Python check in the current Hermes checkout that imports "
            "`looks_like_incomplete_final_fragment` from `agent.stall_retry` and "
            "verifies it returns true for exactly this string: "
            "`I see a lot of discord-res tasks (digest Discord content) and some`. "
            "If the check passes, reply exactly CANARY_FRAGMENT_OK and nothing "
            "else. If it fails, do not use the marker; summarize the failure."
        ),
    ),
)


_INCOMPLETE_TAIL_RE = re.compile(
    r"(?i)(?:^|[\s,;:])("
    r"and|or|but|then|so|because|with|without|to|for|from|in|on|at|of|by|as|"
    r"that|which|who|when|where|while|after|before|since|until|if|though|"
    r"although|some|another"
    r")\s*[.?!\"')\]]*$"
)

_ACTION_PROMISE_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:let me|now let me|i(?:'ll| will)|i need to)\b.*(?:check|inspect|read|run|look|pick|continue)\b"
)

_AUTH_RE = re.compile(r"(?i)(?:401|autherror|invalid api key|api key is invalid|unauthorized)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text

    half = max(1, max_chars // 2)
    return f"{text[:half]}\n...[truncated {len(text) - max_chars} chars]...\n{text[-half:]}"


def looks_like_incomplete_tail(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return False

    if _INCOMPLETE_TAIL_RE.search(stripped):
        return True

    tail = stripped[-240:]

    return bool(_ACTION_PROMISE_RE.search(tail)) and not re.search(r"[.!?]\s*$", tail)


def build_command(
    hermes_bin: str,
    case: CanaryCase,
    *,
    model: str,
    provider: str,
    toolsets: str,
) -> list[str]:
    cmd = [hermes_bin]

    if provider:
        cmd.extend(["--provider", provider])

    if model:
        cmd.extend(["--model", model])

    if toolsets:
        cmd.extend(["--toolsets", toolsets])

    cmd.extend(["-z", case.prompt])

    return cmd


def run_subprocess(cmd: Sequence[str], *, cwd: Path, timeout_s: float) -> CommandResult:
    started = time.monotonic()

    env = os.environ.copy()
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("HERMES_ONESHOT_FAULTHANDLER", "1")
    start_new_session = os.name != "nt"
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=start_new_session,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        elapsed_s = time.monotonic() - started
        return CommandResult(
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed_s=elapsed_s,
        )
    except subprocess.TimeoutExpired:
        if start_new_session and hasattr(signal, "SIGUSR1"):
            try:
                os.killpg(proc.pid, signal.SIGUSR1)
                time.sleep(1.0)
            except Exception:
                pass
        try:
            proc.terminate()
            stdout, stderr = proc.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            stdout, stderr = proc.communicate()
        elapsed_s = time.monotonic() - started
        return CommandResult(returncode=124, stdout=stdout, stderr=stderr, elapsed_s=elapsed_s, timed_out=True)


def classify_result(result: CommandResult, marker: str, *, strict_marker: bool = True) -> str | None:
    combined = f"{result.stdout}\n{result.stderr}".strip()
    stdout = result.stdout.strip()

    if result.timed_out:
        return "timeout"

    if result.returncode != 0:
        return "nonzero-exit"

    if not stdout:
        return "empty-final"

    if _AUTH_RE.search(combined):
        return "auth-error"

    if marker:
        if strict_marker and stdout != marker:
            return "marker-mismatch"
        if not strict_marker and marker not in stdout:
            return "marker-missing"

    if looks_like_incomplete_tail(stdout):
        return "incomplete-tail"

    return None


def record_path(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return log_dir / f"{stamp}.jsonl"


def iter_cases(names: Iterable[str] | None = None) -> list[CanaryCase]:
    cases = list(DEFAULT_CASES)

    if not names:
        return cases

    selected: list[CanaryCase] = []
    available = {case.name: case for case in cases}

    for name in names:
        if name not in available:
            raise SystemExit(f"Unknown canary case {name!r}; available: {', '.join(sorted(available))}")
        selected.append(available[name])

    return selected


def run_case(
    case: CanaryCase,
    *,
    cwd: Path,
    hermes_bin: str,
    model: str,
    provider: str,
    toolsets: str,
    timeout_s: float,
    strict_marker: bool,
    runner: Callable[[Sequence[str], Path, float], CommandResult] | None = None,
) -> dict:
    cmd = build_command(hermes_bin, case, model=model, provider=provider, toolsets=toolsets)

    if runner is None:
        result = run_subprocess(cmd, cwd=cwd, timeout_s=timeout_s)
    else:
        result = runner(cmd, cwd, timeout_s)

    failure = classify_result(result, case.marker, strict_marker=strict_marker)

    return {
        "case": case.name,
        "cmd": cmd[:-1] + ["<prompt>"],
        "cwd": str(cwd),
        "elapsed_s": round(result.elapsed_s, 3),
        "failure": failure,
        "marker": case.marker,
        "ok": failure is None,
        "returncode": result.returncode,
        "stderr": truncate(result.stderr),
        "stdout": truncate(result.stdout),
        "timed_out": result.timed_out,
        "ts": utc_now(),
    }


def write_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=1, help="Number of times to run the selected cases.")
    parser.add_argument("--sleep", type=float, default=10.0, help="Seconds to sleep between rounds.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-case timeout in seconds.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Working directory for hermes -z.")
    parser.add_argument("--hermes-bin", default=os.environ.get("HERMES_BIN", "hermes"), help="Hermes executable.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="Hermes provider override.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hermes model override.")
    parser.add_argument("--toolsets", default=DEFAULT_TOOLSETS, help="Comma-separated toolsets for canary turns.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory for JSONL evidence logs.")
    parser.add_argument("--case", action="append", dest="cases", help="Run only this case name; repeatable.")
    parser.add_argument("--allow-extra-output", action="store_true", help="Allow marker to appear inside extra text.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running Hermes.")
    parser.add_argument("--json", action="store_true", help="Print JSON records to stdout.")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cases = iter_cases(args.cases)

    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")

    log_path = record_path(args.log_dir)
    strict_marker = not args.allow_extra_output

    if args.dry_run:
        plan = {
            "cases": [case.name for case in cases],
            "cwd": str(args.cwd),
            "hermes_bin": args.hermes_bin,
            "log_path": str(log_path),
            "model": args.model,
            "provider": args.provider,
            "rounds": args.rounds,
            "toolsets": args.toolsets,
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    failures = 0

    for round_index in range(args.rounds):
        for case in cases:
            record = run_case(
                case,
                cwd=args.cwd,
                hermes_bin=args.hermes_bin,
                model=args.model,
                provider=args.provider,
                toolsets=args.toolsets,
                timeout_s=args.timeout,
                strict_marker=strict_marker,
            )
            record["round"] = round_index + 1
            write_record(log_path, record)

            if args.json:
                print(json.dumps(record, sort_keys=True), flush=True)
            else:
                status = "ok" if record["ok"] else f"fail:{record['failure']}"
                print(f"{record['ts']} round={round_index + 1} case={case.name} {status}", flush=True)

            if not record["ok"]:
                failures += 1
                print(f"Evidence: {log_path}", file=sys.stderr)
                return 1

        if round_index != args.rounds - 1 and args.sleep > 0:
            time.sleep(args.sleep)

    if not args.json:
        print(f"All canaries passed. Evidence: {log_path}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
