#!/usr/bin/env python3
"""Run the devtonic-agent local test baseline, one target at a time.

Each target runs in its own ``python -m pytest`` subprocess (same isolation
idea as ``scripts/run_tests.sh``) inside the repo's ``uv`` environment, so a
hang or crash in one area never hides the result of another. The summary is
printed as a Markdown table that can be pasted into
``docs/devtonic/test-baseline.md``.

Usage::

    uv run python scripts/devtonic/baseline.py               # default fast targets
    uv run python scripts/devtonic/baseline.py tests/cron    # explicit targets
    uv run python scripts/devtonic/baseline.py --json out.json --timeout 900

The tests' own ``conftest.py`` sandboxes ``HERMES_HOME``, so running this
never reads or writes a real Hermes install.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Targets that pass on a laptop in about two minutes total with the default
# `uv sync --group dev --group test` environment. Heavier or host-sensitive
# areas (hermes_state, acp_adapter, gateway, hermes_cli, agent, tools,
# tui_gateway, e2e) are tracked in docs/devtonic/test-baseline.md and run
# explicitly.
DEFAULT_TARGETS = (
    "tests/devtonic",
    "tests/hermes_platform",
    "tests/conformance",
    "tests/providers",
    "tests/skills",
    "tests/cron",
)

_COUNT_RE = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)")
# Short-summary lines only ("FAILED path.py::node - msg"); captured log records
# also start with "ERROR" but are followed by spaces and a logger name.
_FAILURE_RE = re.compile(r"^(?:FAILED|ERROR) (\S+\.py(?:::\S+)?)")


@dataclass
class TargetResult:
    target: str
    returncode: int
    seconds: float
    counts: dict[str, int]
    summary: str
    failures: list[str]

    @property
    def status(self) -> str:
        if self.returncode == -1:
            return "timeout"
        if self.returncode == 0:
            return "pass"
        if self.returncode == 5:
            return "no tests"
        return "fail"


def parse_summary(output: str) -> tuple[dict[str, int], str, list[str]]:
    """Extract counts, the final summary line, and failing node ids."""
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    summary = ""
    for ln in reversed(lines):
        if _COUNT_RE.search(ln) or "no tests ran" in ln:
            summary = ln.strip("= ")
            break
    counts: dict[str, int] = {}
    for num, kind in _COUNT_RE.findall(summary):
        key = "errors" if kind.startswith("error") else kind
        counts[key] = counts.get(key, 0) + int(num)
    failures = [m.group(1) for ln in lines if (m := _FAILURE_RE.match(ln))]
    return counts, summary, failures


def run_target(target: str, timeout: int, extra: list[str]) -> TargetResult:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE", target, *extra]
    env = dict(os.environ, TZ="UTC", PYTHONHASHSEED="0")
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, check=False, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        counts, _, failures = parse_summary(out)
        return TargetResult(target, -1, time.monotonic() - start, counts, f"timeout after {timeout}s", failures)
    counts, summary, failures = parse_summary(proc.stdout + proc.stderr)
    return TargetResult(target, proc.returncode, time.monotonic() - start, counts, summary, failures)


def expand_target(target: str) -> list[str]:
    """Split a directory target into its test files (upstream's isolation unit).

    The upstream suite is written for one process per test file
    (``scripts/run_tests_parallel.py``); running a whole directory in one
    process leaks module state across files and produces false failures.
    """
    path = REPO_ROOT / target
    if "::" in target or not path.is_dir():
        return [target]
    files = sorted(
        p.relative_to(REPO_ROOT).as_posix()
        for p in path.rglob("test_*.py")
        if "__pycache__" not in p.parts
    )
    return files or [target]


def _exit_rank(rc: int) -> int:
    # Aggregate exit code: any timeout or failure wins over "no tests" or pass.
    return {-1: 3, 0: 0, 5: -1}.get(rc, 2)


def run_isolated(target: str, timeout: int, extra: list[str], jobs: int) -> TargetResult:
    """Run every test file of ``target`` in its own subprocess and aggregate."""
    files = expand_target(target)
    if len(files) == 1:
        return run_target(files[0], timeout, extra)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        parts = list(pool.map(lambda f: run_target(f, timeout, extra), files))
    counts: dict[str, int] = {}
    failures: list[str] = []
    for part in parts:
        for key, num in part.counts.items():
            counts[key] = counts.get(key, 0) + num
        failures.extend(part.failures)
        if part.returncode not in (0, 5) and not part.failures:
            failures.append(f"{part.target} ({part.summary or 'exit ' + str(part.returncode)})")
    worst = max(parts, key=lambda p: _exit_rank(p.returncode)).returncode
    if all(p.returncode == 5 for p in parts):
        worst = 5
    elif worst == 5:
        worst = 0
    summary = ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) + f" across {len(files)} files"
    return TargetResult(target, worst, time.monotonic() - start, counts, summary, failures)


def render_table(results: list[TargetResult]) -> str:
    rows = ["| Target | Status | Passed | Failed | Errors | Skipped | Time (s) |", "|---|---|---|---|---|---|---|"]
    for r in results:
        c = r.counts
        rows.append(
            f"| `{r.target}` | {r.status} | {c.get('passed', 0)} | {c.get('failed', 0)} "
            f"| {c.get('errors', 0)} | {c.get('skipped', 0)} | {r.seconds:.0f} |"
        )
    return "\n".join(rows)


def host_line() -> str:
    """One-line host description for baseline tables (OS, arch, Python, SQLite)."""
    system = platform.system()
    if system == "Darwin":
        os_name = f"macOS {platform.mac_ver()[0]}"
    elif system == "Linux":
        try:
            os_name = platform.freedesktop_os_release().get("PRETTY_NAME", "Linux")
        except OSError:
            os_name = "Linux"
    else:
        os_name = f"{system} {platform.release()}"
    return (
        f"{os_name} {platform.machine()}, Python {platform.python_version()}, "
        f"SQLite {sqlite3.sqlite_version}, {os.cpu_count() or 1} CPUs"
    )


def render_markdown(results: list[TargetResult]) -> str:
    """Host line, results table, and failing node ids, ready for a doc or CI summary."""
    parts = [f"Measured on {host_line()}:", "", render_table(results)]
    failing = [(r.target, node) for r in results for node in r.failures]
    if failing:
        parts += ["", "Failures:", ""]
        parts += [f"- `{target}`: `{node}`" for target, node in failing]
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser(
        description="Run the devtonic-agent local test baseline (args after -- go to pytest)."
    )
    parser.add_argument("targets", nargs="*", help="test paths (default: fast baseline set)")
    parser.add_argument("--timeout", type=int, default=900, help="timeout in seconds per pytest process")
    parser.add_argument("--json", type=Path, help="also write results as JSON to this path")
    parser.add_argument(
        "--markdown",
        type=Path,
        help="also append the host line + results table to this path (e.g. $GITHUB_STEP_SUMMARY)",
    )
    parser.add_argument(
        "--per-dir",
        action="store_true",
        help="run each directory target in one pytest process (default: one process per test file)",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=min(8, os.cpu_count() or 2), help="parallel files per target"
    )
    args = parser.parse_args(argv)

    targets = args.targets or list(DEFAULT_TARGETS)
    results: list[TargetResult] = []
    for target in targets:
        if not (REPO_ROOT / target.split("::", 1)[0]).exists():
            print(f"skip {target}: not found", file=sys.stderr)
            continue
        if args.per_dir:
            res = run_target(target, args.timeout, extra)
        else:
            res = run_isolated(target, args.timeout, extra, args.jobs)
        print(f"{res.status:8} {res.seconds:7.1f}s {target} :: {res.summary}", flush=True)
        for node in res.failures:
            print(f"    {node}", flush=True)
        results.append(res)

    print()
    print(render_table(results))
    if args.json:
        payload = [dict(asdict(r), status=r.status) for r in results]
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    if args.markdown:
        with args.markdown.open("a", encoding="utf-8") as fh:
            fh.write(render_markdown(results))
    if not results:
        return 1
    return 0 if all(r.status in ("pass", "no tests") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
