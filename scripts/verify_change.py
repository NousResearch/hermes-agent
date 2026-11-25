#!/usr/bin/env python3
"""Verify the tests impacted by a change against a known-failure baseline.

The full Hermes suite carries unrelated branch debt. Running every test for a
small change therefore produces a noisy list of failures that tells an agent
nothing about whether *its* change regressed anything. This command narrows the
run to tests belonging to the changed areas and compares the result with a
recorded baseline:

    scripts/verify_change.py --dry-run
    scripts/verify_change.py --baseline .cache/test-baseline.json
    scripts/verify_change.py --record .cache/test-baseline.json

Selection is deliberately conservative: a changed test file always runs, a
changed module runs tests named after it and tests that import it, and a change
to shared test infrastructure runs the whole suite. Tests run through
``scripts/run_tests.sh``; this script never invokes bare pytest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_DEFAULT_BASE = os.environ.get("HERMES_VERIFY_BASE", "origin/main")
_GLOBAL_TEST_PATHS = {"tests/conftest.py"}
_GLOBAL_TEST_PREFIXES = ("tests/_fixtures/",)
_SKIP_DISCOVERY_ROOTS = ("tests/integration/", "tests/e2e/", "tests/docker/")
_FAILURE_RE = re.compile(r"\b(?:FAILED|ERROR)\s+(tests/\S+)")


class VerifyError(RuntimeError):
    """An expected verification failure with a user-facing message."""


@dataclass(frozen=True)
class RunResult:
    returncode: int
    output: str
    failures: frozenset[str]


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise VerifyError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _git_paths(repo: Path, *args: str) -> list[str]:
    return [part for part in _git(repo, *args).split("\0") if part]


def changed_paths(repo: Path, base: str) -> list[str]:
    """Tracked changes against *base* plus untracked files."""
    return sorted(
        set(_git_paths(repo, "diff", "--name-only", "-z", base, "--"))
        | set(_git_paths(repo, "ls-files", "--others", "--exclude-standard", "-z"))
    )


def _is_test_file(path: str) -> bool:
    return path.startswith("tests/") and path.endswith(".py") and Path(path).name.startswith("test_")


def _is_discoverable_test_file(path: str) -> bool:
    return _is_test_file(path) and not path.startswith(_SKIP_DISCOVERY_ROOTS)


def _test_files(repo: Path) -> list[str]:
    files = _git_paths(repo, "ls-files", "-z", "tests")
    files += _git_paths(repo, "ls-files", "-z", "--others", "--exclude-standard", "tests")
    return sorted({path for path in files if _is_discoverable_test_file(path)})


def _module_name(path: str) -> str:
    module = path[:-3].replace("/", ".")
    return module.removesuffix(".__init__")


def _import_needles(path: str) -> tuple[str, ...]:
    module = _module_name(path)
    parent, _, leaf = module.rpartition(".")
    needles = [module]
    if parent:
        needles.append(f"from {parent} import {leaf}")
    return tuple(needles)


def select_tests(
    changed: Iterable[str],
    all_tests: Iterable[str],
    read_text: Callable[[str], str],
) -> tuple[list[str], bool]:
    """Return ``(test paths, global_change)`` for *changed* files.

    ``global_change`` means shared test infrastructure changed, so the caller
    must run the whole suite rather than a named subset.
    """
    changed = sorted(set(changed))
    tests = sorted(set(all_tests))
    selected = {path for path in changed if _is_test_file(path)}
    global_change = any(path in _GLOBAL_TEST_PATHS or path.startswith(_GLOBAL_TEST_PREFIXES)
                        for path in changed)

    text_cache: dict[str, str] = {}

    def read(path: str) -> str:
        if path not in text_cache:
            text_cache[path] = read_text(path)
        return text_cache[path]

    for path in changed:
        if not path.endswith(".py") or path.startswith("tests/"):
            continue
        stem = Path(path).stem
        if stem == "__init__":
            stem = Path(path).parent.name
        prefix = f"test_{stem}"
        selected.update(
            test for test in tests
            if Path(test).name == f"{prefix}.py" or Path(test).name.startswith(f"{prefix}_")
        )
        needles = _import_needles(path)
        selected.update(test for test in tests if any(needle in read(test) for needle in needles))

    if global_change:
        selected.update(tests)
    return sorted(selected), global_change


def parse_failures(output: str) -> frozenset[str]:
    """Pull pytest node ids out of the canonical runner's summary."""
    return frozenset(match.group(1) for match in _FAILURE_RE.finditer(output))


def run_tests(repo: Path, tests: Iterable[str], *, jobs: int | None = None) -> RunResult:
    tests = list(tests)
    command = [str(repo / "scripts" / "run_tests.sh")]
    if jobs is not None:
        command.extend(["-j", str(jobs)])
    command.extend(tests)
    proc = subprocess.run(
        command,
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=7200,
        check=False,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return RunResult(proc.returncode, output, parse_failures(output))


def load_baseline(path: Path) -> frozenset[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise VerifyError(f"baseline does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise VerifyError(f"baseline is not valid JSON: {path}: {exc}") from exc
    failures = data.get("failures") if isinstance(data, dict) else data
    if not isinstance(failures, list) or not all(isinstance(item, str) for item in failures):
        raise VerifyError(f"baseline must contain a list of test ids: {path}")
    return frozenset(failures)


def save_baseline(path: Path, failures: Iterable[str], *, base: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"version": 1, "base": base, "failures": sorted(set(failures))},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def compare_baseline(
    observed: Iterable[str], baseline: Iterable[str],
) -> tuple[list[str], list[str]]:
    observed_set = set(observed)
    baseline_set = set(baseline)
    return sorted(observed_set - baseline_set), sorted(baseline_set - observed_set)


def _resolve_base(repo: Path, requested: str) -> tuple[str, str]:
    for candidate in (requested, "origin/main", "main"):
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"],
            cwd=repo,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            continue
        merge_base = subprocess.run(
            ["git", "merge-base", "HEAD", candidate],
            cwd=repo,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        if merge_base.returncode == 0 and merge_base.stdout.strip():
            return merge_base.stdout.strip(), candidate
    return _git(repo, "rev-parse", "HEAD").strip(), "HEAD"


def _read_text(repo: Path) -> Callable[[str], str]:
    def read(path: str) -> str:
        try:
            return (repo / path).read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            return ""
    return read


def _print_run_summary(result: RunResult, *, limit: int = 40) -> None:
    print(f"runner exit: {result.returncode}")
    if result.failures:
        print("failures:")
        for failure in sorted(result.failures):
            print(f"  {failure}")
    lines = result.output.splitlines()
    if lines:
        print("runner output (tail):")
        for line in lines[-limit:]:
            print(f"  {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=_DEFAULT_BASE,
                        help=f"Revision to diff against (default: {_DEFAULT_BASE})")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline", type=Path,
                      help="Recorded failures that are expected on this branch")
    mode.add_argument("--record", type=Path,
                      help="Write the observed failures as a new baseline")
    parser.add_argument("--jobs", type=int, help="Pass through to scripts/run_tests.sh")
    parser.add_argument("--dry-run", action="store_true", help="Print the selection without running")
    parser.add_argument("--verbose", action="store_true", help="Print the full runner output")
    args = parser.parse_args(argv)

    try:
        base_sha, base_label = _resolve_base(REPO, args.base)
        changed = changed_paths(REPO, base_sha)
        if not changed:
            print(f"No changes against {base_label} ({base_sha[:12]}).")
            return 0
        tests, global_change = select_tests(changed, _test_files(REPO), _read_text(REPO))
        print(f"base: {base_label} ({base_sha[:12]})")
        print(f"changed files: {len(changed)}")
        for path in changed:
            print(f"  {path}")
        if global_change:
            print("shared test infrastructure changed; selecting the full suite")
        if not tests:
            print("no impacted tests found.")
            if args.record:
                save_baseline(args.record, [], base=base_sha)
                print(f"recorded empty baseline: {args.record}")
            return 0
        print(f"impacted tests: {len(tests)}")
        for path in tests:
            print(f"  {path}")
        baseline = load_baseline(args.baseline) if args.baseline else None
        if args.dry_run:
            return 0
        result = run_tests(REPO, tests, jobs=args.jobs)
        if args.verbose:
            print(result.output)
        else:
            _print_run_summary(result)
        if args.record:
            if result.returncode != 0 and not result.failures:
                raise VerifyError("runner failed without reporting test ids; refusing to record a blank baseline")
            save_baseline(args.record, result.failures, base=base_sha)
            print(f"recorded {len(result.failures)} failure(s): {args.record}")
            return 0
        if baseline is not None:
            new, fixed = compare_baseline(result.failures, baseline)
            if new:
                print(f"new failures ({len(new)}):")
                for failure in new:
                    print(f"  {failure}")
            if fixed:
                print(f"baseline failures no longer observed ({len(fixed)}):")
                for failure in fixed:
                    print(f"  {failure}")
            if new or (result.returncode != 0 and not result.failures):
                return 1
            print("no new failures against the baseline.")
            return 0
        if result.returncode != 0 or result.failures:
            print("verification failed; record a baseline only after triaging these failures.")
            return 1
        print("verification passed.")
        return 0
    except VerifyError as exc:
        print(f"verify_change: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
