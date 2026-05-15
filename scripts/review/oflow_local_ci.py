#!/usr/bin/env python3
"""Safe, local-only CI helper for the Oflow pilot.

The helper intentionally stays inside repository metadata and checked-in files.
It never reads secret-bearing paths, opens database files, starts services,
connects to remotes, deploys, or mutates runtime state.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import py_compile
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency in local shells
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM = "origin/main"

SECRET_PATH_PATTERNS = (
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
    "**/*secret*",
    "**/*secrets*",
    "**/*credential*",
    "**/*credentials*",
)

RUNTIME_PATH_PATTERNS = (
    "runtime/**",
    "prod/**",
    "production/**",
    "deploy/**",
    "deployments/**",
    "scripts/deploy*",
    "scripts/**/deploy*",
    "migrations/**",
    "backfills/**",
)

MUTATION_DB_PATH_PATTERNS = (
    "migrations/**",
    "backfills/**",
    "**/*sqlite*",
    "**/*postgres*",
    "**/*psql*",
    "**/*migration*",
    "**/*backfill*",
)

# Checked against text content of changed, non-secret files. Keep patterns broad
# and readable because this is a safety gate, not a style linter.
DENYLIST_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (".env access", re.compile(r"(?<![A-Za-z0-9_-])(?:cat|grep|source|\.\s+|cp|less|more|tail|head)\s+[^\n;&|]*\.env(?:\b|[./_-])", re.I)),
    ("deploy script invocation", re.compile(r"(?:^|[\s./])(?:deploy(?:\.sh|\.py)?|scripts/deploy[^\s;&|]*)\b", re.I)),
    ("systemctl", re.compile(r"\bsystemctl\b", re.I)),
    ("docker compose up/down", re.compile(r"\bdocker\s+compose\s+(?:up|down)\b|\bdocker-compose\s+(?:up|down)\b", re.I)),
    ("ssh", re.compile(r"(?<![\w-])ssh\s+(?!-V\b)", re.I)),
    ("sqlite mutation", re.compile(r"\bsqlite3?\b.*\b(?:insert|update|delete|drop|alter|create|replace|vacuum|reindex)\b", re.I | re.S)),
    ("postgres mutation", re.compile(r"\b(?:psql|postgres)\b.*\b(?:insert|update|delete|drop|alter|create|replace|truncate|vacuum|reindex)\b", re.I | re.S)),
)

SUMMARY_RUNTIME_PATTERNS = RUNTIME_PATH_PATTERNS + (
    ".github/workflows/**",
    "scripts/review/**",
)

# These files intentionally contain denylist words as documentation, tests, or
# the denylist implementation itself. They still get path-level checks; only
# command-content scanning is skipped to avoid fixture/self-reference failures.
CONTENT_SCAN_EXEMPT_PATHS = (
    ".gitignore",
    "docs/**",
    "tests/**",
    "scripts/review/oflow_local_ci.py",
    "scripts/review/oflow_tracked_file_handoff_check.py",
)


@dataclass(frozen=True)
class Violation:
    file: str
    check: str
    line: int | None
    detail: str


def run_git(args: Sequence[str], repo: Path = REPO_ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def normalize(path: Path | str) -> str:
    return str(path).replace(os.sep, "/")


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def changed_files(upstream: str = DEFAULT_UPSTREAM, repo: Path = REPO_ROOT) -> list[str]:
    merge_base = run_git(["merge-base", upstream, "HEAD"], repo=repo).stdout.strip()
    committed = run_git(["diff", "--name-only", f"{merge_base}..HEAD"], repo=repo).stdout
    unstaged = run_git(["diff", "--name-only"], repo=repo).stdout
    staged = run_git(["diff", "--cached", "--name-only"], repo=repo).stdout
    untracked = run_git(["ls-files", "--others", "--exclude-standard"], repo=repo).stdout
    files = [line.strip() for line in (committed + "\n" + unstaged + "\n" + staged + "\n" + untracked).splitlines() if line.strip()]
    return sorted(dict.fromkeys(files))


def text_line_number(text: str, start: int) -> int:
    return text.count("\n", 0, start) + 1


def scan_file(path: str, repo: Path = REPO_ROOT) -> list[Violation]:
    rel = normalize(path)
    violations: list[Violation] = []

    if matches_any(rel, SECRET_PATH_PATTERNS):
        return [Violation(rel, "secret path", None, "secret-bearing paths are not read by this helper")]
    if matches_any(rel, RUNTIME_PATH_PATTERNS):
        violations.append(Violation(rel, "runtime path", None, "runtime/prod/deploy paths are out of scope for local CI"))
    if matches_any(rel, MUTATION_DB_PATH_PATTERNS):
        violations.append(Violation(rel, "database mutation path", None, "database migration/backfill/mutation paths are out of scope"))

    if matches_any(rel, CONTENT_SCAN_EXEMPT_PATHS):
        return violations

    full_path = repo / rel
    if not full_path.exists() or not full_path.is_file():
        return violations

    try:
        data = full_path.read_bytes()
    except OSError as exc:
        violations.append(Violation(rel, "read error", None, str(exc)))
        return violations

    if b"\0" in data:
        return violations

    text = data.decode("utf-8", errors="ignore")
    for label, pattern in DENYLIST_PATTERNS:
        for match in pattern.finditer(text):
            violations.append(
                Violation(
                    rel,
                    label,
                    text_line_number(text, match.start()),
                    match.group(0).splitlines()[0][:160],
                )
            )
    return violations


def run_denylist(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    files = args.files or changed_files(args.upstream, repo=repo)
    violations: list[Violation] = []
    for file_name in files:
        violations.extend(scan_file(file_name, repo=repo))

    payload = {"passed": not violations, "violations": [asdict(v) for v in violations], "files_scanned": files}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not violations else 1


def python_files_for_compile(repo: Path, files: Sequence[str]) -> list[str]:
    if files:
        candidates = files
    else:
        candidates = changed_files(DEFAULT_UPSTREAM, repo=repo)
    return [f for f in candidates if f.endswith(".py") and (repo / f).is_file() and not matches_any(f, SECRET_PATH_PATTERNS)]


def run_py_compile(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    files = python_files_for_compile(repo, args.files)
    failures: list[dict[str, str]] = []
    for rel in files:
        try:
            py_compile.compile(str(repo / rel), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append({"file": rel, "error": str(exc)})
    print(json.dumps({"passed": not failures, "files": files, "failures": failures}, indent=2, sort_keys=True))
    return 0 if not failures else 1


def run_diff_check(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    proc = run_git(["diff", "--check", args.upstream, "HEAD"], repo=repo, check=False)
    print(proc.stdout, end="")
    print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


def run_workflow_yaml(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    workflows = sorted((repo / ".github" / "workflows").glob("*.y*ml"))
    failures: list[dict[str, str]] = []
    for workflow in workflows:
        try:
            text = workflow.read_text(encoding="utf-8")
            if yaml is not None:
                parsed = yaml.safe_load(text)
                if not isinstance(parsed, dict) or "jobs" not in parsed:
                    failures.append({"file": normalize(workflow.relative_to(repo)), "error": "workflow YAML has no jobs mapping"})
            else:
                # Minimal structural fallback when PyYAML is unavailable.
                if not re.search(r"(?m)^jobs:\s*$", text):
                    failures.append({"file": normalize(workflow.relative_to(repo)), "error": "workflow YAML has no jobs: key"})
        except Exception as exc:
            failures.append({"file": normalize(workflow.relative_to(repo)), "error": str(exc)})
    print(json.dumps({"passed": not failures, "files_checked": [normalize(w.relative_to(repo)) for w in workflows], "failures": failures}, indent=2, sort_keys=True))
    return 0 if not failures else 1


def run_pytest_subset(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    targets = args.targets or ["tests/review/test_oflow_local_ci.py"]
    cmd = [sys.executable, "-m", "pytest", *targets]
    proc = subprocess.run(cmd, cwd=repo)
    return proc.returncode


def run_summary(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    files = changed_files(args.upstream, repo=repo)
    touched_areas = sorted({
        "workflow" if matches_any(f, (".github/workflows/**",)) else
        "runtime" if matches_any(f, RUNTIME_PATH_PATTERNS) else
        "review-helper" if matches_any(f, ("scripts/review/**",)) else
        "docs" if matches_any(f, ("docs/**",)) else
        "tests" if matches_any(f, ("tests/**",)) else
        "other"
        for f in files
    })
    payload = {
        "passed": True,
        "changed_files": files,
        "touched_workflow_runtime_areas": touched_areas,
        "hard_stops": {
            "no_merge": True,
            "no_deploy": True,
            "no_restart": True,
            "no_env_inspection": True,
            "no_secret_inspection": True,
            "no_runtime_monitoring": True,
            "no_production_probes": True,
            "no_provider_api_calls": True,
            "no_ssh": True,
            "no_db_access": True,
            "no_trading_or_order_impact": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repo", default=str(REPO_ROOT), help="repository root")
        p.add_argument("--upstream", default=DEFAULT_UPSTREAM, help="upstream ref for changed-file discovery")

    deny = sub.add_parser("denylist", help="scan changed files for local-CI safety denylist patterns")
    add_common(deny)
    deny.add_argument("files", nargs="*", help="explicit repository-relative files to scan")
    deny.set_defaults(func=run_denylist)

    pyc = sub.add_parser("py-compile", help="compile changed Python files")
    add_common(pyc)
    pyc.add_argument("files", nargs="*", help="explicit repository-relative Python files")
    pyc.set_defaults(func=run_py_compile)

    diff = sub.add_parser("diff-check", help="run git diff --check")
    add_common(diff)
    diff.set_defaults(func=run_diff_check)

    workflow = sub.add_parser("workflow-yaml", help="validate workflow YAML structure")
    add_common(workflow)
    workflow.set_defaults(func=run_workflow_yaml)

    pytest_cmd = sub.add_parser("pytest-subset", help="run safe fixture/local pytest subsets")
    pytest_cmd.add_argument("--repo", default=str(REPO_ROOT), help="repository root")
    pytest_cmd.add_argument("targets", nargs="*", help="pytest targets")
    pytest_cmd.set_defaults(func=run_pytest_subset)

    summary = sub.add_parser("summary", help="write artifact-only CI summary")
    add_common(summary)
    summary.add_argument("--output", type=Path, default=Path("artifacts/oflow-local-ci-summary.json"))
    summary.set_defaults(func=run_summary)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
