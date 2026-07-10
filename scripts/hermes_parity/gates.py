"""Parity gate ladder implementation."""

from __future__ import annotations

import json
import py_compile
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from . import forkdelta, gitops, lint_merge_traps, lint_unbound, state


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str
    seconds: float
    metadata: dict[str, object] = field(default_factory=dict)
    repro: str = ""
    skipped: bool = False


GateFn = Callable[[Path], GateResult]


def _record(worktree: Path, result: GateResult) -> None:
    path = worktree / "gates.jsonl"
    payload = {
        "ts": time.time(),
        "gate": result.name,
        "passed": result.passed,
        "detail": result.detail,
        "seconds": result.seconds,
        "metadata": result.metadata,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _timed(name: str, repro: str, fn: Callable[[], tuple[bool, str, dict[str, object] | None]]) -> GateResult:
    start = time.monotonic()
    try:
        passed, detail, metadata = fn()
    except Exception as exc:  # pragma: no cover - defensive gate boundary
        passed, detail, metadata = False, f"{type(exc).__name__}: {exc}", {}
    return GateResult(name, passed, detail, time.monotonic() - start, metadata or {}, repro)


def python_files(repo: Path) -> list[Path]:
    result = gitops.run_git(repo, ["ls-files", "*.py"], check=True)
    return [repo / line for line in result.stdout.splitlines() if line]


def gate_markers(repo: Path) -> GateResult:
    def run() -> tuple[bool, str, dict[str, object]]:
        lines = gitops.conflict_marker_lines(repo)
        return (not lines, f"{len(lines)} marker line(s)", {"samples": lines[:20]})

    return _timed("markers", "git grep -nE '^(<<<<<<<|=======|>>>>>>>)' && git diff --diff-filter=U", run)


def gate_imports(repo: Path) -> GateResult:
    def run() -> tuple[bool, str, dict[str, object]]:
        failures: list[str] = []
        for path in python_files(repo):
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append(f"{path.relative_to(repo)}: {exc.msg}")
                if len(failures) >= 50:
                    break
        return (not failures, f"{len(failures)} compile failure(s)", {"failures": failures})

    return _timed("imports", "python3.11 -m py_compile $(git ls-files '*.py')", run)


def gate_unbound(repo: Path, *, strict: bool = False) -> GateResult:
    def run() -> tuple[bool, str, dict[str, object]]:
        files = python_files(repo)
        issues = lint_unbound.lint_paths(files, repo=repo)
        trap_issues = lint_merge_traps.lint_paths(files, repo=repo)
        samples = [
            f"{issue.path}:{issue.line}:{issue.column}: {issue.name}"
            for issue in issues[:50]
        ]
        trap_samples = [
            f"{issue.path}:{issue.line}: [{issue.kind}] {issue.detail}"
            for issue in trap_issues[:50]
        ]
        total = len(issues) + len(trap_issues)
        return (
            total == 0 or not strict,
            f"{len(issues)} unbound + {len(trap_issues)} merge-trap issue(s)",
            {"samples": samples, "trap_samples": trap_samples, "strict": strict},
        )

    return _timed("traps", "python3.11 -m hermes_parity gates --stage traps --strict", run)


def gate_manifest_forkdelta(repo: Path, *, base: str | None, fork_ref: str) -> GateResult:
    def run() -> tuple[bool, str, dict[str, object]]:
        manifest = repo / forkdelta.DEFAULT_MANIFEST
        if not manifest.exists():
            return False, "manifest missing", {"path": str(forkdelta.DEFAULT_MANIFEST)}
        resolved_base = base or gitops.merge_base(repo, "origin/main", fork_ref)
        # RC1 (Momus pass-2): the non-vacuous trigger is a fork-side file the
        # MERGE changed or deleted relative to fork_ref — NOT merely any file
        # the fork ever touched (all fork-changed files differ from the
        # merge-base by definition; intersecting with base..HEAD would fire on
        # the entire fork delta and make the gate a false-positive machine).
        # Upstream *modifications* of fork files are caught by the manifest
        # `tests` nodes, not by this path check; this arm closes the
        # delete/rename (DU/UD) subset.
        # Working-tree diff UNION ref..HEAD diff: gates run both while the
        # merge is staged-but-uncommitted (HEAD still the fork tip → worktree
        # diff sees it) and after the merge commit (HEAD moved → ref..HEAD
        # sees it).
        touched = set(gitops.worktree_changed_files(repo, fork_ref)) | set(
            gitops.changed_files(repo, fork_ref, "HEAD")
        )
        acked = state.acked_paths(repo)
        report = forkdelta.compute_fork_delta(repo, base=resolved_base, fork_ref=fork_ref, touched_paths=touched)
        uncovered = [path for path in report.uncovered_paths if path not in acked]
        acknowledged = [path for path in report.uncovered_paths if path in acked]
        tests = forkdelta.manifest_nodeids(manifest)
        test_proc = None
        if tests:
            script = repo / "scripts" / "run_tests.sh"
            test_proc = subprocess.run(
                [str(script), *tests],
                cwd=repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        passed = not uncovered and (test_proc is None or test_proc.returncode == 0)
        return (
            passed,
            f"{len(report.covered_paths)} covered, {len(uncovered)} uncovered, {len(acknowledged)} acknowledged path(s)",
            {
                "base": report.base,
                "fork_ref": report.fork_ref,
                "covered_features": list(report.covered_features),
                "uncovered_samples": list(uncovered[:50]),
                "acknowledged": acknowledged,
                "manifest_tests": tests,
                "manifest_test_exit": None if test_proc is None else test_proc.returncode,
            },
        )

    return _timed("manifest+forkdelta", "python3.11 -m hermes_parity gates --stage manifest+forkdelta", run)


def gate_tests(repo: Path, *, tests: Sequence[str]) -> GateResult:
    def run() -> tuple[bool, str, dict[str, object]]:
        script = repo / "scripts" / "run_tests.sh"
        if not script.exists():
            return False, "scripts/run_tests.sh missing", {}
        args = [str(script), *(tests or ["tests/scripts/test_hermes_parity.py"])]
        proc = subprocess.run(
            args,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        tail = proc.stdout.splitlines()[-80:]
        return proc.returncode == 0, f"exit {proc.returncode}", {"tail": tail}

    return _timed("tests", "scripts/run_tests.sh", run)


def gate_linuxonly(repo: Path) -> GateResult:
    def run() -> tuple[bool, str, dict[str, object]]:
        result = gitops.run_git(repo, ["grep", "-nE", r"skipif.*darwin|platform"], check=False)
        if result.returncode == 1:
            return True, "0 linux-only risk line(s)", {"lines": []}
        lines = result.stdout.splitlines()[:100]
        return True, f"{len(lines)} linux-only risk line(s) listed", {"lines": lines}

    return _timed("linuxonly", "git grep -nE 'skipif.*darwin|platform'", run)


ORDERED_STAGE_NAMES = ["markers", "imports", "traps", "manifest+forkdelta", "tests", "linuxonly"]


def selected_stage_names(*, stage: str | None = None, fast: bool = False) -> list[str]:
    names = ORDERED_STAGE_NAMES[:4] if fast else list(ORDERED_STAGE_NAMES)
    if stage:
        if stage not in ORDERED_STAGE_NAMES:
            raise ValueError(f"unknown gate stage: {stage}")
        return [stage]
    return names


def resume_plan(worktree: Path, names: list[str], *, resume: bool) -> list[str]:
    if not resume:
        return names
    valid = state.valid_gate_names_for_current_tree(worktree, ORDERED_STAGE_NAMES)
    return [name for name in names if name not in valid]


def run_gates(
    repo: Path,
    *,
    fast: bool = False,
    stage: str | None = None,
    resume: bool = False,
    strict: bool = False,
    base: str | None = None,
    fork_ref: str = "fork/main",
    tests: Sequence[str] = (),
) -> list[GateResult]:
    names = resume_plan(repo, selected_stage_names(stage=stage, fast=fast), resume=resume)
    runners: dict[str, Callable[[], GateResult]] = {
        "markers": lambda: gate_markers(repo),
        "imports": lambda: gate_imports(repo),
        "traps": lambda: gate_unbound(repo, strict=strict),
        "manifest+forkdelta": lambda: gate_manifest_forkdelta(repo, base=base, fork_ref=fork_ref),
        "tests": lambda: gate_tests(repo, tests=tests),
        "linuxonly": lambda: gate_linuxonly(repo),
    }
    results: list[GateResult] = []
    for name in names:
        result = runners[name]()
        results.append(result)
        _record(repo, result)
        state.record_gate(repo, result.name, ok=result.passed, extra={"detail": result.detail, "repro": result.repro})
        if not result.passed:
            break
    if resume and not results:
        return [GateResult("resume", True, "all requested stages already green at current tree", 0.0, skipped=True)]
    return results


def format_table(results: Sequence[GateResult]) -> str:
    rows = [("gate", "status", "seconds", "detail", "repro")]
    for result in results:
        rows.append((
            result.name,
            "SKIP" if result.skipped else ("PASS" if result.passed else "FAIL"),
            f"{result.seconds:.2f}",
            result.detail,
            result.repro,
        ))
    widths = [max(len(row[index]) for row in rows) for index in range(5)]
    lines = []
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)))
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def parse_gitleaks_version(text: str) -> str | None:
    patterns = [
        r"gitleaks(?:/gitleaks-action)?@([A-Za-z0-9_.-]+)",
        r"gitleaks_version[:=]\s*['\"]?([A-Za-z0-9_.-]+)",
        r"gitleaks\s+version\s+([A-Za-z0-9_.-]+)",
        # Fleet form: a `VER=8.18.4` shell var feeding a gitleaks release
        # download (fleet-secret-scan.yml). Only trust it when 'gitleaks'
        # appears nearby (within a bounded window, across lines) so we don't
        # grab an unrelated VER=.
        r"gitleaks[\s\S]{0,300}?\bVER=v?([0-9]+(?:\.[0-9]+)+)",
        r"\bVER=v?([0-9]+(?:\.[0-9]+)+)[\s\S]{0,300}?gitleaks",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def find_pinned_gitleaks_version(repo: Path) -> str | None:
    workflow_dir = repo / ".github" / "workflows"
    for path in sorted(list(workflow_dir.glob("*.yml")) + list(workflow_dir.glob("*.yaml"))):
        version = parse_gitleaks_version(path.read_text(encoding="utf-8", errors="replace"))
        if version:
            return version
    return None


def ci_reminders(repo: Path) -> list[str]:
    version = find_pinned_gitleaks_version(repo)
    gitleaks = (
        f"gitleaks pinned in CI: {version}"
        if version
        else "⚠️ could not determine pinned gitleaks version — check .github/workflows"
    )
    return [
        gitleaks,
        "contributor-check",
        "config-migration dry-run",
        "tsc on apps/desktop via bash; zsh-zle/exit-194 with 0 errors means tsc never ran",
    ]
