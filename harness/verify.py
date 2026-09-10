"""Independent verification: evidence hierarchy, not model claims.

Strength order (strongest first): tests, type checker, runtime behavior,
static inspection, configuration, agent reasoning. ``"It works"`` from the
model scores lowest and can never alone pass a gate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Sequence

from .state import VerificationCheck, VerificationResult


class CheckStrength:
    CLAIM = 0
    CONFIG = 1
    STATIC = 2
    RUNTIME = 3
    TYPECHECK = 4
    TESTS = 5


def verify(
    checks: Sequence[VerificationCheck], success_criteria: Sequence[str]
) -> VerificationResult:
    collected = list(checks)
    failures = [f"{c.name}: {c.detail or 'failed'}" for c in collected if not c.passed]
    confidence = 0.0
    if collected and not failures:
        # Strongest available evidence sets the confidence (system design
        # §32); a bare claim stays at the floor and never verifies alone.
        strongest = max(c.strength for c in collected)
        confidence = round(0.5 + 0.1 * strongest, 2)
    return VerificationResult(
        passed=bool(collected) and not failures and bool(success_criteria),
        checks=collected,
        failures=failures,
        confidence=confidence,
    )


def completion_allowed(
    goal_done: bool, criteria_done: bool, verified: bool, no_blocker: bool
) -> bool:
    return bool(goal_done and criteria_done and verified and no_blocker)


def pytest_check(
    targets: Sequence[str], cwd: str | Path, timeout_s: int = 300
) -> VerificationCheck:
    """Run pytest as evidence. The subprocess result is the check."""
    cmd = [sys.executable, "-m", "pytest", "-q", *targets]
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout_s
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return VerificationCheck(
            name="pytest",
            passed=False,
            detail=str(exc)[:500],
            strength=CheckStrength.TESTS,
        )
    tail = (proc.stdout + proc.stderr)[-2000:]
    return VerificationCheck(
        name="pytest",
        passed=proc.returncode == 0,
        detail=tail,
        strength=CheckStrength.TESTS,
    )


def command_check(
    name: str, command: Sequence[str], cwd: str | Path, timeout_s: int = 120
) -> VerificationCheck:
    """Run an arbitrary verification command (build, typecheck, migrate)."""
    try:
        proc = subprocess.run(
            list(command),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return VerificationCheck(
            name=name,
            passed=False,
            detail=str(exc)[:500],
            strength=CheckStrength.RUNTIME,
        )
    return VerificationCheck(
        name=name,
        passed=proc.returncode == 0,
        detail=(proc.stdout + proc.stderr)[-2000:],
        strength=CheckStrength.RUNTIME,
    )


def file_contains_check(name: str, path: str | Path, needle: str) -> VerificationCheck:
    """Static source inspection: the file must exist and contain the needle."""
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return VerificationCheck(
            name=name,
            passed=False,
            detail=str(exc)[:300],
            strength=CheckStrength.STATIC,
        )
    ok = needle in content
    return VerificationCheck(
        name=name,
        passed=ok,
        detail="found" if ok else f"missing {needle!r}",
        strength=CheckStrength.STATIC,
    )


Checker = Callable[[], VerificationCheck]


def run_all(checkers: Sequence[Checker]) -> List[VerificationCheck]:
    return [check() for check in checkers]
