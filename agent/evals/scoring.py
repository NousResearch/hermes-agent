"""Deterministic scoring primitives for the Hermes eval subsystem."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .types import CheckResult, CheckType, DeterministicCheck


def run_check(check: DeterministicCheck, workdir: str) -> CheckResult:
    """Execute a single deterministic check and return the result."""
    try:
        return _DISPATCH[check.check_type](check, workdir)
    except Exception as exc:
        return CheckResult(
            check=check,
            passed=False,
            actual=None,
            message=f"check raised: {exc}",
        )


def score_checks(results: list[CheckResult]) -> float:
    """Compute a weighted score in [0.0, 1.0] from check results."""
    if not results:
        return 0.0
    total_weight = sum(r.check.weight for r in results)
    if total_weight == 0:
        return 0.0
    earned = sum(r.check.weight for r in results if r.passed)
    return earned / total_weight


# ── Check implementations ────────────────────────────────────────────────


def _check_file_exists(check: DeterministicCheck, workdir: str) -> CheckResult:
    path = os.path.join(workdir, check.target)
    exists = os.path.isfile(path)
    return CheckResult(
        check=check,
        passed=exists,
        actual=exists,
        message="" if exists else f"file not found: {check.target}",
    )


def _check_file_not_exists(check: DeterministicCheck, workdir: str) -> CheckResult:
    path = os.path.join(workdir, check.target)
    exists = os.path.isfile(path)
    return CheckResult(
        check=check,
        passed=not exists,
        actual=exists,
        message="" if not exists else f"file unexpectedly exists: {check.target}",
    )


def _check_content_contains(check: DeterministicCheck, workdir: str) -> CheckResult:
    path = os.path.join(workdir, check.target)
    try:
        content = _read_file(path)
    except FileNotFoundError:
        return CheckResult(check=check, passed=False, message=f"file not found: {check.target}")
    found = check.expected in content
    return CheckResult(
        check=check,
        passed=found,
        actual=content[:200] if not found else None,
        message="" if found else f"expected substring not found in {check.target}",
    )


def _check_content_not_contains(check: DeterministicCheck, workdir: str) -> CheckResult:
    path = os.path.join(workdir, check.target)
    try:
        content = _read_file(path)
    except FileNotFoundError:
        return CheckResult(check=check, passed=True, message="file not found (vacuously true)")
    found = check.expected in content
    return CheckResult(
        check=check,
        passed=not found,
        actual=None,
        message="" if not found else f"unexpected substring found in {check.target}",
    )


def _check_content_equals(check: DeterministicCheck, workdir: str) -> CheckResult:
    path = os.path.join(workdir, check.target)
    try:
        content = _read_file(path)
    except FileNotFoundError:
        return CheckResult(check=check, passed=False, message=f"file not found: {check.target}")
    passed = content == check.expected
    return CheckResult(
        check=check,
        passed=passed,
        actual=content[:200] if not passed else None,
        message="" if passed else "content mismatch",
    )


def _check_json_valid(check: DeterministicCheck, workdir: str) -> CheckResult:
    path = os.path.join(workdir, check.target)
    try:
        content = _read_file(path)
        json.loads(content)
        return CheckResult(check=check, passed=True)
    except FileNotFoundError:
        return CheckResult(check=check, passed=False, message=f"file not found: {check.target}")
    except json.JSONDecodeError as exc:
        return CheckResult(check=check, passed=False, message=f"invalid JSON: {exc}")


def _check_json_key_exists(check: DeterministicCheck, workdir: str) -> CheckResult:
    path = os.path.join(workdir, check.target)
    try:
        content = _read_file(path)
        data = json.loads(content)
    except FileNotFoundError:
        return CheckResult(check=check, passed=False, message=f"file not found: {check.target}")
    except json.JSONDecodeError as exc:
        return CheckResult(check=check, passed=False, message=f"invalid JSON: {exc}")
    # Support dotted key paths like "meta.version"
    keys = check.expected.split(".") if isinstance(check.expected, str) else [check.expected]
    obj: Any = data
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            return CheckResult(
                check=check, passed=False,
                message=f"key path '{check.expected}' not found",
            )
    return CheckResult(check=check, passed=True, actual=obj)


def _check_regex_match(check: DeterministicCheck, workdir: str) -> CheckResult:
    path = os.path.join(workdir, check.target)
    try:
        content = _read_file(path)
    except FileNotFoundError:
        return CheckResult(check=check, passed=False, message=f"file not found: {check.target}")
    matched = bool(re.search(check.expected, content))
    return CheckResult(
        check=check,
        passed=matched,
        message="" if matched else f"pattern '{check.expected}' not matched",
    )


def _check_exit_code(check: DeterministicCheck, workdir: str) -> CheckResult:
    # target = path to a file containing the exit code as text
    path = os.path.join(workdir, check.target)
    try:
        content = _read_file(path).strip()
        actual_code = int(content)
    except (FileNotFoundError, ValueError) as exc:
        return CheckResult(check=check, passed=False, message=f"exit code check failed: {exc}")
    expected_code = int(check.expected)
    passed = actual_code == expected_code
    return CheckResult(
        check=check,
        passed=passed,
        actual=actual_code,
        message="" if passed else f"exit code {actual_code} != expected {expected_code}",
    )


def _read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


_DISPATCH = {
    CheckType.FILE_EXISTS: _check_file_exists,
    CheckType.FILE_NOT_EXISTS: _check_file_not_exists,
    CheckType.CONTENT_CONTAINS: _check_content_contains,
    CheckType.CONTENT_NOT_CONTAINS: _check_content_not_contains,
    CheckType.CONTENT_EQUALS: _check_content_equals,
    CheckType.JSON_VALID: _check_json_valid,
    CheckType.JSON_KEY_EXISTS: _check_json_key_exists,
    CheckType.REGEX_MATCH: _check_regex_match,
    CheckType.EXIT_CODE: _check_exit_code,
}
