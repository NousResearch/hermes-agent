"""Shared helpers for verifiable Dev acceptance criteria."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, Dict


VERIFICATION_METHODS = {"test", "command", "probe", "manual"}
MACHINE_CHECKABLE_METHODS = {"test", "command"}
SHELL_META_RE = re.compile(r"[;&|<>`$()]")
ALLOWED_VERIFICATION_COMMAND_SHAPES = [
    "scripts/run_tests.sh ...",
    "make test",
    "make test-smoke",
    "make build",
    "swift test ...",
    "pytest tests/...",
]

ACCEPTANCE_CRITERION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["statement", "verification_method", "verification_detail", "machine_checkable"],
    "properties": {
        "statement": {"type": "string"},
        "verification_method": {"type": "string", "enum": sorted(VERIFICATION_METHODS)},
        "verification_detail": {"type": "string"},
        "machine_checkable": {"type": "boolean"},
        "note": {"type": "string"},
    },
}


def allowlisted_command(command: Any) -> tuple[bool, str, str]:
    text = str(command or "").strip()
    if not text:
        return False, "", "verification_detail is empty."
    if SHELL_META_RE.search(text):
        return False, text, "Command contains shell metacharacters."
    try:
        parts = shlex.split(text)
    except ValueError as exc:
        return False, text, f"Command cannot be parsed safely: {exc}."
    if not parts:
        return False, text, "Command is empty."
    if any(_unsafe_token(part) for part in parts):
        return False, text, "Command contains unsafe paths or tokens."
    head = parts[0]
    if head == "scripts/run_tests.sh":
        return True, " ".join(shlex.quote(part) for part in parts), ""
    if head == "make" and len(parts) == 2 and parts[1] in {"test", "test-smoke", "build"}:
        return True, text, ""
    if head == "swift" and len(parts) >= 2 and parts[1] == "test":
        return True, " ".join(shlex.quote(part) for part in parts), ""
    if head == "pytest" and len(parts) >= 2 and any(part.startswith("tests/") for part in parts[1:]):
        return True, " ".join(shlex.quote(part) for part in parts), ""
    return False, text, "Command is not in the v1 verification allowlist."


def normalize_acceptance_criteria(value: Any, *, fallback_machine_checkable: bool = False) -> list[Dict[str, Any]]:
    """Normalize legacy strings and structured criteria into the v2 shape."""

    if not isinstance(value, list):
        return []
    criteria: list[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            statement = item.strip()
            if statement:
                criteria.append({
                    "statement": statement,
                    "verification_method": "manual",
                    "verification_detail": "Review manually.",
                    "machine_checkable": bool(fallback_machine_checkable),
                })
            continue
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement") or item.get("title") or item.get("description") or "").strip()
        if not statement:
            continue
        method = str(item.get("verification_method") or "").strip().lower()
        if method not in VERIFICATION_METHODS:
            method = "manual"
        detail = str(item.get("verification_detail") or "").strip()
        if not detail:
            detail = "Review manually." if method == "manual" else statement
        normalized = {
            "statement": statement,
            "verification_method": method,
            "verification_detail": detail,
            "machine_checkable": bool(item.get("machine_checkable")) and method != "manual",
        }
        note = str(item.get("note") or "").strip()
        if note:
            normalized["note"] = note
        criteria.append(normalized)
    return criteria


def validate_and_downgrade_criteria(
    criteria: Any,
    *,
    repo_roots: list[str] | None = None,
) -> tuple[list[Dict[str, Any]], list[str]]:
    """Conservatively downgrade clearly non-verifiable machine-checkable criteria."""

    roots = _existing_roots(repo_roots)
    warnings: list[str] = []
    validated: list[Dict[str, Any]] = []
    for criterion in normalize_acceptance_criteria(criteria):
        if not criterion.get("machine_checkable") or criterion.get("verification_method") not in MACHINE_CHECKABLE_METHODS:
            validated.append(criterion)
            continue
        allowed, normalized_command, reason = allowlisted_command(criterion.get("verification_detail"))
        if not allowed:
            validated.append(_downgraded_criterion(criterion, reason))
            warnings.append(_downgrade_warning(criterion, reason))
            continue
        missing_paths = _missing_referenced_paths(normalized_command, roots)
        if missing_paths:
            reason = f"Referenced path does not exist under configured repo roots: {', '.join(missing_paths)}."
            validated.append(_downgraded_criterion(criterion, reason))
            warnings.append(_downgrade_warning(criterion, reason))
            continue
        validated.append(criterion)
    return validated, warnings


def acceptance_criteria_to_strings(value: Any) -> list[str]:
    """Project structured criteria to strings for worker task prompts/contracts."""

    strings: list[str] = []
    for criterion in normalize_acceptance_criteria(value):
        statement = criterion["statement"]
        method = criterion["verification_method"]
        detail = criterion["verification_detail"]
        machine = "machine-checkable" if criterion["machine_checkable"] else "manual"
        if method == "manual":
            strings.append(statement)
        else:
            strings.append(f"{statement} (verify via {method}: {detail}; {machine})")
    return strings


def _unsafe_token(token: str) -> bool:
    if not token or token.startswith("-"):
        return False
    if "=" in token and not token.startswith(("tests/", "scripts/")):
        return True
    path = Path(token)
    if path.is_absolute():
        return True
    return ".." in path.parts


def _downgraded_criterion(criterion: Dict[str, Any], reason: str) -> Dict[str, Any]:
    original_detail = str(criterion.get("verification_detail") or "").strip()
    detail = f"Manual review required; original verification_detail was not machine-checkable: {original_detail}"
    return {
        **criterion,
        "verification_method": "manual",
        "verification_detail": detail,
        "machine_checkable": False,
        "note": reason,
    }


def _downgrade_warning(criterion: Dict[str, Any], reason: str) -> str:
    statement = str(criterion.get("statement") or "Acceptance criterion").strip()
    return f"{statement}: downgraded from machine-checkable because {reason}"


def _existing_roots(repo_roots: list[str] | None) -> list[Path]:
    roots: list[Path] = []
    for root in repo_roots or []:
        text = str(root or "").strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if path.exists() and path.is_dir():
            roots.append(path)
    return roots


def _missing_referenced_paths(command: str, roots: list[Path]) -> list[str]:
    if not roots:
        return []
    try:
        parts = shlex.split(command)
    except ValueError:
        return []
    paths = _referenced_path_args(parts)
    return [path for path in paths if not any((root / _path_for_existence(path)).exists() for root in roots)]


def _referenced_path_args(parts: list[str]) -> list[str]:
    if not parts:
        return []
    head = parts[0]
    if head == "scripts/run_tests.sh":
        return _path_args_until_pytest_separator(parts[1:])
    if head == "pytest":
        return [
            part
            for part in parts[1:]
            if _is_unambiguous_relative_path(part)
        ]
    return []


def _path_args_until_pytest_separator(parts: list[str]) -> list[str]:
    paths: list[str] = []
    for part in parts:
        if part == "--":
            break
        if _is_unambiguous_relative_path(part):
            paths.append(part)
    return paths


def _is_unambiguous_relative_path(token: str) -> bool:
    if not token or token.startswith("-"):
        return False
    if any(char in token for char in "*?[]"):
        return False
    path = Path(token)
    if path.is_absolute() or ".." in path.parts:
        return False
    return token.startswith(("tests/", "test/", "scripts/")) or token.endswith((".py", ".swift"))


def _path_for_existence(token: str) -> str:
    return token.split("::", 1)[0]
