from __future__ import annotations

from typing import Any, Iterable, Optional


PASS_STATUSES = {"pass", "passed", "ok", "green", "success", "succeeded"}


def _as_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _status(value: Any) -> str:
    if isinstance(value, dict):
        raw = value.get("status") or value.get("outcome") or value.get("result")
    else:
        raw = value
    return str(raw or "").strip().lower()


def _is_passed(value: Any) -> bool:
    return _status(value) in PASS_STATUSES


def _name(value: Any, default: str) -> str:
    if isinstance(value, dict):
        for key in ("name", "command", "test", "check", "id"):
            raw = value.get(key)
            if raw:
                return str(raw)
    if isinstance(value, str):
        return value
    return default


def _failure_file(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    for key in ("file", "path", "nodeid_file", "source_file"):
        raw = value.get(key)
        if raw:
            return _normalize_path(str(raw))
    nodeid = value.get("test") or value.get("nodeid")
    if isinstance(nodeid, str) and "::" in nodeid:
        maybe_file = nodeid.split("::", 1)[0]
        if "/" in maybe_file or "\\" in maybe_file or "." in maybe_file:
            return _normalize_path(maybe_file)
    return None


def _message(value: Any) -> str:
    if isinstance(value, dict):
        raw = value.get("message") or value.get("reason") or value.get("error")
        if raw:
            return str(raw)
        return _name(value, "failure")
    return str(value)


def _normalize_path(path: str) -> str:
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    while "//" in p:
        p = p.replace("//", "/")
    return p.rstrip("/")


def _path_overlaps(path: str, roots: Iterable[str]) -> bool:
    candidate = _normalize_path(path).lower()
    if not candidate:
        return False
    for root in roots:
        current = _normalize_path(str(root)).lower()
        if not current:
            continue
        if candidate == current or candidate.startswith(current.rstrip("/") + "/"):
            return True
        if current == candidate or current.startswith(candidate.rstrip("/") + "/"):
            return True
    return False


def _failure_payload(
    value: Any,
    *,
    classification: str,
    related: bool,
) -> dict[str, Any]:
    payload = {
        "classification": classification,
        "related": related,
        "name": _name(value, "failure"),
        "message": _message(value),
    }
    file_path = _failure_file(value)
    if file_path:
        payload["file"] = file_path
    return payload


def _add_named_failures(
    blocking: list[dict[str, Any]],
    failures: Any,
    *,
    classification: str,
) -> None:
    for item in _as_items(failures):
        blocking.append(
            _failure_payload(
                item,
                classification=classification,
                related=True,
            )
        )


def _failed_checks(checks: Any) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for index, check in enumerate(_as_items(checks), start=1):
        if not _is_passed(check):
            failed.append(
                _failure_payload(
                    check,
                    classification="required_check_failed",
                    related=True,
                )
            )
            failed[-1]["name"] = _name(check, f"required_check_{index}")
    return failed


def evaluate_scoped_acceptance(
    *,
    touched_files: Any = None,
    task_scope_files: Any = None,
    targeted_tests: Any = None,
    required_checks: Any = None,
    full_suite_failures: Any = None,
    safety_failures: Any = None,
    contract_failures: Any = None,
    migration_failures: Any = None,
    unknown_failures: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    touched = [_normalize_path(str(p)) for p in _as_items(touched_files) if str(p).strip()]
    scope = [_normalize_path(str(p)) for p in _as_items(task_scope_files) if str(p).strip()]
    targeted = _as_items(targeted_tests)
    required = _as_items(required_checks)
    caveats: list[dict[str, Any]] = []
    blocking: list[dict[str, Any]] = []

    if not targeted:
        blocking.append(
            {
                "classification": "targeted_tests_missing",
                "related": True,
                "name": "targeted_tests",
                "message": "targeted tests were not reported",
            }
        )
    else:
        for index, test in enumerate(targeted, start=1):
            if not _is_passed(test):
                failure = _failure_payload(
                    test,
                    classification="targeted_test_failed",
                    related=True,
                )
                failure["name"] = _name(test, f"targeted_test_{index}")
                blocking.append(failure)

    if not required:
        blocking.append(
            {
                "classification": "required_checks_missing",
                "related": True,
                "name": "required_checks",
                "message": "required typecheck/lint/diff checks were not reported",
            }
        )
    else:
        blocking.extend(_failed_checks(required))
    _add_named_failures(blocking, safety_failures, classification="safety_gate_failure")
    _add_named_failures(blocking, contract_failures, classification="contract_failure")
    _add_named_failures(blocking, migration_failures, classification="migration_failure")
    _add_named_failures(blocking, unknown_failures, classification="unknown_failure")

    for failure in _as_items(full_suite_failures):
        file_path = _failure_file(failure)
        if not file_path:
            blocking.append(
                _failure_payload(
                    failure,
                    classification="unknown_failure",
                    related=True,
                )
            )
            continue
        if _path_overlaps(file_path, touched):
            blocking.append(
                _failure_payload(
                    failure,
                    classification="touched_file_failure",
                    related=True,
                )
            )
        elif _path_overlaps(file_path, scope):
            blocking.append(
                _failure_payload(
                    failure,
                    classification="task_scope_failure",
                    related=True,
                )
            )
        else:
            caveats.append(
                _failure_payload(
                    failure,
                    classification="unrelated_full_suite_failure",
                    related=False,
                )
            )

    if blocking:
        decision = "block"
    elif caveats:
        decision = "complete_with_caveat"
    else:
        decision = "complete"

    return {
        "decision": decision,
        "accepted": decision != "block",
        "touched_files": touched,
        "task_scope_files": scope,
        "targeted_tests": targeted,
        "required_checks": required,
        "caveats": caveats,
        "blocking_failures": blocking,
    }


def _failure_label(failure: dict[str, Any]) -> str:
    file_part = failure.get("file") or "unknown path"
    name = failure.get("name") or failure.get("message") or "failure"
    return f"{file_part} ({name})"


def format_scoped_acceptance_report(result: dict[str, Any]) -> str:
    decision = str(result.get("decision") or "unknown")
    lines = [f"Scoped acceptance decision: {decision}"]
    blocking = result.get("blocking_failures") or []
    caveats = result.get("caveats") or []
    if blocking:
        lines.append("Blocking failures:")
        for item in blocking:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('classification', 'failure')}: {_failure_label(item)}"
                )
            else:
                lines.append(f"- {item}")
    if caveats:
        lines.append("Caveats:")
        for item in caveats:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('classification', 'caveat')}: {_failure_label(item)}"
                )
            else:
                lines.append(f"- {item}")
    if not blocking and not caveats:
        lines.append("No blocking failures or caveats.")
    return "\n".join(lines)


def append_acceptance_caveats_to_summary(
    summary: Optional[str],
    result: dict[str, Any],
) -> str:
    base = (summary or "").rstrip()
    caveats = [c for c in (result.get("caveats") or []) if isinstance(c, dict)]
    if result.get("decision") != "complete_with_caveat" or not caveats:
        return base
    labels = "; ".join(_failure_label(c) for c in caveats[:5])
    if len(caveats) > 5:
        labels += f"; +{len(caveats) - 5} more"
    note = (
        "Scoped acceptance caveat: full-suite still has unrelated failures: "
        f"{labels}."
    )
    return f"{base}\n\n{note}" if base else note
