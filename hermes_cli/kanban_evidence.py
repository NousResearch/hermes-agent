"""Evidence and handoff checks for Hermes Kanban tasks.

This module is intentionally read-only: it parses task specs and completion
metadata, then reports gaps for CLI commands and downstream handoff prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Mapping, Optional


CONTRACT_SECTIONS = (
    "goal",
    "approach",
    "acceptance criteria",
    "evidence required",
    "out of scope",
)

EVIDENCE_KEYS = (
    "changed_files",
    "commands_run",
    "tests",
    "acceptance",
    "artifacts",
    "decisions",
    "open_questions",
    "critic_review",
    "temp_files",
    "cleanup",
    "repair_loop",
    "hypothesis_tests",
)

TEMP_FILE_REQUIRED_FIELDS = (
    "path",
    "title",
    "session_id",
    "task_id",
    "run_id",
    "purpose",
    "created_during",
    "disposition",
    "cleanup_status",
)

_SECTION_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*"
    r"(goal|approach|acceptance criteria|evidence required|out of scope)"
    r"\s*(?:\*\*)?\s*(?::|-)?\s*(.*)$",
    re.IGNORECASE,
)


@dataclass
class TaskContract:
    sections: dict[str, str] = field(default_factory=dict)

    @property
    def missing_sections(self) -> list[str]:
        return [name for name in CONTRACT_SECTIONS if not self.sections.get(name)]

    def item_count(self, section: str) -> int:
        return len(split_section_items(self.sections.get(section, "")))


@dataclass
class EvidenceReport:
    task_id: str
    strict: bool
    verdict: str
    summary_present: bool
    metadata_keys: list[str]
    missing: list[str]
    warnings: list[str]
    contract: TaskContract
    acceptance_count: int
    evidence_count: int
    actual_verification_count: int
    temp_file_count: int
    hypothesis_test_count: int
    critic_issue_count: int
    repair_loop: dict[str, Any]
    run_id: Optional[int] = None
    task_status: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "strict": self.strict,
            "verdict": self.verdict,
            "missing": list(self.missing),
            "warnings": list(self.warnings),
            "summary_present": self.summary_present,
            "metadata_keys": list(self.metadata_keys),
            "contract_sections": {
                name: bool(self.contract.sections.get(name))
                for name in CONTRACT_SECTIONS
            },
            "missing_contract_sections": self.contract.missing_sections,
            "acceptance_count": self.acceptance_count,
            "evidence_count": self.evidence_count,
            "actual_verification_count": self.actual_verification_count,
            "temp_file_count": self.temp_file_count,
            "hypothesis_test_count": self.hypothesis_test_count,
            "virtual_experiment_count": self.hypothesis_test_count,
            "critic_issue_count": self.critic_issue_count,
            "repair_loop": dict(self.repair_loop),
            "run_id": self.run_id,
            "task_status": self.task_status,
        }


def parse_task_contract(body: Optional[str]) -> TaskContract:
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None
    for raw_line in (body or "").splitlines():
        line = raw_line.rstrip()
        match = _SECTION_RE.match(line)
        if match:
            current = match.group(1).lower()
            tail = match.group(2).strip()
            sections.setdefault(current, [])
            if tail:
                sections[current].append(tail)
            continue
        if current is not None:
            sections.setdefault(current, []).append(line)
    return TaskContract(
        sections={key: "\n".join(value).strip() for key, value in sections.items()}
    )


def split_section_items(text: str) -> list[str]:
    items: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*+]\s+(?:\[[ xX]\]\s*)?", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        if line:
            items.append(line)
    return items


def evaluate_evidence(
    *,
    task_id: str,
    task_body: Optional[str],
    task_status: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    strict: bool = False,
    run_id: Optional[int] = None,
    prior_attempts: int = 0,
    max_attempts: Optional[int] = None,
) -> EvidenceReport:
    contract = parse_task_contract(task_body)
    meta = _metadata_dict(metadata)
    missing: list[str] = []
    warnings: list[str] = []

    summary_present = bool((summary or "").strip())
    if not summary_present:
        missing.append("summary is missing")

    for section in ("goal", "approach", "acceptance criteria", "evidence required"):
        if not contract.sections.get(section):
            missing.append(f"task body missing {section!r} section")
    if "out of scope" not in contract.sections:
        warnings.append("task body missing optional 'out of scope' section")

    metadata_keys = sorted(str(k) for k in meta.keys())
    if not meta:
        missing.append("metadata evidence object is missing")

    acceptance = meta.get("acceptance")
    if not _has_items(acceptance):
        missing.append("metadata.acceptance evidence is missing")

    commands = meta.get("commands_run")
    tests = _value_with_alias(meta, "tests", "tests_run")
    tests_have_signal = _tests_have_real_signal(tests)
    commands_have_signal = _has_items(commands)
    actual_verification_count = (
        (_item_count(commands) if commands_have_signal else 0)
        + (_item_count(tests) if tests_have_signal else 0)
    )

    hypothesis_tests = _value_with_alias(
        meta, "hypothesis_tests", "virtual_experiments"
    )
    hypothesis_count = _item_count(hypothesis_tests)
    if "virtual_experiments" in meta and "hypothesis_tests" not in meta:
        warnings.append(
            "metadata.virtual_experiments is accepted as a legacy alias; "
            "prefer metadata.hypothesis_tests"
        )

    critic_issues = _critic_review_issues(meta.get("critic_review"))
    if "critic_review" not in meta:
        warnings.append(
            "metadata.critic_review missing; use [] or {'verdict': 'pass'} "
            "when no critic gaps remain"
        )
    if critic_issues:
        missing.extend(f"metadata.critic_review unresolved: {item}" for item in critic_issues)

    if actual_verification_count == 0:
        if hypothesis_count or _has_items(meta.get("critic_review")):
            missing.append(
                "hypothesis_tests or critic_review are present but no real "
                "tests or commands_run were recorded"
            )
        else:
            missing.append("real verification evidence is missing")

    if "changed_files" not in meta:
        warnings.append("metadata.changed_files missing; use [] when no files changed")
    if "artifacts" not in meta:
        warnings.append("metadata.artifacts missing; use [] when no deliverables were produced")
    if _has_items(meta.get("open_questions")):
        missing.append("metadata.open_questions has unresolved items")

    _check_temp_files(meta, task_status, missing, warnings)

    repair = _repair_loop_report(
        meta.get("repair_loop"),
        prior_attempts=prior_attempts,
        max_attempts=max_attempts,
        incomplete=bool(missing),
    )
    if missing and not repair.get("present"):
        warnings.append("repair_loop metadata missing for incomplete work")
    if missing and repair.get("attempts_exhausted"):
        missing.append("repair loop attempts are exhausted")

    if missing:
        verdict = "fail" if strict else "warning"
    elif warnings:
        verdict = "warning"
    else:
        verdict = "pass"

    evidence_count = sum(1 for key in EVIDENCE_KEYS if _has_items(meta.get(key)))
    if "hypothesis_tests" not in meta and _has_items(meta.get("virtual_experiments")):
        evidence_count += 1
    if "tests" not in meta and _has_items(meta.get("tests_run")):
        evidence_count += 1

    return EvidenceReport(
        task_id=task_id,
        strict=strict,
        verdict=verdict,
        summary_present=summary_present,
        metadata_keys=metadata_keys,
        missing=missing,
        warnings=warnings,
        contract=contract,
        acceptance_count=contract.item_count("acceptance criteria"),
        evidence_count=evidence_count,
        actual_verification_count=actual_verification_count,
        temp_file_count=_item_count(meta.get("temp_files")),
        hypothesis_test_count=hypothesis_count,
        critic_issue_count=len(critic_issues),
        repair_loop=repair,
        run_id=run_id,
        task_status=task_status,
    )


def format_report(report: EvidenceReport) -> str:
    lines = [
        f"Kanban evidence check: {report.task_id}",
        f"Verdict: {report.verdict}",
        f"Summary: {'present' if report.summary_present else 'missing'}",
        "Contract:",
    ]
    for section in CONTRACT_SECTIONS:
        lines.append(
            f"- {section}: {'present' if report.contract.sections.get(section) else 'missing'}"
        )
    lines.extend(
        [
            "Evidence:",
            f"- metadata keys: {', '.join(report.metadata_keys) if report.metadata_keys else '(none)'}",
            f"- acceptance criteria: {report.acceptance_count}",
            f"- evidence keys populated: {report.evidence_count}",
            f"- real verification records: {report.actual_verification_count}",
            f"- temp files: {report.temp_file_count}",
            f"- hypothesis tests: {report.hypothesis_test_count}",
            f"- critic issues: {report.critic_issue_count}",
            "Repair loop:",
            (
                f"- attempt {report.repair_loop.get('attempt')}/"
                f"{report.repair_loop.get('max_attempts') or 'unbounded'}, "
                f"can_retry={report.repair_loop.get('can_retry')}"
            ),
        ]
    )
    if report.missing:
        lines.append("Missing:")
        lines.extend(f"- {item}" for item in report.missing)
    if report.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in report.warnings)
    return "\n".join(lines)


def build_prompt_next(
    *,
    task_id: str,
    title: str,
    body: Optional[str],
    status: Optional[str],
    latest_summary: Optional[str],
    latest_metadata: Optional[Mapping[str, Any]],
    report: EvidenceReport,
    worker_context: Optional[str] = None,
) -> str:
    meta = _metadata_dict(latest_metadata)
    lines = [
        f"Continue Hermes Kanban task {task_id}: {title}",
        "",
        "Use the task spec, latest handoff, evidence gaps, critic notes, "
        "and repair notes below. Do not repeat a failed strategy without "
        "a new reason.",
        "",
        "## Task",
        f"- Status: {status or 'unknown'}",
        "",
        _cap(body or "(no task body)", 4000),
        "",
        "## Latest run summary",
        _cap(latest_summary or "(no summary recorded)", 2000),
        "",
        "## Evidence status",
        format_report(report),
        "",
        "## Latest metadata",
        _cap(json.dumps(meta, ensure_ascii=False, sort_keys=True), 3000)
        if meta else "(no metadata recorded)",
    ]
    critic = meta.get("critic_review")
    if _has_items(critic):
        lines.extend(
            [
                "",
                "## Critic review",
                _cap(json.dumps(critic, ensure_ascii=False, sort_keys=True), 2000),
            ]
        )
    temp_lines = _temp_file_prompt_lines(meta)
    if temp_lines:
        lines.extend(["", "## Temp files to inspect", *temp_lines])
    repair = meta.get("repair_loop") if isinstance(meta.get("repair_loop"), Mapping) else {}
    if repair:
        lines.extend(
            [
                "",
                "## Repair loop",
                f"- Failure reason: {repair.get('failure_reason') or '(not recorded)'}",
                f"- Acceptance gap: {_cap(json.dumps(repair.get('acceptance_gap', []), ensure_ascii=False), 1000)}",
                f"- Failed strategies: {_cap(json.dumps(repair.get('failed_strategies', []), ensure_ascii=False), 1000)}",
                f"- Next strategy: {repair.get('next_strategy') or '(not recorded)'}",
                f"- Stop reason: {repair.get('stop_reason') or '(none)'}",
            ]
        )
    hypotheses = _value_with_alias(meta, "hypothesis_tests", "virtual_experiments")
    if _has_items(hypotheses):
        lines.extend(
            [
                "",
                "## Hypothesis tests",
                _cap(json.dumps(hypotheses, ensure_ascii=False, sort_keys=True), 2000),
                "Treat these as hypothesis checks only; still run or cite real verification before completing.",
            ]
        )
    if worker_context:
        lines.extend(
            [
                "",
                "## Existing worker context",
                _cap(worker_context, 4000),
            ]
        )
    lines.extend(
        [
            "",
            "## Next action",
            _next_action(report),
        ]
    )
    return "\n".join(lines).strip()


def _metadata_dict(metadata: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(metadata or {})


def _value_with_alias(meta: Mapping[str, Any], key: str, alias: str) -> Any:
    value = meta.get(key)
    if _has_items(value):
        return value
    return meta.get(alias)


def _has_items(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return any(_has_items(v) for v in value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    return True


def _item_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return 1 if value.strip() else 0
    if isinstance(value, Mapping):
        return len(value) if value else 0
    if isinstance(value, (list, tuple, set)):
        return sum(1 for item in value if _has_items(item))
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return int(value) if value > 0 else 0
    return 1


def _tests_have_real_signal(value: Any) -> bool:
    if not _has_items(value):
        return False
    entries = value if isinstance(value, (list, tuple)) else [value]
    for entry in entries:
        if isinstance(entry, str):
            if entry.strip().lower() not in {"skipped", "not run", "none"}:
                return True
            continue
        if isinstance(entry, Mapping):
            status = str(
                entry.get("status")
                or entry.get("outcome")
                or entry.get("result")
                or ""
            ).strip().lower()
            if status in {"skipped", "skip", "not run", "not_run", "none"}:
                continue
            if _has_items(entry):
                return True
            continue
        if _has_items(entry):
            return True
    return False


def _critic_review_issues(value: Any) -> list[str]:
    if not _has_items(value):
        return []
    issues: list[str] = []
    if isinstance(value, Mapping):
        _extend_review_issues(issues, value)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping):
                _extend_review_issues(issues, item)
    return issues


def _extend_review_issues(issues: list[str], review: Mapping[str, Any]) -> None:
    status = str(
        review.get("status")
        or review.get("verdict")
        or review.get("outcome")
        or ""
    ).strip().lower()
    if status in {"fail", "failed", "block", "blocked", "open", "needs_work"}:
        issues.append(f"review verdict is {status}")
    for key in ("open_issues", "blockers", "gaps", "objections", "unresolved"):
        value = review.get(key)
        if _has_items(value):
            issues.append(f"{key}={_brief(value)}")


def _check_temp_files(
    meta: Mapping[str, Any],
    task_status: Optional[str],
    missing: list[str],
    warnings: list[str],
) -> None:
    temp_files = meta.get("temp_files")
    if temp_files is None:
        return
    if not isinstance(temp_files, list):
        missing.append("metadata.temp_files must be a list of ledger objects")
        return
    for idx, entry in enumerate(temp_files):
        label = f"metadata.temp_files[{idx}]"
        if isinstance(entry, str):
            missing.append(f"{label} must be an object, not a bare path string")
            continue
        if not isinstance(entry, Mapping):
            missing.append(f"{label} must be an object")
            continue
        for field_name in TEMP_FILE_REQUIRED_FIELDS:
            if not _has_items(entry.get(field_name)):
                missing.append(f"{label}.{field_name} is missing")
        if entry.get("cleanup_status") == "kept" and not _has_items(entry.get("keep_reason")):
            missing.append(f"{label}.keep_reason is required when cleanup_status is kept")
        if (
            task_status == "done"
            and entry.get("disposition") == "delete_on_verified_success"
            and entry.get("cleanup_status") in {"pending", "kept"}
        ):
            warnings.append(
                f"{label} is marked delete_on_verified_success but cleanup_status is {entry.get('cleanup_status')}"
            )
    if temp_files and not isinstance(meta.get("cleanup", {}), Mapping):
        missing.append("metadata.cleanup must be an object when temp_files are recorded")


def _repair_loop_report(
    value: Any,
    *,
    prior_attempts: int,
    max_attempts: Optional[int],
    incomplete: bool,
) -> dict[str, Any]:
    repair = value if isinstance(value, Mapping) else {}
    present = isinstance(value, Mapping) and bool(value)
    attempt = _coerce_int(repair.get("attempt"), prior_attempts)
    configured_max = _coerce_int(repair.get("max_attempts"), max_attempts)
    attempts_exhausted = False
    can_retry = False
    if incomplete and configured_max is not None:
        attempts_exhausted = attempt >= configured_max
        can_retry = attempt < configured_max
    elif incomplete:
        can_retry = True
    return {
        "present": present,
        "attempt": attempt,
        "max_attempts": configured_max,
        "can_retry": can_retry,
        "attempts_exhausted": attempts_exhausted,
        "failure_reason": repair.get("failure_reason") if isinstance(repair, Mapping) else None,
        "next_strategy": repair.get("next_strategy") if isinstance(repair, Mapping) else None,
        "stop_reason": repair.get("stop_reason") if isinstance(repair, Mapping) else None,
    }


def _coerce_int(value: Any, default: Optional[int]) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _temp_file_prompt_lines(meta: Mapping[str, Any]) -> list[str]:
    temp_files = meta.get("temp_files")
    if not isinstance(temp_files, list):
        return []
    lines: list[str] = []
    for entry in temp_files:
        if isinstance(entry, Mapping):
            title = entry.get("title") or "(untitled temp file)"
            path = entry.get("path") or "(missing path)"
            purpose = entry.get("purpose") or "(missing purpose)"
            status = entry.get("cleanup_status") or "unknown"
            lines.append(f"- {title}: {path} [{status}] - {purpose}")
        elif isinstance(entry, str):
            lines.append(f"- {entry} [bare path; ledger incomplete]")
    return lines


def _next_action(report: EvidenceReport) -> str:
    if report.verdict == "pass":
        return "Evidence is complete. If the task is not already done, complete it with the recorded summary and metadata."
    if report.repair_loop.get("can_retry"):
        return "Repair the missing evidence or failed acceptance gap, preserve relevant temp files, rerun real verification, then update metadata."
    return "Do not keep looping. Block or escalate the task with the missing evidence and open questions."


def _brief(value: Any) -> str:
    return _cap(json.dumps(value, ensure_ascii=False, sort_keys=True), 240)


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 24)].rstrip() + "\n...[truncated]"
