"""First-class closeout state and bounded resume helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


NO_LIVE_PROVIDER_BOUNDARY = (
    "Do not perform live provider, deploy, DNS, payment, email, webhook, merge, "
    "or other external mutations unless the operator explicitly approved them "
    "in this resumed run."
)
_GREEN_CI = {"success", "passed", "green", "ok"}
_BAD_OR_UNKNOWN_CI = {"", "unknown", "not_checked", "pending", "queued", "failed", "cancelled"}
_CONTRACT_STATUS_VALUES = {"done", "partial", "blocked", "not_started", "not_applicable"}
_INCOMPLETE_CONTRACT_STATUS_VALUES = {"partial", "blocked", "not_started"}
_NO_EVIDENCE_VALUES = {"", "none", "n/a", "na", "unknown", "not checked", "not run", "missing"}
_REQUIREMENT_HEADING_RE = re.compile(
    r"\b(requirements?|wymagania|acceptance\s+criteria|requested\s+items?)\b",
    re.IGNORECASE,
)
_ENUMERATED_REQUIREMENT_RE = re.compile(
    r"^\s*(?:[-*]|\d+[.)]|\[[ xX-]\])\s+(?P<text>.+?)\s*$"
)
_CHECKLIST_FIELD_ALIASES = {
    "requirement": "requirement",
    "req": "requirement",
    "item": "requirement",
    "status": "status",
    "evidence": "evidence",
    "source": "evidence",
    "residual risk": "residual_risk",
    "residual_risk": "residual_risk",
    "risk": "residual_risk",
    "next action": "next_action",
    "next_action": "next_action",
    "next": "next_action",
}


def _clean_text(value: Any, *, max_chars: int = 2_000) -> str:
    from hermes_cli.closure_artifacts import _clean_text as _closure_clean_text

    return _closure_clean_text(value, max_chars=max_chars)


def _clean_list(values: Any, *, max_items: int = 30, max_chars: int = 500) -> list[str]:
    from hermes_cli.closure_artifacts import _clean_list as _closure_clean_list

    return _closure_clean_list(values, max_items=max_items, max_chars=max_chars)


def _clean_requirement_text(value: Any) -> str:
    text = _clean_text(value, max_chars=300)
    text = re.sub(r"^\[[ xX-]\]\s*", "", text).strip()
    text = re.sub(r"^(?:requirement|req|item)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def _split_inline_requirements(text: str) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    if not re.search(r"(?:^|\s)\d+[.)]\s+", value):
        cleaned = _clean_requirement_text(value)
        return [cleaned] if cleaned else []
    parts = re.split(r"(?:^|\s)\d+[.)]\s+", value)
    return [_clean_requirement_text(part) for part in parts if _clean_requirement_text(part)]


def extract_contract_requirements(task_contract: Any) -> list[str]:
    """Return explicit multi-item user requirements from a task contract.

    Deliberately conservative: a one-sentence task or a single bullet remains a
    simple closeout and does not trigger contract-checklist enforcement.
    """

    requirements: list[str] = []
    in_requirement_section = False
    for raw_line in str(task_contract or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _REQUIREMENT_HEADING_RE.search(line):
            in_requirement_section = True
            after_colon = line.split(":", 1)[1].strip() if ":" in line else ""
            requirements.extend(_split_inline_requirements(after_colon))
            continue
        match = _ENUMERATED_REQUIREMENT_RE.match(line)
        if not match:
            continue
        if in_requirement_section or re.match(r"^\s*\d+[.)]", line):
            requirement = _clean_requirement_text(match.group("text"))
            if requirement:
                requirements.append(requirement)

    deduped = list(dict.fromkeys(requirements))
    return deduped if len(deduped) >= 2 else []


def _normalize_checklist_key(key: Any) -> str:
    text = _clean_text(key, max_chars=80).lower().replace("-", "_")
    text = re.sub(r"\s+", " ", text.replace("_", " ")).strip()
    return _CHECKLIST_FIELD_ALIASES.get(text, _CHECKLIST_FIELD_ALIASES.get(text.replace(" ", "_"), ""))


def _normalize_contract_status(value: Any) -> str:
    status = _clean_text(value, max_chars=80).lower()
    status = re.sub(r"[\s-]+", "_", status)
    return re.sub(r"[^a-z_]", "", status)


def _normalize_contract_checklist_item(item: Mapping[str, Any]) -> dict[str, str] | None:
    normalized: dict[str, str] = {
        "requirement": "",
        "status": "",
        "evidence": "",
        "residual_risk": "",
        "next_action": "",
    }
    for raw_key, raw_value in item.items():
        key = _normalize_checklist_key(raw_key)
        if not key:
            continue
        max_chars = 80 if key == "status" else 500
        normalized[key] = _clean_text(raw_value, max_chars=max_chars)
    normalized["status"] = _normalize_contract_status(normalized.get("status"))
    if not any(normalized.values()):
        return None
    return normalized


def _parse_contract_checklist_line(line: str) -> dict[str, str] | None:
    if not re.search(r"\bstatus\s*:", line, flags=re.IGNORECASE):
        return None
    if not re.search(r"\bevidence\s*:", line, flags=re.IGNORECASE):
        return None
    fields: dict[str, str] = {}
    for raw_part in re.split(r"\s*[;|]\s*", line):
        part = re.sub(r"^\s*(?:[-*]|\d+[.)]|\[[ xX-]\])\s*", "", raw_part.strip())
        if ":" not in part:
            continue
        raw_key, raw_value = part.split(":", 1)
        key = _normalize_checklist_key(raw_key)
        if key:
            fields[key] = raw_value.strip()
    if not fields:
        return None
    return _normalize_contract_checklist_item(fields)


def parse_contract_checklist(final_response: Any) -> list[dict[str, str]]:
    """Parse compact markdown/prose checklist rows from a final response."""

    items: list[dict[str, str]] = []
    for line in str(final_response or "").splitlines():
        item = _parse_contract_checklist_line(line)
        if item:
            items.append(item)
        if len(items) >= 30:
            break
    return items


def normalize_contract_checklist(checklist: Any) -> list[dict[str, str]]:
    """Normalize a structured or text contract checklist for closeout storage."""

    if checklist is None:
        return []
    if isinstance(checklist, str):
        return parse_contract_checklist(checklist)
    if not isinstance(checklist, list):
        checklist = [checklist]
    normalized: list[dict[str, str]] = []
    for item in checklist[:30]:
        if isinstance(item, Mapping):
            clean_item = _normalize_contract_checklist_item(item)
        else:
            clean_item = _parse_contract_checklist_line(str(item))
        if clean_item:
            normalized.append(clean_item)
    return normalized


def _has_meaningful_evidence(value: Any) -> bool:
    text = _clean_text(value, max_chars=300).lower()
    return text not in _NO_EVIDENCE_VALUES


def classify_closeout_response(
    final_response: Any,
    *,
    task_contract: str | None = None,
    contract_checklist: list[dict[str, Any]] | str | None = None,
    incomplete_contract_accepted: bool = False,
    pr_url: str | None = None,
    merge_status: str | None = None,
    ci_status: str | None = None,
    invalid_review_children: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    text = str(final_response or "").lower()
    merge = str(merge_status or "").strip().lower()
    ci = str(ci_status or "").strip().lower()
    reasons: list[str] = []

    if invalid_review_children:
        reasons.append("invalid_review_child")
    if "review not completed" in text or "review not complete" in text:
        reasons.append("review_not_completed")
    if "not merged" in text or merge in {"not_merged", "unmerged"}:
        reasons.append("pr_not_merged")
    if "ci not checked" in text or "ci was not checked" in text:
        reasons.append("ci_not_checked")
    if pr_url and merge in {"", "unknown"}:
        reasons.append("pr_merge_unknown")
    if merge == "merged":
        if ci not in _GREEN_CI:
            reasons.append("post_main_ci_not_green")
    elif ci in {"not_checked", "unknown", ""} and ("ci" in text or ci == "not_checked"):
        if "ci_not_checked" not in reasons:
            reasons.append("ci_not_checked")
    elif ci in _BAD_OR_UNKNOWN_CI - {"", "unknown", "not_checked"}:
        reasons.append("ci_not_green")

    contract_requirements = extract_contract_requirements(task_contract)
    clean_contract_checklist = normalize_contract_checklist(contract_checklist)
    if not clean_contract_checklist:
        clean_contract_checklist = parse_contract_checklist(final_response)
    if contract_requirements:
        if not clean_contract_checklist:
            reasons.append("contract_checklist_missing")
        else:
            if len(clean_contract_checklist) < len(contract_requirements):
                reasons.append("contract_checklist_missing_requirements")
            has_invalid_status = False
            has_missing_evidence = False
            has_incomplete_status = False
            has_missing_next_action = False
            for item in clean_contract_checklist:
                status = item.get("status") or ""
                if status not in _CONTRACT_STATUS_VALUES:
                    has_invalid_status = True
                    continue
                if not _has_meaningful_evidence(item.get("evidence")):
                    has_missing_evidence = True
                if status in _INCOMPLETE_CONTRACT_STATUS_VALUES:
                    if not incomplete_contract_accepted:
                        has_incomplete_status = True
                    if not item.get("next_action") and not incomplete_contract_accepted:
                        has_missing_next_action = True
            if has_invalid_status:
                reasons.append("contract_checklist_status_invalid")
            if has_missing_evidence:
                reasons.append("contract_checklist_evidence_missing")
            if has_incomplete_status:
                reasons.append("contract_checklist_incomplete")
            if has_missing_next_action:
                reasons.append("contract_checklist_next_action_missing")

    deduped = list(dict.fromkeys(reasons))
    return {
        "status": "recoverable_incomplete" if deduped else "complete_candidate",
        "reasons": deduped,
        "contract_requirements": contract_requirements,
        "contract_checklist": clean_contract_checklist,
    }


def _clean_invalid_review_children(children: Any) -> list[dict[str, Any]]:
    if not isinstance(children, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for child in children[:20]:
        if not isinstance(child, Mapping):
            continue
        entry: dict[str, Any] = {}
        for key in (
            "task_index",
            "child_session_id",
            "review_evidence_status",
            "goal_preview",
        ):
            if key in child:
                entry[key] = _clean_text(child.get(key), max_chars=240)
        recommendation = child.get("rerun_recommendation")
        if isinstance(recommendation, Mapping):
            entry["rerun_recommendation"] = {
                str(k): _clean_text(v, max_chars=300)
                for k, v in recommendation.items()
                if k in {"goal_preview", "reason", "suggested_toolsets", "suggested_max_iterations"}
            }
        if entry:
            cleaned.append(entry)
    return cleaned


def build_safe_bounded_resume_prompt(packet: Mapping[str, Any]) -> str:
    latest = _clean_text(
        packet.get("latest_session_id") or packet.get("session_id") or "latest child",
        max_chars=120,
    )
    task = _clean_text(packet.get("task_id") or "unknown", max_chars=120)
    remaining = _clean_list(
        packet.get("remaining_closeout_tasks") or packet.get("remaining_checklist"),
        max_items=8,
        max_chars=240,
    )
    remaining_text = "; ".join(remaining) if remaining else "inspect current repo, PR, review, and CI state"
    return (
        f"Resume the closeout for task {task} in session {latest}. "
        "Use only the compact closeout/closure packet and current repo state; "
        "do not load or replay the bloated parent history. "
        f"Remaining closeout tasks: {remaining_text}. "
        "If a PR was merged, independently verify the post-merge main CI/deploy gate "
        "before declaring the run complete. "
        f"{NO_LIVE_PROVIDER_BOUNDARY}"
    )


def write_closeout_state(
    *,
    session_id: str,
    latest_session_id: str | None = None,
    parent_lineage: list[str] | None = None,
    task_id: str | None = None,
    task_contract: str | None = None,
    contract_checklist: list[dict[str, Any]] | str | None = None,
    incomplete_contract_accepted: bool = False,
    final_response: Any = "",
    pr_url: str | None = None,
    head_sha: str | None = None,
    merge_status: str | None = None,
    ci_status: str | None = None,
    invalid_review_children: list[dict[str, Any]] | None = None,
    remaining_closeout_tasks: list[str] | None = None,
    remaining_gates: list[str] | None = None,
    changed_files: list[str] | None = None,
    commits: list[str] | None = None,
    verified_artifacts: list[str] | None = None,
    tests_run: list[str] | None = None,
    failing_tests: list[str] | None = None,
    blockers: list[str] | None = None,
    blocked_review_children: list[str] | None = None,
    next_safe_action: str | None = None,
    required_model: str = "gpt-5.5",
    live_provider_actions_approved: bool = False,
) -> Path:
    from hermes_cli.artifact_contracts import write_required_artifacts
    from hermes_cli.closure_artifacts import read_closure_artifact, write_closure_artifact

    invalid_children = _clean_invalid_review_children(invalid_review_children or [])
    verdict = classify_closeout_response(
        final_response,
        task_contract=task_contract,
        contract_checklist=contract_checklist,
        incomplete_contract_accepted=incomplete_contract_accepted,
        pr_url=pr_url,
        merge_status=merge_status,
        ci_status=ci_status,
        invalid_review_children=invalid_children,
    )
    status = verdict["status"]
    provisional = {
        "session_id": session_id,
        "latest_session_id": latest_session_id or session_id,
        "task_id": task_id or "",
        "remaining_closeout_tasks": _clean_list(remaining_closeout_tasks),
    }
    safe_prompt = build_safe_bounded_resume_prompt(provisional)
    if live_provider_actions_approved:
        safe_prompt = safe_prompt.replace(NO_LIVE_PROVIDER_BOUNDARY, "").strip()

    clean_blockers = _clean_list(blockers if blockers is not None else failing_tests or verdict["reasons"])
    clean_remaining = _clean_list(remaining_closeout_tasks)
    clean_remaining_gates = _clean_list(
        remaining_gates if remaining_gates is not None else remaining_closeout_tasks
    )
    clean_commits = _clean_list(commits if commits is not None else ([head_sha] if head_sha else []))
    clean_blocked_review_children = _clean_list(
        blocked_review_children,
        max_items=20,
        max_chars=300,
    )
    clean_next_action = _clean_text(
        next_safe_action or (clean_remaining[0] if clean_remaining else "return a compact final or blocked answer"),
        max_chars=500,
    )
    required_artifacts = write_required_artifacts(task_contract or "", final_response)
    required_written_paths = [
        item["path"]
        for item in required_artifacts
        if item.get("status") == "written" and item.get("path")
    ]
    clean_verified_artifacts = _clean_list(
        verified_artifacts if verified_artifacts is not None else changed_files
    )
    for artifact_path in required_written_paths:
        if artifact_path not in clean_verified_artifacts:
            clean_verified_artifacts.append(artifact_path)

    path = write_closure_artifact(
        session_id=session_id,
        task_id=task_id or "",
        status=status,
        last_completed_step=final_response,
        changed_files=changed_files or [],
        tests_run=tests_run or [],
        test_results={},
        failing_tests=failing_tests or [],
        remaining_checklist=remaining_closeout_tasks or [],
        exact_resume_prompt=safe_prompt,
        active_session_lease_released=False,
    )
    data = read_closure_artifact(path)
    data.update(
        {
            "latest_session_id": _clean_text(latest_session_id or session_id, max_chars=200),
            "parent_lineage": _clean_list(parent_lineage or [], max_items=20, max_chars=200),
            "pr_url": _clean_text(pr_url or "", max_chars=500),
            "head_sha": _clean_text(head_sha or "", max_chars=120),
            "merge_status": _clean_text(merge_status or "", max_chars=80),
            "ci_status": _clean_text(ci_status or "", max_chars=80),
            "invalid_review_children": invalid_children,
            "remaining_closeout_tasks": clean_remaining,
            "safe_bounded_resume_prompt": safe_prompt,
            "closeout_reasons": verdict["reasons"],
            "task_contract": _clean_text(task_contract or task_id or "", max_chars=900),
            "contract_requirements": verdict.get("contract_requirements") or [],
            "contract_checklist": verdict.get("contract_checklist") or [],
            "incomplete_contract_accepted": bool(incomplete_contract_accepted),
            "commits": clean_commits,
            "remaining_gates": clean_remaining_gates,
            "blocked_review_children": clean_blocked_review_children,
            "provider_actions_approval_status": (
                "approved" if live_provider_actions_approved else "not_approved"
            ),
            "required_artifacts": required_artifacts,
            "verified_artifacts": clean_verified_artifacts,
            "blockers": clean_blockers,
            "next_safe_action": clean_next_action,
            "required_model": _clean_text(required_model or "gpt-5.5", max_chars=80),
            "live_provider_actions_approved": bool(live_provider_actions_approved),
        }
    )
    from hermes_cli.finalization_mode import build_compact_finalization_prompt

    data["compact_finalization_prompt"] = build_compact_finalization_prompt(data)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def latest_closeout_state(
    *,
    session_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any] | None:
    from hermes_cli.closure_artifacts import latest_closure_artifact

    return latest_closure_artifact(session_id=session_id, task_id=task_id)


def build_closeout_resume_command(
    packet: Mapping[str, Any],
    *,
    hermes_command: str = "hermes",
    max_turns: int = 8,
) -> str:
    target = _clean_text(
        packet.get("latest_session_id") or packet.get("session_id") or "",
        max_chars=200,
    )
    prompt = _clean_text(
        packet.get("safe_bounded_resume_prompt") or build_safe_bounded_resume_prompt(packet),
        max_chars=4_000,
    )
    prompt_arg = "'" + prompt.replace("'", "''") + "'"
    command = _clean_text(hermes_command or "hermes", max_chars=200)
    return f"{command} chat --resume {target} --max-turns {int(max_turns)} --query {prompt_arg}"


def build_commit_ci_check_command(head_sha: str, *, limit: int = 20) -> list[str]:
    sha = str(head_sha or "").strip()
    if not re.fullmatch(r"[A-Fa-f0-9]{6,64}", sha):
        raise ValueError("head_sha must be a commit-like hex string")
    return ["gh", "run", "list", "--commit", sha, "--limit", str(int(limit))]
