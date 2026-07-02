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


def _clean_text(value: Any, *, max_chars: int = 2_000) -> str:
    from hermes_cli.closure_artifacts import _clean_text as _closure_clean_text

    return _closure_clean_text(value, max_chars=max_chars)


def _clean_list(values: Any, *, max_items: int = 30, max_chars: int = 500) -> list[str]:
    from hermes_cli.closure_artifacts import _clean_list as _closure_clean_list

    return _closure_clean_list(values, max_items=max_items, max_chars=max_chars)


def classify_closeout_response(
    final_response: Any,
    *,
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

    deduped = list(dict.fromkeys(reasons))
    return {
        "status": "recoverable_incomplete" if deduped else "complete_candidate",
        "reasons": deduped,
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
    final_response: Any = "",
    pr_url: str | None = None,
    head_sha: str | None = None,
    merge_status: str | None = None,
    ci_status: str | None = None,
    invalid_review_children: list[dict[str, Any]] | None = None,
    remaining_closeout_tasks: list[str] | None = None,
    changed_files: list[str] | None = None,
    verified_artifacts: list[str] | None = None,
    tests_run: list[str] | None = None,
    failing_tests: list[str] | None = None,
    blockers: list[str] | None = None,
    next_safe_action: str | None = None,
    required_model: str = "gpt-5.5",
    live_provider_actions_approved: bool = False,
) -> Path:
    from hermes_cli.closure_artifacts import read_closure_artifact, write_closure_artifact

    invalid_children = _clean_invalid_review_children(invalid_review_children or [])
    verdict = classify_closeout_response(
        final_response,
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
    clean_next_action = _clean_text(
        next_safe_action or (clean_remaining[0] if clean_remaining else "return a compact final or blocked answer"),
        max_chars=500,
    )

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
            "verified_artifacts": _clean_list(verified_artifacts if verified_artifacts is not None else changed_files),
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
