"""Deterministic first-stage controls for token-efficient complex tasks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def enrich_usage_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return canonical prompt/processed counters without dropping legacy keys."""
    result = dict(row)
    new_input = int(row.get("input_tokens") or row.get("new_input_tokens") or 0)
    cache_read = int(row.get("cache_read_tokens") or 0)
    cache_write = int(row.get("cache_write_tokens") or 0)
    output = int(row.get("output_tokens") or 0)
    api_calls = int(row.get("api_calls") or row.get("api_call_count") or 0)
    cache_input = cache_read + cache_write
    prompt = new_input + cache_input
    result.update(
        new_input_tokens=new_input,
        cache_input_tokens=cache_input,
        prompt_tokens=prompt,
        processed_tokens=prompt + output,
        avg_prompt_tokens_per_call=(prompt / api_calls if api_calls else None),
    )
    cost_status = row.get("cost_status")
    result["cost_unknown"] = cost_status == "unknown" or (
        cost_status in (None, "") and row.get("estimated_cost") is None
        and row.get("estimated_cost_usd") is None
    )
    return result


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _contract_hash(contract: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in contract.items() if k != "contract_hash"}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_task_contract(
    *,
    task_id: str,
    requirement_id: str,
    revision: int,
    original_request: str,
    business_goal: str,
    acceptance_criteria: Sequence[Mapping[str, Any]],
    in_scope: Sequence[str] = (),
    out_of_scope: Sequence[str] = (),
    constraints: Sequence[str] = (),
    protected_assets: Sequence[str] = (),
    authority: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    contract = {
        "task_id": task_id,
        "requirement_id": requirement_id,
        "revision": int(revision),
        "original_request": original_request,
        "business_goal": business_goal,
        "in_scope": list(in_scope),
        "out_of_scope": list(out_of_scope),
        "acceptance_criteria": [dict(item) for item in acceptance_criteria],
        "constraints": list(constraints),
        "protected_assets": list(protected_assets),
        "authority": dict(authority or {
            "may_edit_code": True,
            "may_commit": False,
            "may_push": False,
            "may_deploy": False,
            "may_delete": False,
        }),
        "success_definition": "所有验收标准通过",
        "contract_version": 1,
    }
    contract["contract_hash"] = _contract_hash(contract)
    return contract


_AUTHORITY_FIELDS = (
    "may_edit_code", "may_commit", "may_push", "may_deploy", "may_delete",
)


def validate_task_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "task_id", "requirement_id", "revision", "original_request",
        "business_goal", "acceptance_criteria", "authority", "contract_hash",
    }
    missing = sorted(required - set(contract))
    if missing:
        return {"valid": False, "reason": "missing_fields", "missing": missing}
    if not isinstance(contract.get("original_request"), str) or not contract["original_request"]:
        return {"valid": False, "reason": "original_request_missing"}
    if not isinstance(contract.get("revision"), int) or isinstance(contract.get("revision"), bool) or contract["revision"] < 1:
        return {"valid": False, "reason": "revision_invalid"}
    if not isinstance(contract.get("business_goal"), str) or not contract["business_goal"]:
        return {"valid": False, "reason": "business_goal_invalid"}
    criteria = contract.get("acceptance_criteria")
    if not isinstance(criteria, list):
        return {"valid": False, "reason": "acceptance_criteria_invalid"}
    criterion_ids: set[str] = set()
    for item in criteria:
        if not isinstance(item, Mapping):
            return {"valid": False, "reason": "acceptance_criteria_invalid"}
        criterion_id = item.get("id")
        if not isinstance(criterion_id, str) or not criterion_id or criterion_id in criterion_ids:
            return {"valid": False, "reason": "acceptance_criteria_invalid"}
        if not isinstance(item.get("requirement"), str) or not item["requirement"]:
            return {"valid": False, "reason": "acceptance_criteria_invalid"}
        if not isinstance(item.get("verification"), str) or not item["verification"]:
            return {"valid": False, "reason": "acceptance_criteria_invalid"}
        criterion_ids.add(criterion_id)
    authority = contract.get("authority")
    if not isinstance(authority, Mapping) or set(authority) != set(_AUTHORITY_FIELDS):
        return {"valid": False, "reason": "authority_invalid"}
    if any(type(authority[field]) is not bool for field in _AUTHORITY_FIELDS):
        return {"valid": False, "reason": "authority_invalid"}
    for field in ("in_scope", "out_of_scope", "constraints", "protected_assets"):
        value = contract.get(field, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            return {"valid": False, "reason": f"{field}_invalid"}
    if _contract_hash(contract) != contract.get("contract_hash"):
        return {"valid": False, "reason": "contract_hash_mismatch"}
    return {"valid": True, "reason": None}


@dataclass(frozen=True)
class BudgetPolicy:
    api_call_limit: int
    prompt_token_limit: int
    absolute_prompt_limit: int
    summarize_ratio: float = 0.70
    freeze_scope_ratio: float = 0.85


class PromptBudget:
    """Per-stage prompt budget using input + cache read + cache write."""

    def __init__(self, policy: BudgetPolicy):
        self.policy = policy
        self.api_calls = 0
        self.prompt_tokens = 0
        self.processed_tokens = 0

    def record(self, *, input_tokens: int, cache_read_tokens: int,
               cache_write_tokens: int, output_tokens: int) -> dict[str, Any]:
        prompt = int(input_tokens) + int(cache_read_tokens) + int(cache_write_tokens)
        self.api_calls += 1
        self.prompt_tokens += prompt
        self.processed_tokens += prompt + int(output_tokens)
        ratio = self.prompt_tokens / self.policy.prompt_token_limit
        status = "active"
        action = "continue"
        if self.prompt_tokens > self.policy.absolute_prompt_limit:
            status, action = "budget_exhausted", "pause"
        elif self.prompt_tokens >= self.policy.prompt_token_limit or self.api_calls >= self.policy.api_call_limit:
            status, action = "budget_exhausted", "handoff"
        elif ratio >= self.policy.freeze_scope_ratio:
            action = "freeze_scope"
        elif ratio >= self.policy.summarize_ratio:
            action = "summarize"
        return {
            "status": status,
            "action": action,
            "api_calls": self.api_calls,
            "prompt_tokens": self.prompt_tokens,
            "processed_tokens": self.processed_tokens,
            "prompt_ratio": ratio,
        }


def _git_probe(workspace: Path) -> tuple[bool, str | None, bool | None]:
    try:
        top = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if top.returncode != 0:
            return False, None, None
        branch = subprocess.run(
            ["git", "-C", str(workspace), "branch", "--show-current"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip() or None
        dirty = bool(subprocess.run(
            ["git", "-C", str(workspace), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip())
        return True, branch, dirty
    except (OSError, subprocess.SubprocessError):
        return False, None, None


def preflight_workspace(
    workspace: str | os.PathLike[str], *,
    required_files: Sequence[str] = (),
    test_commands: Sequence[str] = (),
    dependencies_complete: bool = True,
    revision: int | None = None,
    current_revision: int | None = None,
    successful_artifact: bool = False,
    active_run: bool = False,
    require_git: bool = False,
) -> dict[str, Any]:
    path = Path(workspace)
    exists = path.is_dir()
    files = [p for p in path.rglob("*") if p.is_file()] if exists else []
    git_repo, branch, dirty = _git_probe(path) if exists else (False, None, None)
    missing = [name for name in required_files if not (path / name).exists()] if exists else list(required_files)
    status, block_reason = "ready", None
    if not exists:
        status, block_reason = "blocked", "workspace_missing"
    elif not files:
        status, block_reason = "blocked", "workspace_empty"
        missing.append("workspace_empty")
    elif current_revision is not None and revision is not None and revision < current_revision:
        status, block_reason = "superseded", "revision_superseded"
    elif successful_artifact:
        status, block_reason = "duplicate", "successful_artifact_exists"
    elif active_run:
        status, block_reason = "blocked", "active_run_exists"
    elif not dependencies_complete:
        status, block_reason = "blocked", "dependencies_incomplete"
    elif require_git and not git_repo:
        status, block_reason = "blocked", "git_repository_required"
    elif missing:
        status, block_reason = "blocked", "required_files_missing"
    return {
        "status": status,
        "workspace_exists": exists,
        "workspace_file_count": len(files),
        "git_repository": git_repo,
        "git_branch": branch,
        "git_dirty": dirty,
        "test_commands": list(test_commands),
        "missing": sorted(set(missing)),
        "block_reason": block_reason,
    }


def write_handoff(
    path: str | os.PathLike[str], *,
    contract: Mapping[str, Any],
    stage_completed: str,
    next_stage: str,
    acceptance_traceability: Sequence[Mapping[str, Any]],
    alignment: Mapping[str, Any],
    confirmed_facts: Sequence[Any] = (),
    decisions: Sequence[Any] = (),
    completed_actions: Sequence[Any] = (),
    artifacts: Sequence[Any] = (),
    tests: Sequence[Any] = (),
    open_issues: Sequence[Any] = (),
    risks: Sequence[Any] = (),
    next_actions: Sequence[Any] = (),
) -> Path:
    valid = validate_task_contract(contract)
    if not valid["valid"]:
        raise ValueError(valid["reason"])
    payload = {
        "task_id": contract["task_id"],
        "revision": contract["revision"],
        "contract_hash": contract["contract_hash"],
        "stage_completed": stage_completed,
        "confirmed_facts": list(confirmed_facts),
        "decisions": list(decisions),
        "completed_actions": list(completed_actions),
        "artifacts": list(artifacts),
        "tests": list(tests),
        "open_issues": list(open_issues),
        "risks": list(risks),
        "next_stage": next_stage,
        "next_actions": list(next_actions),
        "acceptance_traceability": [dict(item) for item in acceptance_traceability],
        "alignment": dict(alignment),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def validate_handoff(handoff: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    contract_result = validate_task_contract(contract)
    if not contract_result["valid"]:
        return contract_result
    for key in ("task_id", "revision", "contract_hash"):
        if handoff.get(key) != contract.get(key):
            return {"valid": False, "reason": f"{key}_mismatch"}
    alignment = handoff.get("alignment") or {}
    if alignment.get("status") != "aligned":
        return {"valid": False, "reason": "alignment_not_aligned"}
    if alignment.get("criteria_missing") or alignment.get("out_of_scope_changes") or alignment.get("unapproved_goals"):
        return {"valid": False, "reason": "alignment_gate_failed"}
    expected = {item.get("id") for item in contract.get("acceptance_criteria", [])}
    traced = {item.get("criterion_id") for item in handoff.get("acceptance_traceability", [])}
    missing = sorted(x for x in expected - traced if x)
    if missing:
        return {"valid": False, "reason": "criteria_untracked", "missing": missing}
    return {"valid": True, "reason": None}
