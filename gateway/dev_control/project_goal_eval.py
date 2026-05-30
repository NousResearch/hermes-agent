"""Evidence assembly, machine-criteria checks, and judge-based re-evaluation."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from gateway.dev_control.acceptance_criteria import MACHINE_CHECKABLE_METHODS, normalize_acceptance_criteria
from gateway.dev_control.acceptance_verification import DevVerificationStore
from gateway.dev_control.ci_status import fetch_ci_status
from gateway.dev_control.plan_artifacts import DevPlanArtifactStore
from gateway.dev_control.project_goals import (
    DevProjectGoalStore,
    recompute_rollup,
)
from gateway.dev_control.project_scope import DEFAULT_PROJECT_ID
from gateway.dev_execution import DevExecutionStore, TASK_COMPLETED_STATUSES, TASK_FAILED_STATUSES
from hermes_cli.goals import (
    DEFAULT_JUDGE_TIMEOUT,
    _goal_judge_max_tokens,
    _goal_judge_route,
    _parse_judge_response,
    _truncate,
)

logger = logging.getLogger(__name__)

PROJECT_JUDGE_SYSTEM_PROMPT = (
    "You are a strict judge evaluating whether a project subgoal is satisfied "
    "based on collected execution evidence (verification results, CI status, "
    "and task statuses). Manual criteria must have concrete supporting evidence "
    "in the digest — do not infer completion from generic summaries.\n\n"
    "Reply ONLY with a single JSON object on one line:\n"
    '{"done": <true|false>, "reason": "<one-sentence rationale>"}'
)

PROJECT_JUDGE_USER_PROMPT = (
    "Subgoal:\n{goal}\n\n"
    "Manual criteria to evaluate:\n{manual_criteria_block}\n\n"
    "Evidence digest:\n{evidence}\n\n"
    "Are all manual criteria satisfied based on this evidence?"
)

_EVIDENCE_SNIPPET_CHARS = 6000


@dataclass
class MachineCriteriaReport:
    all_passed: bool
    results: List[Dict[str, Any]] = field(default_factory=list)
    manual_criteria: List[Dict[str, Any]] = field(default_factory=list)


def project_goals_tick_enabled() -> bool:
    raw = str(os.getenv("HERMES_DEV_PROJECT_GOALS_TICK") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_plan_id(
    subgoal: Dict[str, Any],
    *,
    plan_artifact_store: Optional[DevPlanArtifactStore] = None,
) -> Optional[str]:
    payload = subgoal.get("payload") or {}
    plan_id = str(payload.get("plan_id") or "").strip()
    if plan_id:
        return plan_id
    artifact_id = str(subgoal.get("plan_artifact_id") or "").strip()
    if not artifact_id or plan_artifact_store is None:
        return None
    builds = plan_artifact_store.list_builds(artifact_id, limit=1)
    if not builds:
        return None
    return str(builds[0].get("plan_id") or "").strip() or None


def assemble_evidence(
    subgoal: Dict[str, Any],
    *,
    verification_store: Optional[DevVerificationStore] = None,
    execution_store: Optional[DevExecutionStore] = None,
    plan_artifact_store: Optional[DevPlanArtifactStore] = None,
    ci_fetcher: Callable[..., Dict[str, Any]] = fetch_ci_status,
) -> Dict[str, Any]:
    plan_id = resolve_plan_id(subgoal, plan_artifact_store=plan_artifact_store)
    evidence: Dict[str, Any] = {
        "goal_id": subgoal.get("goal_id"),
        "plan_id": plan_id,
        "plan_artifact_id": subgoal.get("plan_artifact_id"),
        "verification": {},
        "execution_plan": {},
        "ci": {},
        "assembled_at": time.time(),
    }

    if plan_id and execution_store is not None:
        plan = execution_store.get_plan(plan_id) or {}
        tasks = plan.get("tasks") or []
        evidence["execution_plan"] = {
            "plan_id": plan_id,
            "status": plan.get("status"),
            "title": plan.get("title"),
            "tasks": [
                {
                    "task_id": task.get("task_id"),
                    "goal": task.get("goal"),
                    "status": task.get("status"),
                    "profile_id": task.get("profile_id"),
                }
                for task in tasks
            ],
            "completed_task_count": sum(
                1 for task in tasks if str(task.get("status") or "").lower() in TASK_COMPLETED_STATUSES
            ),
            "failed_task_count": sum(
                1 for task in tasks if str(task.get("status") or "").lower() in TASK_FAILED_STATUSES
            ),
        }
        repo, ref = _repo_ref_from_plan(plan)
        if repo and ref:
            evidence["ci"] = ci_fetcher(repo=repo, ref=ref)

    if plan_id and verification_store is not None:
        runs = verification_store.list_runs(plan_id=plan_id, limit=1)
        if runs:
            latest = runs[0]
            evidence["verification"] = {
                "verification_run_id": latest.get("verification_run_id"),
                "status": latest.get("status"),
                "verdict": latest.get("verdict"),
                "acceptance_verification_score": latest.get("acceptance_verification_score"),
                "results": latest.get("results") or [],
                "warnings": latest.get("warnings") or [],
            }

    return evidence


def format_evidence_digest(evidence: Dict[str, Any]) -> str:
    return _truncate(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True), _EVIDENCE_SNIPPET_CHARS)


def check_machine_criteria(
    subgoal: Dict[str, Any],
    evidence: Dict[str, Any],
) -> MachineCriteriaReport:
    criteria = normalize_acceptance_criteria(subgoal.get("acceptance_criteria"))
    machine: List[Dict[str, Any]] = []
    manual: List[Dict[str, Any]] = []
    for criterion in criteria:
        method = str(criterion.get("verification_method") or "manual").lower()
        if criterion.get("machine_checkable") and method in MACHINE_CHECKABLE_METHODS:
            machine.append(criterion)
        else:
            manual.append(criterion)

    verification_results = (evidence.get("verification") or {}).get("results") or []
    results: List[Dict[str, Any]] = []
    all_passed = True
    for criterion in machine:
        passed, reason = _machine_criterion_passed(criterion, verification_results, evidence)
        results.append({
            "statement": criterion.get("statement"),
            "verification_method": criterion.get("verification_method"),
            "verification_detail": criterion.get("verification_detail"),
            "passed": passed,
            "reason": reason,
        })
        if not passed:
            all_passed = False

    return MachineCriteriaReport(all_passed=all_passed, results=results, manual_criteria=manual)


def judge_project_goal(
    goal: Dict[str, Any],
    evidence_digest: str,
    *,
    manual_criteria: Optional[List[Dict[str, Any]]] = None,
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
) -> tuple[str, str, bool]:
    """Return ``(verdict, reason, parse_failed)`` with fail-open semantics."""

    manual = manual_criteria or []
    if not manual:
        return "done", "no manual criteria to judge", False

    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:
        logger.debug("project goal judge: auxiliary client import failed: %s", exc)
        return "continue", "auxiliary client unavailable", False

    manual_block = "\n".join(
        f"- {index}. {item.get('statement', '').strip()}"
        for index, item in enumerate(manual, start=1)
        if str(item.get("statement") or "").strip()
    )
    goal_text = _truncate(
        f"{goal.get('title', '').strip()}\n{goal.get('markdown', '').strip()}".strip(),
        2000,
    )
    prompt = PROJECT_JUDGE_USER_PROMPT.format(
        goal=goal_text,
        manual_criteria_block=_truncate(manual_block or "(none)", 2000),
        evidence=_truncate(evidence_digest, _EVIDENCE_SNIPPET_CHARS),
    )
    current_time = datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    try:
        judge_provider, judge_model = _goal_judge_route()
        resp = call_llm(
            "goal_judge",
            provider=judge_provider,
            model=judge_model,
            messages=[
                {"role": "system", "content": PROJECT_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Current time: {current_time}\n\n{prompt}"},
            ],
            temperature=0,
            max_tokens=_goal_judge_max_tokens(),
            timeout=timeout,
        )
    except Exception as exc:
        logger.info("project goal judge: API call failed (%s) — fail-open continue", exc)
        return "continue", f"judge error: {type(exc).__name__}", False

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    done, reason, parse_failed = _parse_judge_response(raw)
    verdict = "done" if done else "continue"
    logger.info("project goal judge: verdict=%s reason=%s", verdict, _truncate(reason, 120))
    return verdict, reason, parse_failed


def reevaluate_project_goal(
    *,
    store: DevProjectGoalStore,
    goal_id: str,
    verification_store: Optional[DevVerificationStore] = None,
    execution_store: Optional[DevExecutionStore] = None,
    plan_artifact_store: Optional[DevPlanArtifactStore] = None,
    ci_fetcher: Callable[..., Dict[str, Any]] = fetch_ci_status,
) -> Dict[str, Any]:
    goal = store.get(goal_id)
    if not goal:
        raise KeyError(f"Project goal not found: {goal_id}")
    if goal.get("kind") != "subgoal":
        raise ValueError("Only subgoal nodes can be re-evaluated")
    if goal.get("status") == "achieved":
        return {
            "object": "hermes.dev_project_goal_reevaluation",
            "goal_id": goal_id,
            "verdict": "skipped",
            "reason": "already achieved",
            "transition": None,
        }
    if goal.get("status") == "abandoned":
        return {
            "object": "hermes.dev_project_goal_reevaluation",
            "goal_id": goal_id,
            "verdict": "skipped",
            "reason": "abandoned",
            "transition": None,
        }

    evidence = assemble_evidence(
        goal,
        verification_store=verification_store,
        execution_store=execution_store,
        plan_artifact_store=plan_artifact_store,
        ci_fetcher=ci_fetcher,
    )
    digest = format_evidence_digest(evidence)
    machine = check_machine_criteria(goal, evidence)
    result: Dict[str, Any] = {
        "object": "hermes.dev_project_goal_reevaluation",
        "goal_id": goal_id,
        "machine_criteria": machine.results,
        "evidence": evidence,
        "verdict": "continue",
        "reason": "",
        "transition": None,
    }

    if not machine.all_passed:
        result["reason"] = "machine-checkable criteria not satisfied"
        store.append_judge_audit(goal_id, {
            "verdict": "continue",
            "reason": result["reason"],
            "machine_criteria": machine.results,
        })
        return result

    if not machine.manual_criteria:
        updated = store.update(goal_id, {"status": "achieved"})
        if updated.get("parent_goal_id"):
            recompute_rollup(store, updated["parent_goal_id"])
        store.append_judge_audit(goal_id, {
            "verdict": "done",
            "reason": "all machine-checkable criteria passed; no manual criteria",
            "machine_criteria": machine.results,
        })
        result.update({
            "verdict": "done",
            "reason": "all machine-checkable criteria passed",
            "transition": {"goal_id": goal_id, "to": "achieved"},
        })
        return result

    verdict, reason, parse_failed = judge_project_goal(goal, digest, manual_criteria=machine.manual_criteria)
    store.append_judge_audit(goal_id, {
        "verdict": verdict,
        "reason": reason,
        "parse_failed": parse_failed,
        "machine_criteria": machine.results,
    })
    result["verdict"] = verdict
    result["reason"] = reason
    result["parse_failed"] = parse_failed
    if verdict == "done":
        updated = store.update(goal_id, {"status": "achieved"})
        if updated.get("parent_goal_id"):
            recompute_rollup(store, updated["parent_goal_id"])
        result["transition"] = {"goal_id": goal_id, "to": "achieved", "reason": reason}
    return result


def goals_tick(
    *,
    store: DevProjectGoalStore,
    project_id: Optional[str] = None,
    verification_store: Optional[DevVerificationStore] = None,
    execution_store: Optional[DevExecutionStore] = None,
    plan_artifact_store: Optional[DevPlanArtifactStore] = None,
    ci_fetcher: Callable[..., Dict[str, Any]] = fetch_ci_status,
) -> Dict[str, Any]:
    project_ids = [resolve_project_id(project_id)] if project_id else _project_ids_with_active_subgoals(store)
    transitions: List[Dict[str, Any]] = []
    evaluated = 0
    for pid in project_ids:
        for subgoal in store.list(project_id=pid, kind="subgoal", status="active", include_abandoned=False):
            evaluated += 1
            outcome = reevaluate_project_goal(
                store=store,
                goal_id=subgoal["goal_id"],
                verification_store=verification_store,
                execution_store=execution_store,
                plan_artifact_store=plan_artifact_store,
                ci_fetcher=ci_fetcher,
            )
            transition = outcome.get("transition")
            if transition:
                transitions.append(transition)
    return {
        "object": "hermes.dev_project_goals_tick",
        "project_ids": project_ids,
        "evaluated": evaluated,
        "transitions": transitions,
    }


def maybe_run_goals_tick(*, db_path: Path, project_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not project_goals_tick_enabled():
        return None
    store = DevProjectGoalStore(db_path=db_path)
    try:
        verification_store = DevVerificationStore(db_path=db_path)
        execution_store = DevExecutionStore(db_path=db_path)
        plan_artifact_store = DevPlanArtifactStore(db_path=db_path)
        try:
            return goals_tick(
                store=store,
                project_id=project_id,
                verification_store=verification_store,
                execution_store=execution_store,
                plan_artifact_store=plan_artifact_store,
            )
        finally:
            plan_artifact_store.close()
            execution_store.close()
            verification_store.close()
    finally:
        store.close()


def _machine_criterion_passed(
    criterion: Dict[str, Any],
    verification_results: List[Dict[str, Any]],
    evidence: Dict[str, Any],
) -> tuple[bool, str]:
    if not verification_results:
        return False, "no verification results available"

    statement = _normalize_text(criterion.get("statement"))
    detail = _normalize_text(criterion.get("verification_detail"))
    for result in verification_results:
        if not _result_matches_criterion(result, statement, detail):
            continue
        if result.get("passed") is True:
            return True, "verification result passed"
        status = str(result.get("status") or "").lower()
        if status in {"passed", "success", "completed"}:
            return True, f"verification status={status}"
        return False, f"verification result not passing (status={status or 'unknown'})"
    return False, "no matching verification result"


def _result_matches_criterion(
    result: Dict[str, Any],
    statement: str,
    detail: str,
) -> bool:
    result_statement = _normalize_text(result.get("statement"))
    result_detail = _normalize_text(result.get("verification_detail") or result.get("command_run"))
    if statement and result_statement == statement:
        return True
    if detail and result_detail == detail:
        return True
    if detail and detail in result_detail:
        return True
    return False


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _repo_ref_from_plan(plan: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    payload = plan.get("payload") if isinstance(plan.get("payload"), dict) else {}
    repo = str((payload or {}).get("repo") or (payload or {}).get("github_repo") or "").strip()
    ref = str((payload or {}).get("ref") or (payload or {}).get("branch") or "").strip()
    for task in plan.get("tasks") or []:
        task_payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
        nested = task_payload.get("payload") if isinstance(task_payload.get("payload"), dict) else {}
        repo = repo or str(
            task_payload.get("repo")
            or task_payload.get("github_repo")
            or nested.get("repo")
            or nested.get("github_repo")
            or ""
        ).strip()
        ref = ref or str(
            task_payload.get("ref")
            or task_payload.get("branch")
            or nested.get("ref")
            or nested.get("branch")
            or ""
        ).strip()
    return repo or None, ref or None


def _project_ids_with_active_subgoals(store: DevProjectGoalStore) -> List[str]:
    rows = store.list(kind="subgoal", status="active", include_abandoned=False, limit=500)
    project_ids = sorted({str(row.get("project_id") or DEFAULT_PROJECT_ID) for row in rows})
    return project_ids or [DEFAULT_PROJECT_ID]


def resolve_project_id(value: Optional[str]) -> str:
    text = str(value or "").strip()
    return text or DEFAULT_PROJECT_ID
