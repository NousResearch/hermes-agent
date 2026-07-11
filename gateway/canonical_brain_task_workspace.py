"""Cache-safe Canonical Task Workspace recovery for interrupted gateway runs.

This module runs only at a restart/resume boundary and only inside the gateway
worker thread.  It never mutates the system prompt, classifies user text, or
chooses among ambiguous cases.  Exact structured state is prepended to the
current replayable user-turn snapshot and, for one unambiguous active plan, mechanically
hydrates an empty TodoStore.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import queue
import threading
import time
from typing import Any, Mapping


MAX_DISCOVERY_CASES = 10
MAX_ACTIVE_CANDIDATES = 3
MAX_RESUME_NOTE_CHARS = 14_000
MAX_RECOVERY_SECONDS = 15.0

_EXACT_NOTE_PREFIX = (
    "[Canonical Task Workspace — exact restart resume]\n"
    "The durable model-authored plan below is linked exactly to this Discord thread. "
    "Continue from its resume_cursor/next unverified step. Do not blindly replay old tool "
    "calls; re-check live external state when needed, do not repeat completed steps, and "
    "persist a new task.plan.updated snapshot after meaningful progress. If a new user "
    "message follows, GPT decides whether it changes or supersedes this plan. Historical "
    "approval receipts are audit evidence only and do not restore dangerous authority.\n"
)


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _bounded_text(value: Any, maximum: int) -> tuple[str, bool]:
    text = str(value or "")
    if len(text) <= maximum:
        return text, False
    return text[:maximum] + "…", True


def _compact_value(
    value: Any,
    *,
    depth: int = 2,
    max_items: int = 12,
    max_text: int = 500,
) -> tuple[Any, bool]:
    """Return a bounded, JSON-safe structural copy of ``value``.

    String leaves may be shortened, and containers may omit tail entries, but
    the serialized JSON document itself is never sliced.
    """
    if isinstance(value, str):
        return _bounded_text(value, max_text)
    if value is None or isinstance(value, (bool, int, float)):
        return value, False
    if depth <= 0:
        return {"omitted_sha256": _sha256(value)}, True
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        truncated = len(value) > max_items
        for raw_key, nested in list(value.items())[:max_items]:
            key, key_truncated = _bounded_text(raw_key, 120)
            compact, nested_truncated = _compact_value(
                nested,
                depth=depth - 1,
                max_items=max_items,
                max_text=max_text,
            )
            if key in result:
                # Preserve a deterministic signal instead of silently
                # overwriting two long keys with the same compact prefix.
                key = f"field_{len(result)}_{_sha256(raw_key)[:12]}"
                key_truncated = True
            result[key] = compact
            truncated = truncated or key_truncated or nested_truncated
        return result, truncated
    if isinstance(value, (list, tuple)):
        result = []
        truncated = len(value) > max_items
        for nested in list(value)[:max_items]:
            compact, nested_truncated = _compact_value(
                nested,
                depth=depth - 1,
                max_items=max_items,
                max_text=max_text,
            )
            result.append(compact)
            truncated = truncated or nested_truncated
        return result, truncated
    return _bounded_text(value, max_text)


def _bounded_json_note(
    prefix: str,
    payload_variants: list[Mapping[str, Any]],
    *,
    emergency_reason: str,
) -> str:
    """Choose the richest structural payload that fits the hard note bound."""
    for payload in payload_variants:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        note = prefix + encoded
        if len(note) <= MAX_RESUME_NOTE_CHARS:
            return note

    emergency = {
        "status": "incomplete",
        "reason": emergency_reason,
        "omitted_payload_sha256": _sha256(payload_variants[0] if payload_variants else {}),
        "truncated": True,
        "query_instruction": (
            "Use canonical_brain_query with the exact current Discord thread and "
            "view=summary; then inspect an exact case_id with view=resume_bundle."
        ),
    }
    note = prefix + json.dumps(emergency, ensure_ascii=False, sort_keys=True)
    if len(note) > MAX_RESUME_NOTE_CHARS:  # fixed-size invariant, not a slicer
        raise AssertionError("fixed Canonical Task Workspace note exceeds hard bound")
    return note


def _query(*, deadline: float, **kwargs: Any) -> tuple[dict[str, Any], str | None]:
    """Run one Canonical Brain query within the shared recovery deadline.

    The query runs in a daemon thread so a final slow Cloud SQL operation cannot
    make the overall recovery call exceed its global wall-clock budget. The
    current gateway contextvars are copied into the worker so authorization
    remains bound to the observed session.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return {}, "recovery_deadline_exceeded"

    outcomes: queue.Queue[tuple[dict[str, Any], str | None]] = queue.Queue(maxsize=1)
    context = contextvars.copy_context()

    def _invoke() -> None:
        try:
            from tools.canonical_brain_tool import (
                canonical_brain_query_tool,
                check_canonical_brain_requirements,
            )

            if not check_canonical_brain_requirements():
                outcomes.put_nowait(({}, "canonical_brain_unavailable"))
                return
            value = json.loads(canonical_brain_query_tool(**kwargs))
            if not isinstance(value, dict) or value.get("success") is not True:
                outcomes.put_nowait(({}, "canonical_brain_query_failed"))
                return
            outcomes.put_nowait((value, None))
        except Exception:
            try:
                outcomes.put_nowait(({}, "canonical_brain_query_failed"))
            except queue.Full:
                pass

    worker = threading.Thread(
        target=lambda: context.run(_invoke),
        name="canonical-task-workspace-query",
        daemon=True,
    )
    worker.start()
    worker.join(timeout=max(0.0, remaining))
    if worker.is_alive():
        return {}, "recovery_deadline_exceeded"
    try:
        return outcomes.get_nowait()
    except queue.Empty:
        return {}, "canonical_brain_query_failed"


def _candidate_case_ids(
    thread_id: str,
    *,
    deadline: float,
) -> tuple[list[str], bool, str | None]:
    result, error = _query(
        deadline=deadline,
        thread_id=thread_id,
        view="summary",
        limit=80,
    )
    if error:
        return [], False, error
    case_ids = []
    for case in result.get("cases") or []:
        case_id = str(_dict(case).get("case_id") or "").strip()
        if case_id and case_id not in case_ids:
            case_ids.append(case_id)
        if len(case_ids) >= MAX_DISCOVERY_CASES:
            break
    return case_ids, bool(result.get("candidate_cases_truncated")), None


def _resume_case(
    case_id: str,
    *,
    deadline: float,
) -> tuple[dict[str, Any], str | None]:
    result, error = _query(
        deadline=deadline,
        case_id=case_id,
        view="resume_bundle",
        limit=80,
    )
    if error:
        return {}, error
    if result.get("support_incomplete"):
        support = _dict(result.get("support"))
        reasons = [str(value) for value in support.get("reasons") or [] if value]
        return {}, (
            "candidate_resume_support_incomplete"
            + (":" + ",".join(reasons) if reasons else "")
        )
    cases = result.get("cases") or []
    if len(cases) != 1 or not isinstance(cases[0], Mapping):
        return {}, "candidate_resume_unresolved"
    case = _dict(cases[0])
    if str(case.get("case_id") or "") != case_id:
        return {}, "candidate_resume_unresolved"
    return case, None


def _active_plan(case: Mapping[str, Any]) -> dict[str, Any]:
    workspace = _dict(case.get("workspace"))
    plan = _dict(workspace.get("plan"))
    if plan.get("state") != "active":
        return {}
    steps = plan.get("steps")
    if not isinstance(steps, list) or not any(
        isinstance(step, Mapping) and step.get("status") in {"pending", "in_progress"}
        for step in steps
    ):
        return {}
    return plan


def _canonical_todo_items(plan: Mapping[str, Any]) -> list[dict[str, str]]:
    items = []
    for step in plan.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        items.append({
            "id": str(step.get("id") or ""),
            "content": str(step.get("content") or ""),
            "status": str(step.get("status") or "pending"),
        })
    return items


def _local_todo_items(todo_store: Any) -> tuple[list[dict[str, str]], str | None]:
    if todo_store is None or not hasattr(todo_store, "read"):
        return [], "local_todo_unavailable"
    try:
        raw_items = todo_store.read()
    except Exception:
        return [], "local_todo_unavailable"
    if not isinstance(raw_items, list):
        return [], "local_todo_unavailable"
    items = []
    for item in raw_items:
        if not isinstance(item, Mapping):
            return [], "local_todo_unavailable"
        items.append({
            "id": str(item.get("id") or ""),
            "content": str(item.get("content") or ""),
            "status": str(item.get("status") or "pending"),
        })
    return items, None


def _local_todo_summary(items: list[Mapping[str, Any]]) -> dict[str, Any]:
    compact = []
    for item in items[:24]:
        item_id, _ = _bounded_text(item.get("id"), 160)
        compact.append({
            "id": item_id,
            "status": str(item.get("status") or "")[:32],
        })
    return {
        "count": len(items),
        "steps": compact,
        "truncated": len(items) > len(compact),
        "sha256": _sha256(items),
    }


def _compact_step(step: Mapping[str, Any], *, content_chars: int) -> tuple[dict[str, Any], bool]:
    step_id, id_truncated = _bounded_text(step.get("id"), 180)
    content, content_truncated = _bounded_text(step.get("content"), content_chars)
    dependencies = list(step.get("depends_on") or [])
    compact_dependencies = []
    dependencies_truncated = len(dependencies) > 12
    for dependency in dependencies[:12]:
        value, value_truncated = _bounded_text(dependency, 180)
        compact_dependencies.append(value)
        dependencies_truncated = dependencies_truncated or value_truncated
    blocker, blocker_truncated = _compact_value(
        _dict(step.get("blocker")),
        depth=2,
        max_items=8,
        max_text=300,
    )
    return {
        "id": step_id,
        "content": content,
        "status": str(step.get("status") or "")[:32],
        "depends_on": compact_dependencies,
        "blocker": blocker,
    }, (
        id_truncated
        or content_truncated
        or dependencies_truncated
        or blocker_truncated
    )


def _minimal_plan_steps(plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    steps = [step for step in plan.get("steps") or [] if isinstance(step, Mapping)]
    wanted_ids = [
        str(plan.get("current_step_id") or ""),
        str(_dict(plan.get("resume_cursor")).get("next_step_id") or ""),
    ]
    selected: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for wanted in wanted_ids:
        for step in steps:
            step_id = str(step.get("id") or "")
            if wanted and step_id == wanted and step_id not in seen:
                selected.append(step)
                seen.add(step_id)
    for step in steps:
        step_id = str(step.get("id") or "")
        if (
            len(selected) < 12
            and step_id not in seen
            and step.get("status") in {"pending", "in_progress", "blocked"}
        ):
            selected.append(step)
            seen.add(step_id)
    compact = []
    truncated = len(selected) < len([
        step for step in steps
        if step.get("status") in {"pending", "in_progress", "blocked"}
    ])
    for step in selected:
        step_id, id_truncated = _bounded_text(step.get("id"), 160)
        content, content_truncated = _bounded_text(step.get("content"), 180)
        compact.append({
            "id": step_id,
            "content": content,
            "status": str(step.get("status") or "")[:32],
        })
        truncated = truncated or id_truncated or content_truncated
    return compact, truncated


def _bounded_exact_note(
    case: Mapping[str, Any],
    *,
    todo_hydrated: bool,
    local_todo_state: str,
) -> str:
    workspace = _dict(case.get("workspace"))
    plan = _dict(workspace.get("plan"))
    truncated = False

    compact_steps = []
    raw_steps = [step for step in plan.get("steps") or [] if isinstance(step, Mapping)]
    if len(raw_steps) > 24:
        truncated = True
    for step in raw_steps[:24]:
        value, value_truncated = _compact_step(step, content_chars=650)
        compact_steps.append(value)
        truncated = truncated or value_truncated

    compact_criteria = []
    raw_criteria = [
        criterion
        for criterion in plan.get("success_criteria") or []
        if isinstance(criterion, Mapping)
    ]
    if len(raw_criteria) > 16:
        truncated = True
    for criterion in raw_criteria[:16]:
        criterion_id, id_truncated = _bounded_text(criterion.get("id"), 180)
        content, content_truncated = _bounded_text(
            criterion.get("content") or criterion.get("description"),
            350,
        )
        compact_criteria.append({"id": criterion_id, "content": content})
        truncated = truncated or id_truncated or content_truncated

    resume_cursor, cursor_truncated = _compact_value(
        _dict(plan.get("resume_cursor")),
        depth=2,
        max_items=12,
        max_text=900,
    )
    next_action, action_truncated = _compact_value(
        _dict(case.get("next_action")),
        depth=2,
        max_items=12,
        max_text=700,
    )
    truncated = truncated or cursor_truncated or action_truncated
    case_id, case_id_truncated = _bounded_text(case.get("case_id"), 240)
    plan_id, plan_id_truncated = _bounded_text(plan.get("plan_id"), 180)
    objective, objective_truncated = _bounded_text(plan.get("objective"), 1800)
    current_step_id, current_truncated = _bounded_text(plan.get("current_step_id"), 180)
    truncated = truncated or any((
        case_id_truncated,
        plan_id_truncated,
        objective_truncated,
        current_truncated,
    ))

    remaining_ids = []
    raw_remaining_ids = list(workspace.get("remaining_step_ids") or [])
    if len(raw_remaining_ids) > 32:
        truncated = True
    for raw_id in raw_remaining_ids[:32]:
        value, value_truncated = _bounded_text(raw_id, 180)
        remaining_ids.append(value)
        truncated = truncated or value_truncated

    verification_ids = []
    raw_verifications = [
        item
        for item in workspace.get("verifications") or []
        if isinstance(item, Mapping) and item.get("event_id")
    ]
    if len(raw_verifications) > 24:
        truncated = True
    for item in raw_verifications[:24]:
        value, value_truncated = _bounded_text(item.get("event_id"), 80)
        verification_ids.append(value)
        truncated = truncated or value_truncated

    detailed = {
        "case_id": case_id,
        "plan_event_id": str(workspace.get("plan_event_id") or "")[:80],
        "plan": {
            "plan_id": plan_id,
            "revision": plan.get("revision"),
            "objective": objective,
            "state": plan.get("state"),
            "current_step_id": current_step_id,
            "resume_cursor": resume_cursor,
            "success_criteria": compact_criteria,
            "steps": compact_steps,
        },
        "remaining_step_ids": remaining_ids,
        "verification_event_ids": verification_ids,
        "next_action": next_action,
        "todo_hydrated": todo_hydrated,
        "local_todo_state": local_todo_state,
        "approval_receipts_are_informational": True,
        "truncated": truncated,
    }

    minimal_steps, minimal_steps_truncated = _minimal_plan_steps(plan)
    minimal_cursor = {
        "next_step_id": _bounded_text(
            _dict(plan.get("resume_cursor")).get("next_step_id"), 180
        )[0],
        "summary": _bounded_text(
            _dict(plan.get("resume_cursor")).get("summary"), 700
        )[0],
    }
    minimal = {
        "case_id": case_id,
        "case_id_sha256": _sha256(case.get("case_id")),
        "plan_event_id": str(workspace.get("plan_event_id") or "")[:80],
        "plan": {
            "plan_id": plan_id,
            "revision": plan.get("revision"),
            "objective": _bounded_text(plan.get("objective"), 700)[0],
            "state": plan.get("state"),
            "current_step_id": current_step_id,
            "resume_cursor": minimal_cursor,
            "success_criterion_ids": [
                _bounded_text(item.get("id"), 160)[0]
                for item in raw_criteria[:12]
            ],
            "steps": minimal_steps,
        },
        "remaining_step_ids": [
            _bounded_text(value, 160)[0]
            for value in raw_remaining_ids[:16]
        ],
        "next_action": _compact_value(
            _dict(case.get("next_action")),
            depth=1,
            max_items=8,
            max_text=300,
        )[0],
        "todo_hydrated": todo_hydrated,
        "local_todo_state": local_todo_state,
        "approval_receipts_are_informational": True,
        "truncated": True,
        "query_instruction": (
            "Call canonical_brain_query with this exact case_id and "
            "view=resume_bundle for omitted details."
        ),
        "omitted_remaining_steps": minimal_steps_truncated,
    }
    return _bounded_json_note(
        _EXACT_NOTE_PREFIX,
        [detailed, minimal],
        emergency_reason="exact_resume_payload_exceeded_safe_bound",
    )


def _compact_candidate(case: Mapping[str, Any]) -> dict[str, Any]:
    plan = _dict(_dict(case.get("workspace")).get("plan"))
    cursor = _dict(plan.get("resume_cursor"))
    return {
        "case_id": _bounded_text(case.get("case_id"), 240)[0],
        "case_id_sha256": _sha256(case.get("case_id")),
        "plan_id": _bounded_text(plan.get("plan_id"), 180)[0],
        "revision": plan.get("revision"),
        "objective": _bounded_text(plan.get("objective"), 500)[0],
        "current_step_id": _bounded_text(plan.get("current_step_id"), 180)[0],
        "resume_cursor": {
            "next_step_id": _bounded_text(cursor.get("next_step_id"), 180)[0],
            "summary": _bounded_text(cursor.get("summary"), 500)[0],
        },
    }


def _bounded_ambiguous_note(
    cases: list[Mapping[str, Any]],
    *,
    reasons: list[str] | None = None,
    unresolved_case_ids: list[str] | None = None,
    local_todo: list[Mapping[str, Any]] | None = None,
) -> str:
    reasons = list(dict.fromkeys(str(reason) for reason in (reasons or []) if reason))
    unresolved_case_ids = list(dict.fromkeys(
        str(case_id) for case_id in (unresolved_case_ids or []) if case_id
    ))
    state = "incomplete" if reasons or unresolved_case_ids else "ambiguous"
    if state == "incomplete":
        prefix = (
            "[Canonical Task Workspace — incomplete restart resume]\n"
            "The exact thread lookup could not prove one complete, unambiguous active plan. "
            "Runtime has not selected or hydrated a plan. GPT must inspect the structured "
            "state, retry exact Canonical Brain reads when appropriate, and report a focused "
            "blocker or ask only if the intended case remains genuinely ambiguous. Do not "
            "select by keywords or replay old tool calls blindly.\n"
        )
    else:
        prefix = (
            "[Canonical Task Workspace — ambiguous restart resume]\n"
            "Several active plans are exactly linked to this thread. Runtime has not selected "
            "or hydrated one. GPT must inspect the structured candidates, use "
            "canonical_brain_query with an exact case_id when needed, and ask only if the "
            "intended case remains genuinely ambiguous. Do not select by keywords or replay "
            "old tool calls blindly.\n"
        )

    compact_unresolved = [
        {
            "case_id": _bounded_text(case_id, 240)[0],
            "case_id_sha256": _sha256(case_id),
        }
        for case_id in unresolved_case_ids[:MAX_DISCOVERY_CASES]
    ]
    payload = {
        "status": state,
        "reasons": reasons[:8],
        "candidates": [_compact_candidate(case) for case in cases[:MAX_ACTIVE_CANDIDATES]],
        "candidate_count_in_note": min(len(cases), MAX_ACTIVE_CANDIDATES),
        "active_candidates_truncated": len(cases) > MAX_ACTIVE_CANDIDATES,
        "unresolved_cases": compact_unresolved,
        "unresolved_cases_truncated": len(unresolved_case_ids) > len(compact_unresolved),
        "local_todo": _local_todo_summary(list(local_todo or [])) if local_todo is not None else None,
        "todo_hydrated": False,
        "truncated": (
            len(cases) > MAX_ACTIVE_CANDIDATES
            or len(unresolved_case_ids) > len(compact_unresolved)
        ),
    }
    minimal = {
        "status": state,
        "reasons": reasons[:8],
        "candidate_refs": [
            {
                "case_id": _bounded_text(case.get("case_id"), 160)[0],
                "case_id_sha256": _sha256(case.get("case_id")),
            }
            for case in cases[:MAX_ACTIVE_CANDIDATES]
        ],
        "unresolved_case_sha256": [
            _sha256(case_id) for case_id in unresolved_case_ids[:MAX_DISCOVERY_CASES]
        ],
        "local_todo_sha256": _sha256(local_todo) if local_todo is not None else None,
        "todo_hydrated": False,
        "truncated": True,
        "query_instruction": (
            "Retry canonical_brain_query for the exact current Discord thread before "
            "selecting or hydrating a task plan."
        ),
    }
    return _bounded_json_note(
        prefix,
        [payload, minimal],
        emergency_reason="incomplete_resume_payload_exceeded_safe_bound",
    )


def prepare_task_workspace_resume(
    *,
    thread_id: str,
    session_key: str,
    todo_store: Any,
    hydrate_local_state: bool = True,
) -> dict[str, Any]:
    """Return a replayable user-turn snapshot and mechanically restored local state."""
    del session_key  # authority comes from contextvars, never this caller string
    thread_id = str(thread_id or "").strip()
    if not thread_id:
        return {"status": "none", "note": ""}

    deadline = time.monotonic() + MAX_RECOVERY_SECONDS
    candidate_case_ids, discovery_truncated, discovery_error = _candidate_case_ids(
        thread_id,
        deadline=deadline,
    )
    if discovery_error:
        return {
            "status": "incomplete",
            "reason": discovery_error,
            "note": _bounded_ambiguous_note([], reasons=[discovery_error]),
            "candidate_case_ids": [],
            "unresolved_case_ids": [],
            "todo_hydrated": False,
        }

    candidates: list[dict[str, Any]] = []
    unresolved_case_ids: list[str] = []
    reasons: list[str] = []
    if discovery_truncated:
        reasons.append("candidate_discovery_truncated")

    for index, case_id in enumerate(candidate_case_ids[:MAX_DISCOVERY_CASES]):
        if time.monotonic() >= deadline:
            unresolved_case_ids.extend(candidate_case_ids[index:MAX_DISCOVERY_CASES])
            reasons.append("recovery_deadline_exceeded")
            break
        case, error = _resume_case(case_id, deadline=deadline)
        if error:
            unresolved_case_ids.append(case_id)
            reasons.append(error)
            if error == "recovery_deadline_exceeded":
                unresolved_case_ids.extend(
                    candidate_case_ids[index + 1:MAX_DISCOVERY_CASES]
                )
                break
            continue
        if _active_plan(case):
            candidates.append(case)

    if len(candidate_case_ids) > MAX_DISCOVERY_CASES:
        unresolved_case_ids.extend(candidate_case_ids[MAX_DISCOVERY_CASES:])
        reasons.append("candidate_check_limit_reached")

    # Any unresolved candidate makes uniqueness unknowable. Never hydrate or
    # report an exact plan merely because another candidate happened to load.
    if reasons or unresolved_case_ids:
        return {
            "status": "incomplete",
            "reason": reasons[0] if reasons else "candidate_resume_unresolved",
            "note": _bounded_ambiguous_note(
                candidates,
                reasons=reasons,
                unresolved_case_ids=unresolved_case_ids,
            ),
            "candidate_case_ids": [case.get("case_id") for case in candidates],
            "unresolved_case_ids": list(dict.fromkeys(unresolved_case_ids)),
            "discovery_truncated": discovery_truncated,
            "todo_hydrated": False,
        }

    if len(candidates) == 1:
        case = candidates[0]
        plan = _active_plan(case)
        canonical_items = _canonical_todo_items(plan)
        local_items, local_error = _local_todo_items(todo_store)
        if local_error:
            return {
                "status": "incomplete",
                "reason": local_error,
                "case_id": case.get("case_id"),
                "note": _bounded_ambiguous_note(
                    [case],
                    reasons=[local_error],
                ),
                "candidate_case_ids": [case.get("case_id")],
                "todo_hydrated": False,
            }
        if local_items and local_items != canonical_items:
            return {
                "status": "incomplete",
                "reason": "local_state_conflict",
                "case_id": case.get("case_id"),
                "note": _bounded_ambiguous_note(
                    [case],
                    reasons=["local_state_conflict"],
                    local_todo=local_items,
                ),
                "candidate_case_ids": [case.get("case_id")],
                "todo_hydrated": False,
                "local_state_conflict": True,
            }

        todo_hydrated = False
        local_todo_state = "exact_match" if local_items else "empty"
        if not local_items and hydrate_local_state:
            try:
                todo_store.write(canonical_items)
                todo_hydrated = True
                local_todo_state = "hydrated_from_canonical"
            except Exception:
                return {
                    "status": "incomplete",
                    "reason": "local_todo_hydration_failed",
                    "case_id": case.get("case_id"),
                    "note": _bounded_ambiguous_note(
                        [case],
                        reasons=["local_todo_hydration_failed"],
                    ),
                    "candidate_case_ids": [case.get("case_id")],
                    "todo_hydrated": False,
                }
        elif not local_items:
            local_todo_state = "empty_not_hydrated"

        return {
            "status": "exact",
            "case_id": case.get("case_id"),
            "note": _bounded_exact_note(
                case,
                todo_hydrated=todo_hydrated,
                local_todo_state=local_todo_state,
            ),
            "todo_hydrated": todo_hydrated,
            "local_todo_state": local_todo_state,
        }
    if len(candidates) > 1:
        return {
            "status": "ambiguous",
            "note": _bounded_ambiguous_note(candidates),
            "candidate_case_ids": [case.get("case_id") for case in candidates],
            "todo_hydrated": False,
        }
    return {"status": "none", "note": ""}


__all__ = ["prepare_task_workspace_resume"]
