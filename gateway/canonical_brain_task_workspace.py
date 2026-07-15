"""Cache-safe Canonical Task Workspace recovery at session boundaries.

This module runs only inside the gateway worker thread.  It never mutates the
system prompt, classifies user text, or chooses among ambiguous cases.  Exact
structured state is prepended to the current replayable user-turn snapshot.
At involuntary/fresh boundaries, one unambiguous execution-active plan may
mechanically hydrate an empty TodoStore.  A blocked plan is surfaced without
reactivation so GPT can evaluate new evidence.  An explicit ``/new`` is
different: it always preserves the user's fresh-conversation choice and
merely presents exact linked unfinished-plan candidates for GPT/the user to
continue, supersede, or leave behind.
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

BOUNDARY_RESTART_RESUME = "restart_resume"
BOUNDARY_INVOLUNTARY_RESET = "involuntary_reset"
BOUNDARY_FRESH_SESSION = "fresh_session"
BOUNDARY_EXPLICIT_NEW = "explicit_new"
_BOUNDARIES = frozenset(
    {
        BOUNDARY_RESTART_RESUME,
        BOUNDARY_INVOLUNTARY_RESET,
        BOUNDARY_FRESH_SESSION,
        BOUNDARY_EXPLICIT_NEW,
    }
)

_EXACT_NOTE_PREFIX = (
    "[Canonical Task Workspace — exact restart resume]\n"
    "The durable model-authored plan below is linked exactly to this Discord thread. "
    "Continue from its resume_cursor/next unverified step. Do not blindly replay old tool "
    "calls; re-check live external state when needed, do not repeat completed steps, and "
    "persist a new task.plan.updated snapshot after meaningful progress. If a new user "
    "message follows, GPT decides whether it changes or supersedes this plan. Historical "
    "approval receipts are audit evidence only and do not restore dangerous authority.\n"
)

_INVOLUNTARY_EXACT_NOTE_PREFIX = (
    "[Canonical Task Workspace — exact involuntary-session recovery]\n"
    "The local conversation was reset by an involuntary runtime boundary, but the "
    "durable model-authored plan below remains linked exactly to this Discord thread. "
    "Continue from its resume_cursor/next unverified step. Do not blindly replay old "
    "tool calls; re-check live external state when needed, do not repeat completed "
    "steps, and persist a new task.plan.updated snapshot after meaningful progress. "
    "If a new user message follows, GPT decides whether it changes or supersedes this "
    "plan. Historical approval receipts are audit evidence only and do not restore "
    "dangerous authority.\n"
)

_FRESH_EXACT_NOTE_PREFIX = (
    "[Canonical Task Workspace — exact fresh-session recovery]\n"
    "The durable model-authored plan below remains active and is linked exactly to this "
    "Discord thread, while the local conversation session is fresh. Continue from its "
    "resume_cursor/next unverified step. Do not blindly replay old tool calls; re-check "
    "live external state when needed, do not repeat completed steps, and persist a new "
    "task.plan.updated snapshot after meaningful progress. GPT decides whether the "
    "current user message changes or supersedes this plan. Historical approval receipts "
    "are audit evidence only and do not restore dangerous authority.\n"
)

_EXPLICIT_NEW_CHOICE_PREFIX = (
    "[Canonical Task Workspace — explicit-new unfinished-plan choices]\n"
    "The user explicitly opened a new conversation. Runtime has not selected or "
    "hydrated any prior plan. The exact active or blocked plan candidates below "
    "remain linked "
    "to this Discord thread only as choices. Treat the current user turn as fresh: "
    "GPT must surface the applicable continue, supersede, or keep-new choice to the "
    "user, unless the current user turn itself already makes that choice explicit. "
    "Never continue a candidate merely because it is the only one, never select by "
    "keywords, and never treat historical approval receipts as current authority.\n"
)

_BLOCKED_PLAN_NOTE_PREFIX = (
    "[Canonical Task Workspace — exact blocked-plan recovery]\n"
    "The durable model-authored plan below is linked exactly to this Discord thread "
    "and records a blocker contract. Runtime has not assumed that the blocker cleared, "
    "selected a workaround, hydrated executable local state, or restored historical "
    "approval authority. GPT must inspect the exact blocker/resume_when contract and the "
    "current user turn, re-check live read-only evidence when useful, and decide whether "
    "to author a new active plan revision and continue. If the blocker still applies, "
    "exhaust safe alternatives and ask only for the exact missing input or authority.\n"
)


def _exact_note_prefix(boundary: str) -> str:
    if boundary == BOUNDARY_INVOLUNTARY_RESET:
        return _INVOLUNTARY_EXACT_NOTE_PREFIX
    if boundary == BOUNDARY_FRESH_SESSION:
        return _FRESH_EXACT_NOTE_PREFIX
    return _EXACT_NOTE_PREFIX


def resolve_task_workspace_boundary(
    *,
    is_new_session: bool,
    was_auto_reset: bool,
    was_fresh_reset: bool,
    fresh_reset_reason: str | None,
) -> str | None:
    """Resolve one exact session-lifecycle boundary without reading user text.

    Unknown manual-reset reasons fail safe to ``explicit_new``: they may expose
    candidates but can never silently resume a plan.  Compression exhaustion is
    the sole ``reset_session`` reason that mechanically proves an involuntary
    boundary today.
    """

    if was_auto_reset:
        return BOUNDARY_INVOLUNTARY_RESET
    if was_fresh_reset:
        if str(fresh_reset_reason or "") == "compression_exhaustion":
            return BOUNDARY_INVOLUNTARY_RESET
        return BOUNDARY_EXPLICIT_NEW
    if is_new_session:
        return BOUNDARY_FRESH_SESSION
    return None


def attach_task_workspace_snapshot_to_user_turn(
    message: str,
    note: str,
    *,
    existing_provenance: Any = None,
) -> tuple[str, str, bool]:
    """Attach one exact snapshot and return API/persistence-identical text.

    Idempotency requires an exact internal provenance binding; authored text
    alone can never prove that a runtime snapshot is already attached.
    Persisting precisely what the API sees keeps the next turn's cached history
    prefix byte-stable.
    """

    message = str(message or "")
    note = str(note or "")
    if not note:
        return message, message, False
    from agent.message_provenance import (
        CANONICAL_WORKSPACE_NOTE_KIND,
        MESSAGE_PROVENANCE_KEY,
        message_fragment_is_bound,
        neutralize_untrusted_canonical_workspace_markers,
    )

    already_bound = message_fragment_is_bound(
        {MESSAGE_PROVENANCE_KEY: existing_provenance},
        kind=CANONICAL_WORKSPACE_NOTE_KIND,
        exact_text=note,
    )
    if already_bound and (
        message == note or message.startswith(note + "\n\n")
    ):
        return message, message, False
    message = str(
        neutralize_untrusted_canonical_workspace_markers(
            message,
            existing_provenance,
        )
        or ""
    )
    combined = note + ("\n\n" + message if message else "")
    return combined, combined, True


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
            "view=workspace_candidates; then inspect an exact case_id with "
            "view=resume_bundle."
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
        view="workspace_candidates",
        limit=80,
    )
    if error:
        return [], False, error
    case_ids = []
    for case in result.get("cases") or []:
        case_id = str(_dict(case).get("case_id") or "").strip()
        if case_id and case_id not in case_ids:
            case_ids.append(case_id)
    truncated = bool(
        result.get("candidate_cases_truncated")
        or len(case_ids) > MAX_DISCOVERY_CASES
    )
    return case_ids[:MAX_DISCOVERY_CASES], truncated, None


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


def _recoverable_plan(case: Mapping[str, Any]) -> dict[str, Any]:
    """Return exact non-terminal plan state without interpreting task prose.

    ``blocked`` is intentionally recoverable but never executable: its exact
    blocker contract must remain visible on the next user turn so GPT can
    decide whether new evidence clears it.  Runtime must not silently hydrate
    or reactivate it.
    """
    workspace = _dict(case.get("workspace"))
    plan = _dict(workspace.get("plan"))
    state = plan.get("state")
    if state == "blocked":
        blocker = _dict(plan.get("blocker"))
        if blocker.get("reason") and blocker.get("resume_when"):
            return plan
        return {}
    if state != "active":
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


def _recovered_todo_binding(
    case: Mapping[str, Any],
    plan: Mapping[str, Any],
    canonical_items: list[dict[str, str]],
) -> dict[str, Any]:
    """Build one exact binding without mutating the live TodoStore.

    Recovery is a Canonical readback, so the same public TodoStore validator
    used by normal checkpoints can prove that the recovered item snapshot is
    already normalized and checksum-bind it to the exact case/plan/event.  A
    throwaway store keeps malformed recovery metadata from creating a window
    in which executable local todos exist without their Canonical binding.
    """

    from tools.todo_tool import TodoStore

    verifier = TodoStore()
    normalized = verifier.write(canonical_items)
    if normalized != canonical_items:
        raise ValueError("recovered canonical todos are not normalized")
    return verifier.bind_canonical_workspace(
        case_id=str(case.get("case_id") or ""),
        plan_id=str(plan.get("plan_id") or ""),
        plan_revision=plan.get("revision"),
        plan_state=str(plan.get("state") or ""),
        plan_event_id=str(
            _dict(case.get("workspace")).get("plan_event_id") or ""
        ),
        # Historical query projections expose the exact event and plan body,
        # but not the append response's content digest. Empty is an explicit
        # supported value for this recovery-only field; the event id and exact
        # workspace Todo digest remain mandatory.
        canonical_content_sha256="",
        workspace_todos_sha256=_sha256(plan.get("steps") or []),
        items=canonical_items,
    )


def _discard_local_todos_after_binding_failure(todo_store: Any) -> bool:
    """Remove any executable local copy after a failed Canonical bind.

    ``install_verified_canonical_snapshot`` clears items and all binding/sync
    metadata atomically on the real runtime store.  The write fallback keeps
    small test/adaptor stores fail-closed as well.  The caller reports whether
    the empty state was read back exactly instead of assuming cleanup worked.
    """

    try:
        installer = getattr(
            todo_store,
            "install_verified_canonical_snapshot",
            None,
        )
        if callable(installer):
            installer([], binding=None)
        else:
            todo_store.write([])
        items, error = _local_todo_items(todo_store)
        if error or items:
            return False
        binding_reader = getattr(todo_store, "canonical_binding_state", None)
        return not callable(binding_reader) or binding_reader() is None
    except Exception:
        return False


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
    boundary: str,
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
    plan_blocker, plan_blocker_truncated = _compact_value(
        _dict(plan.get("blocker")),
        depth=2,
        max_items=12,
        max_text=900,
    )
    truncated = (
        truncated
        or cursor_truncated
        or action_truncated
        or plan_blocker_truncated
    )
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
            "blocker": plan_blocker,
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
            "blocker": _compact_value(
                _dict(plan.get("blocker")),
                depth=1,
                max_items=8,
                max_text=350,
            )[0],
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
    prefix = (
        _BLOCKED_PLAN_NOTE_PREFIX
        if plan.get("state") == "blocked"
        else _exact_note_prefix(boundary)
    )
    return _bounded_json_note(
        prefix,
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
        "state": plan.get("state"),
        "objective": _bounded_text(plan.get("objective"), 500)[0],
        "current_step_id": _bounded_text(plan.get("current_step_id"), 180)[0],
        "resume_cursor": {
            "next_step_id": _bounded_text(cursor.get("next_step_id"), 180)[0],
            "summary": _bounded_text(cursor.get("summary"), 500)[0],
        },
        "blocker": _compact_value(
            _dict(plan.get("blocker")),
            depth=1,
            max_items=8,
            max_text=350,
        )[0],
    }


def _bounded_ambiguous_note(
    cases: list[Mapping[str, Any]],
    *,
    boundary: str,
    reasons: list[str] | None = None,
    unresolved_case_ids: list[str] | None = None,
    local_todo: list[Mapping[str, Any]] | None = None,
) -> str:
    reasons = list(dict.fromkeys(str(reason) for reason in (reasons or []) if reason))
    unresolved_case_ids = list(dict.fromkeys(
        str(case_id) for case_id in (unresolved_case_ids or []) if case_id
    ))
    state = (
        "incomplete"
        if reasons or unresolved_case_ids
        else "choice"
        if boundary == BOUNDARY_EXPLICIT_NEW
        else "ambiguous"
    )
    if state == "incomplete":
        if boundary == BOUNDARY_EXPLICIT_NEW:
            prefix = (
                "[Canonical Task Workspace — incomplete explicit-new choices]\n"
                "The user explicitly opened a new conversation, but the exact thread "
                "lookup could not fully enumerate its unfinished-plan choices. Runtime has "
                "not selected, continued, or hydrated any plan. GPT must surface the "
                "loaded exact candidates as provisional choices, retry exact Canonical "
                "Brain reads when appropriate, and report the focused discovery blocker. "
                "Never select by keywords or treat a lone loaded candidate as complete.\n"
            )
        else:
            boundary_name = boundary.replace("_", "-")
            prefix = (
                f"[Canonical Task Workspace — incomplete {boundary_name} recovery]\n"
                "The exact thread lookup could not prove one complete, unambiguous unfinished "
                "plan. Runtime has not selected or hydrated a plan. GPT must inspect the "
                "structured state, retry exact Canonical Brain reads when appropriate, and "
                "report a focused blocker or ask only if the intended case remains genuinely "
                "ambiguous. Do not select by keywords or replay old tool calls blindly.\n"
            )
    elif state == "choice":
        prefix = _EXPLICIT_NEW_CHOICE_PREFIX
    else:
        prefix = (
            f"[Canonical Task Workspace — ambiguous {boundary.replace('_', '-')} recovery]\n"
            "Several active or blocked plans are exactly linked to this thread. Runtime has not selected "
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
            "Retry canonical_brain_query with view=workspace_candidates for the exact "
            "current Discord thread before selecting or hydrating a task plan."
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
    boundary: str = BOUNDARY_RESTART_RESUME,
) -> dict[str, Any]:
    """Return a replayable user-turn snapshot and any permitted local restoration."""
    del session_key  # authority comes from contextvars, never this caller string
    if boundary not in _BOUNDARIES:
        raise ValueError(f"unsupported Canonical Task Workspace boundary: {boundary!r}")
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
            "note": _bounded_ambiguous_note(
                [],
                boundary=boundary,
                reasons=[discovery_error],
            ),
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
        if _recoverable_plan(case):
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
                boundary=boundary,
                reasons=reasons,
                unresolved_case_ids=unresolved_case_ids,
            ),
            "candidate_case_ids": [case.get("case_id") for case in candidates],
            "unresolved_case_ids": list(dict.fromkeys(unresolved_case_ids)),
            "discovery_truncated": discovery_truncated,
            "todo_hydrated": False,
        }

    # ``/new`` is an explicit conversation boundary, not a recovery command.
    # Even one exact unfinished plan is only a choice. Return before touching the
    # local TodoStore so stale local state can neither select nor hydrate it.
    if boundary == BOUNDARY_EXPLICIT_NEW:
        if candidates:
            return {
                "status": "choice",
                "note": _bounded_ambiguous_note(
                    candidates,
                    boundary=boundary,
                ),
                "candidate_case_ids": [case.get("case_id") for case in candidates],
                "todo_hydrated": False,
            }
        return {"status": "none", "note": ""}

    if len(candidates) == 1:
        case = candidates[0]
        plan = _recoverable_plan(case)
        if plan.get("state") == "blocked":
            return {
                "status": "blocked",
                "case_id": case.get("case_id"),
                "plan_id": plan.get("plan_id"),
                "plan_revision": plan.get("revision"),
                "plan_event_id": _dict(case.get("workspace")).get("plan_event_id"),
                "note": _bounded_exact_note(
                    case,
                    boundary=boundary,
                    todo_hydrated=False,
                    local_todo_state="preserved_blocked_plan",
                ),
                "todo_hydrated": False,
                "local_todo_state": "preserved_blocked_plan",
            }
        canonical_items = _canonical_todo_items(plan)
        local_items, local_error = _local_todo_items(todo_store)
        if local_error:
            return {
                "status": "incomplete",
                "reason": local_error,
                "case_id": case.get("case_id"),
                "note": _bounded_ambiguous_note(
                    [case],
                    boundary=boundary,
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
                    boundary=boundary,
                    reasons=["local_state_conflict"],
                    local_todo=local_items,
                ),
                "candidate_case_ids": [case.get("case_id")],
                "todo_hydrated": False,
                "local_state_conflict": True,
            }

        todo_hydrated = False
        local_todo_state = "exact_match" if local_items else "empty"
        if not local_items and not hydrate_local_state:
            local_todo_state = "empty_not_hydrated"
        else:
            try:
                binding = _recovered_todo_binding(
                    case,
                    plan,
                    canonical_items,
                )
                installer = getattr(
                    todo_store,
                    "install_verified_canonical_snapshot",
                    None,
                )
                if not callable(installer):
                    raise TypeError(
                        "TodoStore cannot atomically install a Canonical binding"
                    )
                installer(canonical_items, binding=binding)
                if not local_items:
                    todo_hydrated = True
                    local_todo_state = "hydrated_from_canonical"
            except Exception:
                fail_closed = _discard_local_todos_after_binding_failure(
                    todo_store
                )
                reason = (
                    "canonical_todo_binding_failed"
                    if fail_closed
                    else "canonical_todo_binding_failed_state_unknown"
                )
                return {
                    "status": "incomplete",
                    "reason": reason,
                    "case_id": case.get("case_id"),
                    "note": _bounded_ambiguous_note(
                        [case],
                        boundary=boundary,
                        reasons=[reason],
                    ),
                    "candidate_case_ids": [case.get("case_id")],
                    "todo_hydrated": False,
                    "local_todo_state": (
                        "cleared_after_binding_failure"
                        if fail_closed
                        else "binding_failure_state_unknown"
                    ),
                }

        return {
            "status": "exact",
            "case_id": case.get("case_id"),
            # Expose only the exact structural plan binding.  Runtime safety
            # boundaries (for example owner-approval escalation) may need to
            # bind a receipt to the same model-authored plan without parsing
            # the human-facing recovery note or inspecting task prose.
            "plan_id": plan.get("plan_id"),
            "plan_revision": plan.get("revision"),
            "plan_event_id": _dict(case.get("workspace")).get("plan_event_id"),
            "note": _bounded_exact_note(
                case,
                boundary=boundary,
                todo_hydrated=todo_hydrated,
                local_todo_state=local_todo_state,
            ),
            "todo_hydrated": todo_hydrated,
            "local_todo_state": local_todo_state,
        }
    if len(candidates) > 1:
        return {
            "status": "ambiguous",
            "note": _bounded_ambiguous_note(candidates, boundary=boundary),
            "candidate_case_ids": [case.get("case_id") for case in candidates],
            "todo_hydrated": False,
        }
    return {"status": "none", "note": ""}


__all__ = [
    "BOUNDARY_EXPLICIT_NEW",
    "BOUNDARY_FRESH_SESSION",
    "BOUNDARY_INVOLUNTARY_RESET",
    "BOUNDARY_RESTART_RESUME",
    "attach_task_workspace_snapshot_to_user_turn",
    "prepare_task_workspace_resume",
    "resolve_task_workspace_boundary",
]
