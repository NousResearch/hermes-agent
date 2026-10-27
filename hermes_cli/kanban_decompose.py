"""Kanban decomposer — fan a triage task out into a graph of child tasks.

Invoked by ``hermes kanban decompose [task_id | --all]`` and the gateway
dispatcher's auto-decompose path. Reads the profile roster (with
descriptions), asks the auxiliary LLM for a task graph in JSON, then
atomically create the children, links them under the root, and flips the
root ``triage -> todo``. The root stays alive as the *child* (in the link
table) of every leaf so ``recompute_ready`` keeps it in ``todo`` until the
graph completes and its assignee (the orchestrator profile) wakes up to
judge completion and add more work.

Mirrors ``kanban_specify`` (lazy aux import, lenient parse, never raises on
expected failures). ``fanout=false`` collapses to the ``specify`` behaviour
(tighten + promote, no children), making ``decompose`` a strict superset.
Unknown assignees are rewritten to ``default_assignee`` — a child NEVER ends
up with ``assignee=None``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_routing_guard as rg
from hermes_cli import profiles as profiles_mod
from hermes_cli.kanban_specify import (
    _call_aux, _extract_json_blob, _load_triage_task, _task_prompt_fields, _title_body,
)
from hermes_cli.kanban_specify import _profile_author as _specify_author

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are the Kanban decomposer for the Hermes Agent board.

A user dropped a rough idea into the Triage column. Your job is to break it
into a small graph of concrete child tasks and route each one to the best-
matching profile from the available roster.

You will be given:
  - The original task title and body
  - The list of available profiles (each with name + description)
  - The fallback "default_assignee" used when no profile fits

Output a single JSON object with this exact shape:

  {
    "fanout": true,
    "rationale": "<one sentence on why this decomposition>",
    "tasks": [
      {
        "title": "<concrete task title, imperative voice, <= 80 chars>",
        "body":  "<detailed spec for the worker on this child task>",
        "assignee": "<profile name from the roster, or null for default>",
        "parents": [<int>, ...]
      },
      ...
    ]
  }

Rules:
  - "parents" is a list of INDICES (0-based) into this same "tasks" list,
    expressing actual data dependencies. Tasks with no parents run in
    PARALLEL. Tasks with parents wait until every parent completes.
  - Prefer parallelism. If two tasks can be done independently, give
    them no parents so the dispatcher fans them out at once.
  - Use 2-6 tasks for normal work. Don't create 20 tiny tasks. Don't
    cram everything into 1 task.
  - Pick assignees from the roster by matching the task to the profile's
    DESCRIPTION (not just the name). When nothing matches well, use null
    and the system will route to the default_assignee.
  - Each child task body is what a fresh worker will read with no other
    context — be specific about goal, approach, and acceptance criteria.

When the task is genuinely a single unit of work (no useful decomposition),
return:

  {
    "fanout": false,
    "rationale": "<one sentence>",
    "title": "<tightened title>",
    "body":  "<concrete spec for a single worker>",
    "assignee": "<profile name from the roster, or null for default>"
  }

In that case the task stays as one work item, just with a tightened spec and
a concrete assignee. If no profile fits, use null and the system will route to
the default_assignee.

No preamble, no closing remarks, no code fences. Output only the JSON object.
"""


_USER_TEMPLATE = """Task id: {task_id}
Title: {title}
Body:
{body}

Available profiles (assignees you may pick from):
{roster}

Default assignee (used when no profile fits a task): {default_assignee}
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class DecomposeOutcome:
    """Result of decomposing a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    fanout: bool = False
    child_ids: list[str] | None = None
    new_title: Optional[str] = None
    mode: str = "auto"  # "auto", "forced_fanout", "forced_tighten", "repair"
    new_status: Optional[str] = None  # set by the repair path only


def _profile_author() -> str:
    """Mirror of ``hermes_cli.kanban._profile_author``."""
    return _specify_author("decomposer")


def _resolve_profile_from_cfg(cfg: dict, key: str, *, fallback: Optional[str] = None) -> str:
    """``kanban.<key>`` if it names an existing profile, else ``fallback``
    (the root task's own assignee) if that does, else the active default
    profile — so a task is never stranded for lack of an owner.
    ``orchestrator_profile`` owns the root after fan-out; ``default_assignee``
    catches children the decomposer can't route.

    The root's assignee sits before the active profile because the decomposer
    runs inside whatever profile hosts the dispatcher — an operator's
    credential-less incognito profile, say — and that profile must never
    silently become the owner of work the card was assigned away from (#114294).
    """
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    explicit = (kanban_cfg.get(key) or "").strip()
    for candidate in (explicit, (fallback or "").strip()):
        if candidate:
            try:
                if profiles_mod.profile_exists(candidate):
                    return candidate
            except Exception:
                pass
    try:
        return profiles_mod.get_active_profile_name() or "default"
    except Exception:
        return "default"


def _build_roster() -> tuple[list[dict], set[str]]:
    """``(roster_for_prompt, valid_assignee_names)``; entries are
    ``{name, description, has_description}``."""
    try:
        all_profiles = profiles_mod.list_profiles()
    except Exception as exc:
        logger.warning("decompose: failed to list profiles: %s", exc)
        return [], set()
    roster = []
    for p in all_profiles:
        desc = (p.description or "").strip()
        roster.append({
            "name": p.name,
            "description": desc or f"(no description; profile named {p.name!r})",
            "has_description": bool(desc),
        })
    return roster, {p.name for p in all_profiles}


def _format_roster(roster: list[dict]) -> str:
    if not roster:
        return "  (no profiles installed — decomposer cannot route work)"
    return "\n".join(
        f"  - {entry['name']}{'' if entry['has_description'] else ' ⚠ undescribed'}: {entry['description']}"
        for entry in roster
    )


def _normalize_assignee_choice(assignee: object, *, default_assignee: str, valid_names: set[str]) -> str:
    """A valid assignee, else ``default_assignee`` — promoted work is never
    left unassigned."""
    if not isinstance(assignee, str) or not assignee.strip():
        return default_assignee
    chosen = assignee.strip()
    return chosen if chosen in valid_names else default_assignee


@dataclass
class _Routing:
    """Config-derived routing context for one decomposition."""

    orchestrator: str
    default_assignee: str
    auto_promote: bool
    roster: list[dict]
    valid_names: set[str]


def _load_routing(*, root_assignee: Optional[str] = None) -> _Routing:
    from hermes_cli.config import load_config_readonly
    try:
        cfg = load_config_readonly()
    except Exception:  # decompose_task promises ok=False, never a raise, on config trouble
        cfg = {}
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    roster, valid_names = _build_roster()
    return _Routing(
        orchestrator=_resolve_profile_from_cfg(cfg, "orchestrator_profile", fallback=root_assignee),
        default_assignee=_resolve_profile_from_cfg(cfg, "default_assignee", fallback=root_assignee),
        auto_promote=bool(kanban_cfg.get("auto_promote_children", True)),
        roster=roster,
        valid_names=valid_names,
    )


def _apply_single(task: kb.Task, parsed: dict, routing: _Routing, author: str, mode: str = "auto") -> DecomposeOutcome:
    """``fanout=false``: single-task spec promotion (same effect as specify)."""
    title_val, body_val = _title_body(parsed)
    assignee_val = None
    decision = None  # type: Optional[rg.GuardDecision]
    if not task.assignee:
        assignee_val = _normalize_assignee_choice(
            parsed.get("assignee"), default_assignee=routing.default_assignee, valid_names=routing.valid_names,
        )
        # Routing guard: a card editing a profile's own instruction surface never
        # routes to that profile. Apply after normalization so the guard sees the
        # same candidate the decomposer would have used.
        decision = rg.guard_assignee(
            assignee_val, title_val or "", body_val or "",
            default_assignee=routing.default_assignee, eligible_names=routing.valid_names,
        )
        if decision.rewritten:
            assignee_val = decision.assignee
            logger.info(
                "decompose: routing guard fired on task %s (single): %s -> %s (%s)",
                task.id, decision.original_assignee, decision.assignee, decision.lane_source,
            )
    if title_val is None and body_val is None:
        return DecomposeOutcome(task.id, False, "decomposer returned fanout=false with no title/body", mode=mode)
    with kbc.connect_closing() as conn:
        ok = kb.specify_triage_task(
            conn, task.id, title=title_val, body=body_val, assignee=assignee_val, author=author,
        )
        if ok and decision is not None and decision.rewritten:
            # Record the guard audit trail inside the same connection so the
            # event and the assignment are atomic from the caller's perspective.
            try:
                rg.record_guard_decision(conn, task.id, decision, author=author)
            except Exception:
                logger.exception("decompose: failed to record routing guard audit for task %s", task.id)
    if not ok:
        return DecomposeOutcome(task.id, False, "task moved out of triage before promotion", mode=mode)
    return DecomposeOutcome(task.id, True, "single task (no fanout)", fanout=False, new_title=title_val, mode=mode)


def _clean_children(task_id: str, raw_tasks: list, routing: _Routing) -> tuple[list[dict], list[tuple[int, rg.GuardDecision]], str]:
    """Validate/normalise the LLM's ``tasks`` list; ``(children, guard_decisions, "")`` or ``([], [], reason)``.
    Unknown assignees route to the default; never assignee=None.
    guard_decisions is a list of (child_index, decision) for children where the guard fired."""
    children: list[dict] = []
    guard_decisions: list[tuple[int, rg.GuardDecision]] = []
    for idx, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict):
            return [], [], f"tasks[{idx}] is not an object"
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            return [], [], f"tasks[{idx}].title is missing or empty"
        body = entry.get("body")
        assignee = entry.get("assignee")
        chosen = _normalize_assignee_choice(
            assignee, default_assignee=routing.default_assignee, valid_names=routing.valid_names,
        )
        if isinstance(assignee, str) and assignee.strip() and assignee.strip() not in routing.valid_names:
            logger.info(
                "decompose: task %s child %d picked unknown assignee %r — "
                "routing to default_assignee %r",
                task_id, idx, assignee, routing.default_assignee,
            )
        # Routing guard: a card editing a profile's own instruction surface never
        # routes to that profile. Apply per-child using the child's own title/body.
        decision = rg.guard_assignee(
            chosen, title.strip(), (body.strip() if isinstance(body, str) else ""),
            default_assignee=routing.default_assignee, eligible_names=routing.valid_names,
        )
        if decision.rewritten:
            chosen = decision.assignee
            guard_decisions.append((idx, decision))
            logger.info(
                "decompose: routing guard fired on task %s child %d: %s -> %s (%s)",
                task_id, idx, decision.original_assignee, decision.assignee, decision.lane_source,
            )
        parents = entry.get("parents") or []
        if not isinstance(parents, list):
            parents = []
        children.append({
            "title": title.strip()[:200],
            "body": body.strip() if isinstance(body, str) else "",
            "assignee": chosen,
            # Drop non-int, out-of-range and self parent indices.
            "parents": [p for p in parents if isinstance(p, int) and 0 <= p < len(raw_tasks) and p != idx],
        })
    return children, guard_decisions, ""


def _apply_fanout(task_id: str, parsed: dict, routing: _Routing, author: str, mode: str = "auto") -> DecomposeOutcome:
    raw_tasks = parsed.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return DecomposeOutcome(task_id, False, "decomposer returned fanout=true with empty tasks list", mode=mode)
    children, guard_decisions, reason = _clean_children(task_id, raw_tasks, routing)
    if reason:
        return DecomposeOutcome(task_id, False, reason, mode=mode)
    try:
        with kbc.connect_closing() as conn:
            child_ids = decompose_triage_task(
                conn,
                task_id,
                root_assignee=routing.orchestrator,
                children=children,
                author=author,
                auto_promote=routing.auto_promote,
            )
            # Record routing guard audit for each child where the guard fired.
            # The child_ids list is in the same order as the children list, so we
            # can map (child_index, decision) to the actual child task id.
            if child_ids and guard_decisions:
                for child_idx, decision in guard_decisions:
                    if 0 <= child_idx < len(child_ids):
                        child_task_id = child_ids[child_idx]
                        try:
                            rg.record_guard_decision(conn, child_task_id, decision, author=author)
                        except Exception:
                            logger.exception(
                                "decompose: failed to record routing guard audit for child %s (task %s child %d)",
                                child_task_id, task_id, child_idx,
                            )
    except ValueError as exc:
        return DecomposeOutcome(task_id, False, f"DB rejected graph: {exc}", mode=mode)
    except Exception as exc:
        # Size gate: OversizedChildError is a typed failure from the decomposer's
        # size contract. Surface it as a clear reason, not a crash.
        from hermes_cli.kanban_size_gate import OversizedChildError

        if isinstance(exc, OversizedChildError):
            logger.warning(
                "decompose: task %s refused - child %d (%s) is oversized (score=%d, threshold=%d)",
                task_id, exc.child_index, exc.child_title, exc.estimate.score, exc.threshold,
            )
            return DecomposeOutcome(
                task_id, False,
                f"size gate: {exc}",
                mode=mode,
            )
        logger.exception("decompose: DB error on task %s", task_id)
        return DecomposeOutcome(task_id, False, f"DB error: {type(exc).__name__}", mode=mode)
    if child_ids is None:
        return DecomposeOutcome(task_id, False, "task already decomposed or moved out of triage", mode=mode)
    return DecomposeOutcome(
        task_id, True, f"decomposed into {len(child_ids)} children", fanout=True, child_ids=child_ids, mode=mode,
    )


def _record_mode_event(task_id: str, *, mode: str, forced: bool, outcome: DecomposeOutcome) -> None:
    """Board-visible audit: the card shows WHY it fanned out or was rewritten.

    ``mode`` rides the outcome and the CLI ok-line, but neither is durable for
    whoever reads the board later - one ``decompose_mode`` task event records
    the mode, whether it was forced, and whether the run succeeded. An audit
    failure never masks the outcome.
    """
    try:
        with kbc.connect_closing() as conn:
            with kb.write_txn(conn, allow_nested=True):
                kb._append_event(conn, task_id, "decompose_mode", {
                    "mode": mode, "forced": forced, "fanout": outcome.fanout, "ok": outcome.ok,
                })
    except Exception:
        logger.exception("decompose: failed to record mode audit for task %s", task_id)


def decompose_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
    force_fanout: Optional[bool] = None,
) -> DecomposeOutcome:
    """Decompose a triage task into a graph of child tasks. Expected failures
    (not in triage, no aux client, API error, malformed/empty reply) surface
    as ``ok=False``.

    ``force_fanout``:
      - ``None`` (default): model chooses the mode (auto).
      - ``True``: force fan-out — produces children even when the model reply
        describes one task (wraps the single-task output as one child).
      - ``False``: force tighten — produces a single tightened card, never
        children. If the model reply disagrees (returned fanout=true), this is
        a typed failure, not a silent override.

    Repair: a triage card that already has a ``decomposed`` event is a stuck
    root (the size gate re-triaged it). Instead of calling the LLM and failing,
    repair it: move to ``ready`` if all children are done, else ``todo``. When
    the repair declines because the block-loop brake parked the card for a
    human, the card is left alone and this returns a typed failure — never the
    LLM, which cannot fan out a second time and runs on every sweep tick.
    """
    task, reason = _load_triage_task(task_id)
    if task is None:
        return DecomposeOutcome(task_id, False, reason)

    # Repair path: a stuck decomposed root in triage.
    from hermes_cli.kanban_db_graph import (
        latest_decomposed_payload, repair_stuck_decomposed_root, triage_park_cause,
    )
    with kbc.connect_closing() as conn:
        new_status = repair_stuck_decomposed_root(conn, task_id)
        already_decomposed = (
            new_status is None and latest_decomposed_payload(conn, task_id) is not None
        )
        parked_by = triage_park_cause(conn, task_id) if already_decomposed else None
    if new_status is not None:
        return DecomposeOutcome(
            task_id, True,
            f"repaired stuck decomposed root: moved to {new_status}",
            fanout=False, mode="repair", new_status=new_status,
        )
    if already_decomposed:
        # Parked deliberately (``block_loop_detected``) or moved by a
        # concurrent writer: a decomposed root can never fan out again, so the
        # LLM call would burn a tick and fail. Leave it where it is.
        return DecomposeOutcome(
            task_id, False,
            "already decomposed and parked for a human"
            f" ({parked_by or 'not a size-gate divert'}); not fanning out again",
        )

    routing = _load_routing(root_assignee=task.assignee)
    raw, reason = _call_aux(
        "decompose", task_id, aux_task="kanban_decomposer", system=_SYSTEM_PROMPT,
        user=_USER_TEMPLATE.format(
            **_task_prompt_fields(task),
            roster=_format_roster(routing.roster),
            default_assignee=routing.default_assignee,
        ),
        max_tokens=4000, timeout=timeout or 180, log=logger,
    )
    if raw is None:
        return DecomposeOutcome(task_id, False, reason)

    parsed = _extract_json_blob(raw, _FENCE_RE)
    if parsed is None:
        return DecomposeOutcome(task_id, False, "LLM returned malformed JSON")

    audit_author = author or _profile_author()
    model_fanout = bool(parsed.get("fanout"))

    # Determine effective mode and handle forced-vs-model disagreements.
    if force_fanout is True:
        mode = "forced_fanout"
        if not model_fanout:
            # Force fan-out: wrap the model's single-task output as one child.
            title_val, body_val = _title_body(parsed)
            assignee_val = _normalize_assignee_choice(
                parsed.get("assignee"), default_assignee=routing.default_assignee, valid_names=routing.valid_names,
            )
            synthetic_tasks = [{
                "title": title_val or task.title or "Tightened task",
                "body": body_val or "",
                "assignee": assignee_val,
                "parents": [],
            }]
            parsed = {"fanout": True, "tasks": synthetic_tasks, "rationale": "forced fan-out (model chose tighten)"}
            logger.info("decompose: task %s forced fan-out over model's tighten reply", task_id)
        outcome = _apply_fanout(task_id, parsed, routing, audit_author, mode=mode)
    elif force_fanout is False:
        mode = "forced_tighten"
        if model_fanout:
            # Force tighten but model returned fanout — typed failure, not silent collapse.
            outcome = DecomposeOutcome(
                task_id, False,
                "forced tighten but model returned fanout=true (refusing to silently collapse)",
                fanout=False, mode=mode,
            )
        else:
            outcome = _apply_single(task, parsed, routing, audit_author, mode=mode)
    else:
        mode = "auto"
        if not model_fanout:
            outcome = _apply_single(task, parsed, routing, audit_author, mode=mode)
        else:
            outcome = _apply_fanout(task_id, parsed, routing, audit_author, mode=mode)

    _record_mode_event(task_id, mode=mode, forced=force_fanout is not None, outcome=outcome)
    return outcome


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Return task ids currently in the triage column."""
    with kbc.connect_closing() as conn:
        rows = kb.list_tasks(conn, status="triage", tenant=tenant, limit=1000)
    return [row.id for row in rows]
