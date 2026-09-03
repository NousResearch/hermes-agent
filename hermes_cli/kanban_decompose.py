"""Kanban decomposer — fan a triage task out into a graph of child tasks.

Invoked by ``hermes kanban decompose [task_id | --all]`` and the
auto-decompose path in the gateway dispatcher loop. Reads the user's
profile roster (with descriptions) and asks the auxiliary LLM to
return a task graph in JSON. Then atomically creates the children,
links them under the root, and flips the root ``triage -> todo``.

The root task stays alive and becomes the parent of every leaf child,
so when the whole graph completes the root wakes back up — its
assignee (the orchestrator profile) gets a chance to judge completion
and add more tasks if the work isn't done yet.

Design notes
------------

* Mirrors the shape of ``hermes_cli/kanban_specify.py``: lazy aux
  client import inside the function, lenient response parse, never
  raises on expected failure modes.

* The system prompt sees the *configured* profile roster — names plus
  descriptions plus the default fallback. Profiles without a
  description are still listed (with a note) so the decomposer can
  match on name as a fallback, but the user has an obvious incentive
  to describe them.

* ``fanout=false`` collapses to the same effect as ``kanban specify``:
  we tighten the body and flip ``triage -> todo`` as a single task,
  no children created. This makes ``decompose`` a strict superset of
  ``specify`` from the user's perspective.

* If the LLM picks an assignee that doesn't exist as a profile, we
  rewrite it to the configured ``default_assignee`` (or the default
  profile if unset). A child task NEVER ends up with ``assignee=None``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import profiles as profiles_mod

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


@dataclass
class PredictedRouting:
    """Pure prediction produced by steps 1-4 of the routing pipeline.

    Shared by the live (mutating) path and the dry-run (non-mutating)
    path so the two can never diverge in *what* gets predicted, only in
    whether the prediction is persisted.
    """

    ok: bool
    reason: str = ""
    fanout: bool = False
    rationale: str = ""
    orchestrator: str = ""
    default_assignee: str = ""
    roster: list[dict] | None = None
    # single-task (fanout=false) path
    title: Optional[str] = None
    body: Optional[str] = None
    assignee: Optional[str] = None
    # fan-out (fanout=true) path — validated children, never written here
    children: list[dict] | None = None


class _AdHocTask:
    """Stand-in for ``kb.Task`` when previewing routing on text that has
    no backing DB row (dry-run's ``title``/``body`` input mode)."""

    def __init__(self, title: str, body: str) -> None:
        self.id = "(preview)"
        self.title = title
        self.body = body
        self.status = "triage"
        self.assignee: Optional[str] = None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _extract_json_blob(raw: str) -> Optional[dict]:
    if not raw:
        return None
    stripped = _FENCE_RE.sub("", raw.strip())
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = stripped[first : last + 1]
    try:
        val = json.loads(candidate)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(val, dict):
        return None
    return val


def _profile_author() -> str:
    """Mirror of ``hermes_cli.kanban._profile_author``."""
    return (
        os.environ.get("HERMES_PROFILE")
        or os.environ.get("USER")
        or "decomposer"
    )


def _load_config() -> dict:
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception:
        return {}


def _resolve_orchestrator_profile(cfg: dict) -> str:
    """Resolve which profile owns the root/orchestration task after fan-out.

    Falls back to the active default profile when ``kanban.orchestrator_profile``
    is unset, so a task is never stranded for lack of an orchestrator.
    """
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    explicit = (kanban_cfg.get("orchestrator_profile") or "").strip()
    if explicit:
        try:
            if profiles_mod.profile_exists(explicit):
                return explicit
        except Exception:
            pass
    # Fall back to the active default profile.
    try:
        return profiles_mod.get_active_profile_name() or "default"
    except Exception:
        return "default"


def _resolve_default_assignee(cfg: dict) -> str:
    """Resolve which profile catches child tasks the orchestrator can't route."""
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    explicit = (kanban_cfg.get("default_assignee") or "").strip()
    if explicit:
        try:
            if profiles_mod.profile_exists(explicit):
                return explicit
        except Exception:
            pass
    try:
        return profiles_mod.get_active_profile_name() or "default"
    except Exception:
        return "default"


def _build_roster() -> tuple[list[dict], set[str]]:
    """Return (roster_for_prompt, valid_assignee_names).

    Each roster entry is ``{name, description, has_description}``. The
    valid-set is used after the LLM responds to rewrite invalid
    assignees to the default fallback.
    """
    roster: list[dict] = []
    valid: set[str] = set()
    try:
        all_profiles = profiles_mod.list_profiles()
    except Exception as exc:
        logger.warning("decompose: failed to list profiles: %s", exc)
        return roster, valid
    for p in all_profiles:
        desc = (p.description or "").strip()
        roster.append({
            "name": p.name,
            "description": desc or f"(no description; profile named {p.name!r})",
            "has_description": bool(desc),
        })
        valid.add(p.name)
    return roster, valid


def _format_roster(roster: list[dict]) -> str:
    if not roster:
        return "  (no profiles installed — decomposer cannot route work)"
    lines = []
    for entry in roster:
        tag = "" if entry["has_description"] else " ⚠ undescribed"
        lines.append(f"  - {entry['name']}{tag}: {entry['description']}")
    return "\n".join(lines)


def _normalize_assignee_choice(
    assignee: object,
    *,
    default_assignee: str,
    valid_names: set[str],
) -> str:
    """Return a valid assignee, falling back to ``default_assignee``.

    Fan-out children and the single-task fallback should share the same
    routing guarantee: promoted work must not be left unassigned.
    """
    if not isinstance(assignee, str) or not assignee.strip():
        return default_assignee
    chosen = assignee.strip()
    if chosen not in valid_names:
        return default_assignee
    return chosen


def _predict_routing(
    task: "kb.Task | _AdHocTask",
    *,
    task_id_for_log: str,
    timeout: Optional[int] = None,
) -> PredictedRouting:
    """Steps 1-4 of routing: read (already done by caller), resolve roster,
    call the LLM, parse + validate its JSON. Pure — never touches the DB.

    Shared verbatim by ``decompose_task`` (live) and ``dry_run_route``
    (preview) so the two can never predict differently; only what happens
    *after* this call (persist vs. return) differs.
    """
    cfg = _load_config()
    orchestrator = _resolve_orchestrator_profile(cfg)
    default_assignee = _resolve_default_assignee(cfg)
    roster, valid_names = _build_roster()

    try:
        from agent.auxiliary_client import call_llm  # type: ignore
    except Exception as exc:
        logger.debug("decompose: auxiliary client import failed: %s", exc)
        return PredictedRouting(False, "auxiliary client unavailable")

    user_msg = _USER_TEMPLATE.format(
        task_id=task.id,
        title=_truncate(task.title or "", 400),
        body=_truncate(task.body or "(no body)", 4000),
        roster=_format_roster(roster),
        default_assignee=default_assignee,
    )

    try:
        # Route through call_llm so auxiliary.kanban_decomposer.* config
        # (provider/model/base_url, extra_body, reasoning_effort, retries)
        # all apply — the previous direct client.chat.completions.create()
        # path dropped auxiliary.<task>.extra_body entirely (#35566).
        resp = call_llm(
            task="kanban_decomposer",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=4000,
            timeout=timeout or 180,
        )
    except Exception as exc:
        logger.info(
            "decompose: API call failed for %s (%s)", task_id_for_log, exc,
        )
        return PredictedRouting(False, f"LLM error: {type(exc).__name__}")

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    parsed = _extract_json_blob(raw)
    if parsed is None:
        return PredictedRouting(False, "LLM returned malformed JSON")

    fanout = bool(parsed.get("fanout"))
    _rationale_raw = parsed.get("rationale")
    rationale = _rationale_raw if isinstance(_rationale_raw, str) else ""

    if not fanout:
        new_title = parsed.get("title")
        new_body = parsed.get("body")
        title_val = new_title.strip() if isinstance(new_title, str) and new_title.strip() else None
        body_val = new_body if isinstance(new_body, str) and new_body.strip() else None
        assignee_val = None
        if not getattr(task, "assignee", None):
            assignee_val = _normalize_assignee_choice(
                parsed.get("assignee"),
                default_assignee=default_assignee,
                valid_names=valid_names,
            )
        if title_val is None and body_val is None:
            return PredictedRouting(
                False, "decomposer returned fanout=false with no title/body",
            )
        return PredictedRouting(
            True, "", fanout=False, rationale=rationale,
            orchestrator=orchestrator, default_assignee=default_assignee,
            roster=roster, title=title_val, body=body_val, assignee=assignee_val,
        )

    raw_tasks = parsed.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return PredictedRouting(
            False, "decomposer returned fanout=true with empty tasks list",
        )

    # Rewrite invalid assignees to the default fallback. Never leave a
    # task with assignee=None — the user explicitly does not want that.
    children: list[dict] = []
    for idx, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict):
            return PredictedRouting(False, f"tasks[{idx}] is not an object")
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            return PredictedRouting(False, f"tasks[{idx}].title is missing or empty")
        body = entry.get("body")
        if not isinstance(body, str):
            body = ""
        assignee = entry.get("assignee")
        chosen = _normalize_assignee_choice(
            assignee,
            default_assignee=default_assignee,
            valid_names=valid_names,
        )
        if (
            isinstance(assignee, str)
            and assignee.strip()
            and assignee.strip() not in valid_names
        ):
            logger.info(
                "decompose: task %s child %d picked unknown assignee %r — "
                "routing to default_assignee %r",
                task_id_for_log, idx, assignee, default_assignee,
            )
        parents = entry.get("parents") or []
        if not isinstance(parents, list):
            parents = []
        # Clean parent indices: drop non-int and out-of-range.
        clean_parents = [p for p in parents if isinstance(p, int) and 0 <= p < len(raw_tasks) and p != idx]
        children.append({
            "title": title.strip()[:200],
            "body": body.strip(),
            "assignee": chosen,
            "parents": clean_parents,
        })

    return PredictedRouting(
        True, "", fanout=True, rationale=rationale,
        orchestrator=orchestrator, default_assignee=default_assignee,
        roster=roster, children=children,
    )


def decompose_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
) -> DecomposeOutcome:
    """Decompose a triage task into a graph of child tasks.

    Returns an outcome describing what happened. Never raises for
    expected failure modes (task not in triage, no aux client
    configured, API error, malformed response, decomposer returned
    fanout=true with empty task list) — those surface via ``ok=False``.
    """
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    if task is None:
        return DecomposeOutcome(task_id, False, "unknown task id")
    if task.status != "triage":
        return DecomposeOutcome(
            task_id, False, f"task is not in triage (status={task.status!r})"
        )

    cfg = _load_config()
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    auto_promote = bool(kanban_cfg.get("auto_promote_children", True))
    audit_author = author or _profile_author()

    prediction = _predict_routing(task, task_id_for_log=task_id, timeout=timeout)
    if not prediction.ok:
        return DecomposeOutcome(task_id, False, prediction.reason)

    if not prediction.fanout:
        with kb.connect_closing() as conn:
            ok = kb.specify_triage_task(
                conn,
                task_id,
                title=prediction.title,
                body=prediction.body,
                assignee=prediction.assignee,
                author=audit_author,
            )
        if not ok:
            return DecomposeOutcome(
                task_id, False, "task moved out of triage before promotion",
            )
        return DecomposeOutcome(
            task_id, True, "single task (no fanout)",
            fanout=False, new_title=prediction.title,
        )

    try:
        with kb.connect_closing() as conn:
            child_ids = kb.decompose_triage_task(
                conn,
                task_id,
                root_assignee=prediction.orchestrator,
                children=prediction.children,
                author=audit_author,
                auto_promote=auto_promote,
            )
    except ValueError as exc:
        return DecomposeOutcome(task_id, False, f"DB rejected graph: {exc}")
    except Exception as exc:
        logger.exception("decompose: DB error on task %s", task_id)
        return DecomposeOutcome(task_id, False, f"DB error: {type(exc).__name__}")

    if child_ids is None:
        return DecomposeOutcome(
            task_id, False, "task moved out of triage before decomposition",
        )

    return DecomposeOutcome(
        task_id, True, f"decomposed into {len(child_ids)} children",
        fanout=True, child_ids=child_ids,
    )


@dataclass
class DryRunResult:
    """Predicted routing decision, never persisted.

    Mirrors what a live call to ``decompose_task`` would decide and
    (for an existing ``task_id``) what a dispatched worker would see as
    its ``worker_context``, but stops before the two mutating DB calls
    (``kb.specify_triage_task`` / ``kb.decompose_triage_task``) that
    ``decompose_task`` uses to persist the decision. No row is ever
    created, so the dispatcher never has anything to spawn a worker for.
    """

    ok: bool
    reason: str = ""
    predicted_owner: Optional[str] = None
    context_envelope: Optional[dict] = None
    dependency_graph: Optional[list[dict]] = None
    rationale: str = ""
    fanout: bool = False


def dry_run_route(
    *,
    task_id: Optional[str] = None,
    title: Optional[str] = None,
    body: Optional[str] = None,
    timeout: Optional[int] = None,
) -> DryRunResult:
    """Preview the routing decision for a task without mutating anything.

    Exactly one of ``task_id`` (an existing row, any status, read-only
    lookup) or ``title`` (with optional ``body``, no backing row at all)
    must be given.

    Shares steps 1-4 of the live pipeline via ``_predict_routing`` so the
    prediction cannot drift from what live routing would actually decide.
    Deliberately never calls ``kb.specify_triage_task``,
    ``kb.decompose_triage_task``, or any other write helper — those two
    functions are the *only* mutating calls in the live path (see
    ``docs/rfcs/yout-plus-dry-run-routing.md``), and this function does
    not reference either name.
    """
    if (task_id is None) == (not title):
        return DryRunResult(False, "exactly one of task_id or title must be given")

    if task_id is not None:
        with kb.connect_closing() as conn:
            task = kb.get_task(conn, task_id)
        if task is None:
            return DryRunResult(False, "unknown task id")
        log_id = task_id
    else:
        task = _AdHocTask(title=title or "", body=body or "")
        log_id = "(preview)"

    prediction = _predict_routing(task, task_id_for_log=log_id, timeout=timeout)
    if not prediction.ok:
        return DryRunResult(False, prediction.reason)

    if not prediction.fanout:
        owner = prediction.assignee or getattr(task, "assignee", None) or prediction.default_assignee
        envelope = {
            "title": prediction.title or task.title,
            "body": prediction.body or task.body,
            "assignee": owner,
            "parent_handoffs": [],
            "roster": prediction.roster,
            "orchestrator": prediction.orchestrator,
            "default_assignee": prediction.default_assignee,
        }
        if task_id is not None:
            with kb.connect_closing() as conn:
                envelope["worker_context"] = kb.build_worker_context(conn, task_id)
        return DryRunResult(
            True, predicted_owner=owner, context_envelope=envelope,
            dependency_graph=None, rationale=prediction.rationale, fanout=False,
        )

    dependency_graph = [
        {"index": idx, **child}
        for idx, child in enumerate(prediction.children or [])
    ]
    envelope = {
        "title": task.title,
        "body": task.body,
        "assignee": prediction.orchestrator,
        "parent_handoffs": [],
        "roster": prediction.roster,
        "orchestrator": prediction.orchestrator,
        "default_assignee": prediction.default_assignee,
    }
    if task_id is not None:
        with kb.connect_closing() as conn:
            envelope["worker_context"] = kb.build_worker_context(conn, task_id)
    return DryRunResult(
        True, predicted_owner=prediction.orchestrator, context_envelope=envelope,
        dependency_graph=dependency_graph, rationale=prediction.rationale, fanout=True,
    )


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Return task ids currently in the triage column."""
    with kb.connect_closing() as conn:
        rows = kb.list_tasks(
            conn,
            status="triage",
            tenant=tenant,
            limit=1000,
        )
    return [row.id for row in rows]
