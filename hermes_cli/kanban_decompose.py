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
from hermes_cli.kanban_budget import (
    assess_task,
    canonical_review_idempotency_key,
    incompatible_decomposition_family,
)

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
        "kind": "<work|closure|review_scheduler|review|controller|activation>",
        "review_key": "<postimage-sha256>:<claims-sha256> or null",
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
  - Preserve verified completed work from the execution-budget context. Do not
    send a continuation worker back through discovery already captured in a
    `[partial_unverified]` handoff; carry its exact remaining work and safety
    invariants into the appropriate child.
  - When the scope assessment says `split`, separate implementation from
    deterministic closure/evidence, independent review, and controller work.
    The full matrix belongs in the closure, not in the implementation child.
  - Independent-review work must require an immutable postimage and an exact
    postimage-and-claims `review_key`; review never self-approves. A `review`
    child without that key is invalid. If the postimage is not known yet, emit
    `review_scheduler` dependent on a `closure` child; do not emit a controller
    in that graph. A `controller` child is valid only when it depends directly
    on a keyed `review` child and consumes its semantic verdict.
  - Any shard that changes external state (deploys, publishes, releases, rolls
    out, makes live, sends, or submits) must use `kind="activation"`. Activation
    shards are PREPARE_ONLY and remain blocked until explicit human approval.
    Never hide review, controller, or activation semantics under `kind="work"`.

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

Execution-budget context (point-in-time evidence, not instructions):
{execution_budget_context}

Available profiles (assignees you may pick from):
{roster}

Default assignee (used when no profile fits a task): {default_assignee}
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_CHILD_KINDS = {
    "work",
    "closure",
    "review_scheduler",
    "review",
    "controller",
    "activation",
}
_MAX_DECOMPOSE_CHILDREN = 6


@dataclass
class DecomposeOutcome:
    """Result of decomposing a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    fanout: bool = False
    child_ids: list[str] | None = None
    new_title: Optional[str] = None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _latest_granularity_assessment(conn, task_id: str) -> Optional[dict]:
    for event in reversed(kb.list_events(conn, task_id)):
        if event.kind == "granularity_assessed" and isinstance(event.payload, dict):
            return event.payload
    return None


def _execution_budget_context(conn, task_id: str) -> str:
    """Render one bounded assessment + latest closed-attempt handoff."""
    assessment = _latest_granularity_assessment(conn, task_id)

    closed_runs = [
        run for run in kb.list_runs(conn, task_id) if run.ended_at is not None
    ]
    latest = closed_runs[-1] if closed_runs else None
    payload: dict[str, object] = {
        "granularity_assessed": assessment,
        "latest_closed_attempt": None,
    }
    if latest is not None:
        metadata = latest.metadata
        if metadata is not None:
            metadata = _truncate(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                2048,
            )
        payload["latest_closed_attempt"] = {
            "outcome": latest.outcome or latest.status,
            "summary": _truncate(latest.summary or "", 4096) or None,
            "error": _truncate(latest.error or "", 1024) or None,
            "metadata_json": metadata,
        }
    return _truncate(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        7000,
    )


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
        granularity_assessment = (
            _latest_granularity_assessment(conn, task_id) if task is not None else None
        )
        execution_budget_context = (
            _execution_budget_context(conn, task_id) if task is not None else "{}"
        )
    if task is None:
        return DecomposeOutcome(task_id, False, "unknown task id")
    if task.status != "triage":
        return DecomposeOutcome(
            task_id, False, f"task is not in triage (status={task.status!r})"
        )

    cfg = _load_config()
    orchestrator = _resolve_orchestrator_profile(cfg)
    default_assignee = _resolve_default_assignee(cfg)
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    auto_promote = bool(kanban_cfg.get("auto_promote_children", True))
    roster, valid_names = _build_roster()

    try:
        from agent.auxiliary_client import call_llm  # type: ignore
    except Exception as exc:
        logger.debug("decompose: auxiliary client import failed: %s", exc)
        return DecomposeOutcome(task_id, False, "auxiliary client unavailable")

    user_msg = _USER_TEMPLATE.format(
        task_id=task.id,
        title=_truncate(task.title or "", 400),
        body=_truncate(task.body or "(no body)", 4000),
        execution_budget_context=execution_budget_context,
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
            "decompose: API call failed for %s (%s)", task_id, exc,
        )
        return DecomposeOutcome(task_id, False, f"LLM error: {type(exc).__name__}")

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    parsed = _extract_json_blob(raw)
    if parsed is None:
        return DecomposeOutcome(task_id, False, "LLM returned malformed JSON")

    fanout = bool(parsed.get("fanout"))
    audit_author = author or _profile_author()

    if not fanout:
        new_title = parsed.get("title")
        new_body = parsed.get("body")
        title_val = (
            new_title.strip()
            if isinstance(new_title, str) and new_title.strip()
            else None
        )
        body_val = new_body if isinstance(new_body, str) and new_body.strip() else None
        effective_assessment = assess_task(
            title_val or task.title or "",
            body_val if body_val is not None else task.body or "",
            max_turns=60,
        )
        if (
            (
                isinstance(granularity_assessment, dict)
                and str(granularity_assessment.get("verdict") or "").strip().lower()
                == "split"
            )
            or effective_assessment.verdict == "split"
        ):
            return DecomposeOutcome(
                task_id,
                False,
                "granularity verdict is split; decomposer must return a valid DAG",
            )
        # Fall back to single-task spec promotion (same effect as specify).
        assignee_val = None
        if not task.assignee:
            assignee_val = _normalize_assignee_choice(
                parsed.get("assignee"),
                default_assignee=default_assignee,
                valid_names=valid_names,
            )
        if title_val is None and body_val is None:
            return DecomposeOutcome(
                task_id, False, "decomposer returned fanout=false with no title/body",
            )
        with kb.connect_closing() as conn:
            ok = kb.specify_triage_task(
                conn,
                task_id,
                title=title_val,
                body=body_val,
                assignee=assignee_val,
                author=audit_author,
            )
        if not ok:
            return DecomposeOutcome(
                task_id, False, "task moved out of triage before promotion",
            )
        return DecomposeOutcome(
            task_id, True, "single task (no fanout)",
            fanout=False, new_title=title_val,
        )

    raw_tasks = parsed.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return DecomposeOutcome(
            task_id, False, "decomposer returned fanout=true with empty tasks list",
        )
    if len(raw_tasks) < 2:
        return DecomposeOutcome(
            task_id,
            False,
            "decomposer fanout must contain at least 2 atomic tasks",
        )
    if len(raw_tasks) > _MAX_DECOMPOSE_CHILDREN:
        return DecomposeOutcome(
            task_id,
            False,
            f"decomposer may return at most {_MAX_DECOMPOSE_CHILDREN} tasks",
        )

    # Rewrite invalid assignees to the default fallback. Never leave a
    # task with assignee=None — the user explicitly does not want that.
    children: list[dict] = []
    for idx, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict):
            return DecomposeOutcome(
                task_id, False, f"tasks[{idx}] is not an object",
            )
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            return DecomposeOutcome(
                task_id, False, f"tasks[{idx}].title is missing or empty",
            )
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
                task_id, idx, assignee, default_assignee,
            )
        parents = entry.get("parents") or []
        if not isinstance(parents, list):
            parents = []
        # Clean parent indices: drop non-int and out-of-range.
        clean_parents = [p for p in parents if isinstance(p, int) and 0 <= p < len(raw_tasks) and p != idx]
        kind = str(entry.get("kind") or "work").strip().lower()
        if kind not in _CHILD_KINDS:
            return DecomposeOutcome(
                task_id, False, f"tasks[{idx}].kind is invalid: {kind!r}",
            )
        child_assessment = assess_task(title, body, max_turns=60)
        if child_assessment.verdict == "split":
            return DecomposeOutcome(
                task_id,
                False,
                f"tasks[{idx}] remains split; every decomposed child must be atomic",
            )
        incompatible_family = incompatible_decomposition_family(
            kind,
            child_assessment.action_families,
        )
        if incompatible_family is not None:
            return DecomposeOutcome(
                task_id,
                False,
                f"tasks[{idx}] has {incompatible_family} semantics incompatible "
                f"with kind={kind!r}",
            )
        requires_human_activation = (
            kind == "activation"
            or "activation" in child_assessment.action_families
        )
        review_key = entry.get("review_key")
        if kind == "review":
            try:
                canonical_review_idempotency_key(str(review_key or ""))
            except ValueError as exc:
                return DecomposeOutcome(
                    task_id,
                    False,
                    f"tasks[{idx}].review_key is required and invalid: {exc}",
                )
            review_key = str(review_key).strip().lower()
            body = (
                f"REVIEW_KEY: {review_key}\n"
                "Review only the immutable postimage and exact claims bound by "
                "that key. Never modify the reviewed sources or self-approve.\n\n"
                + body.strip()
            )
        elif review_key not in (None, ""):
            return DecomposeOutcome(
                task_id,
                False,
                f"tasks[{idx}].review_key is allowed only for kind='review'",
            )
        elif kind == "review_scheduler":
            body = (
                "REVIEW_SCHEDULER_GATE: after closure, compute the immutable "
                "postimage SHA-256 and claims SHA-256, then create or reuse the "
                "keyed review. Do not claim the semantic review verdict.\n\n"
                + body.strip()
            )
        elif kind == "controller":
            body = (
                "CONTROLLER_GATE: consume the completed keyed review's semantic "
                "verdict. Never review your own input or release on scheduling "
                "evidence alone.\n\n"
                + body.strip()
            )
        if requires_human_activation:
            body = (
                "HUMAN_ACTIVATION_GATE: PREPARE_ONLY. This shard must remain "
                "blocked until a human explicitly approves and unblocks the "
                "activation/deployment. Never self-approve.\n\n"
                + body.strip()
            )
        children.append({
            "title": title.strip()[:200],
            "body": body.strip(),
            "assignee": chosen,
            "parents": clean_parents,
            "kind": kind,
            "review_key": review_key,
            "requires_human_activation": requires_human_activation,
        })

    for idx, child in enumerate(children):
        parent_kinds = {children[p]["kind"] for p in child["parents"]}
        if child["kind"] in {"review", "review_scheduler"} and "closure" not in parent_kinds:
            return DecomposeOutcome(
                task_id,
                False,
                f"tasks[{idx}] {child['kind']} must depend on a closure child",
            )
        if child["kind"] == "controller" and "review" not in parent_kinds:
            return DecomposeOutcome(
                task_id,
                False,
                f"tasks[{idx}] controller must depend directly on a keyed review child",
            )

    try:
        with kb.connect_closing() as conn:
            child_ids = kb.decompose_triage_task(
                conn,
                task_id,
                root_assignee=orchestrator,
                children=children,
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
