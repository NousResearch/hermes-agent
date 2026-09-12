"""Kanban decomposer — fan a triage task out into a graph of child tasks.

Invoked by ``hermes kanban decompose [task_id | --all]`` and the gateway
dispatcher's auto-decompose path. Reads the profile roster (with
descriptions), asks the auxiliary LLM for a task graph in JSON, then
atomically creates the children, links them under the root, and flips the
root ``triage -> todo``. The root stays alive as parent of every leaf child so
it wakes back up when the graph completes and its assignee (the orchestrator
profile) can judge completion and add more work.

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
    # FLEET: --dry-run computes the graph and returns it here, writing nothing.
    dry_run_plan: Optional[list[dict]] = None


def _profile_author() -> str:
    """Mirror of ``hermes_cli.kanban._profile_author``."""
    return _specify_author("decomposer")


def _load_config() -> dict:
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception:
        return {}


def _resolve_profile_from_cfg(cfg: dict, key: str) -> str:
    """``kanban.<key>`` if it names an existing profile, else the active
    default profile — so a task is never stranded for lack of an owner.
    ``orchestrator_profile`` owns the root after fan-out; ``default_assignee``
    catches children the decomposer can't route."""
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    explicit = (kanban_cfg.get(key) or "").strip()
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


# ===========================================================================
# FLEET ADDITIONS (WeRoll) — decision/deploy shaping and the review-cycle
# guard. No upstream equivalent.
# ===========================================================================
# Titles whose completion would constitute an unsigned product/PM decision.
# The auto-decomposer routes children whose title matches this to ``triage``
# (never ``ready``), so a ghost PM-run cannot self-complete them and lock a
# decision the owner never signed.
#
# The verbs are chosen so only deliberately decision-shaped phrasings match, not
# incidental uses inside implementation titles. Bare ``\b`` word boundaries are
# NOT sufficient — ``\b lock \b`` matches the standalone word "lock", so title
# like "Lock the report row rendering" and "Sign off on the backend" would be
# misclassified. Ambiguous verbs that appear routinely in implementation titles
# are therefore dropped outright (lock, sign off, redefine) or gated on a
# decision-shaped noun context: "approve" matches only before a design/plan/
# approach/... noun, and "amend" only before a document (PRD/plan/spec/design).
# Unambiguous decision verbs that only ever read as a decision when used bare
# (decide, ratify) and the verb phrase "spec the" match directly.
_DECISION_TITLE_RE = re.compile(
    r"\bdecide\b"
    r"|\bapprove\s+(?:the|an?|this)\s+"
    r"(?:(?:[a-z0-9-]+\s+)?"
    r"(?:design|plan|approach|choice|spec|decision|model|schema|architecture|"
    r"strategy|option|direction|source|interface)\b)"
    r"|\bspec\s+the\b"
    r"|\bratify\b"
    r"|\bamend\s+the\s+(?:prd|plan|spec|design(?:\s+doc)?|roadmap|requirements)\b",
    re.IGNORECASE,
)

# Exactly-matched author that stamps auto-decompose children (see
# kanban_db.decompose_triage_task's ``created_by``). Used both to (a) route
# decision-shaped children to triage and (b) exclude auto-decomposer-created
# triage from re-decomposition on later ticks.
AUTO_DECOMPOSER_AUTHOR = "auto-decomposer"

# Terminal children that, if they auto-promote, ship code to production or
# release to a human without an owner sign-off. Their titles read as deploy /
# release / rollout / ship-to-prod. The auto-decomposer holds these at
# creation (`operator_hold`) so they can never slip a deployment past an
# approval gate (2026-09-04 GlobalAside: the deploy card had a prose "HELD"
# but no DB hold, so it auto-promoted the moment its verify parent completed).
_DEPLOY_TITLE_RE = re.compile(
    r"\bdeploy(?:ment)?\b"
    r"|\brelease\b"
    r"|\brollout\b"
    r"|\bship-to?-prod(?:uction)?\b"
    r"|\bgo live\b"
    r"|\bput .* into production\b",
    re.IGNORECASE,
)


def _is_decision_shaped(title: str) -> bool:
    """True when a child title reads as a product decision, not a task."""
    if not title:
        return False
    return bool(_DECISION_TITLE_RE.search(title))


def _is_deploy_shaped(title: str) -> bool:
    """True when a child title names a deploy/release/rollout to production.

    Used to force a real ``operator_hold`` on auto-decomposed deploy children
    (see ``decompose_task``), so a deployment gate is machine-enforced rather
    than advisory prose.
    """
    if not title:
        return False
    return bool(_DEPLOY_TITLE_RE.search(title))


# Outcomes that are "decisive" about whether a card is currently inside an
# active review cycle. ``review_requested`` = card handed to a reviewer;
# ``changes_requested`` = reviewer sent it back for rework. ``completed`` is
# the state that CLOSES a review cycle (the card reached done). ``blocked`` /
# ``crashed`` / ``timed_out`` mid-fix are NOT decisive — they say nothing about
# which lane the card is in, so they are ignored when deciding whether a card
# is mid-review.
_REVIEW_CYCLE_OUTCOMES = frozenset({"review_requested", "changes_requested"})


_REVIEW_CYCLE_TERMINAL = frozenset({"review_requested", "changes_requested", "completed"})


def _has_active_review_cycle(task_id: str) -> bool:
    """Return True when a task is inside a review cycle and must be resumed,
    never decomposed.

    Guard 1 of the auto-decomposer fix (t_c52b9bc3): a healthy card in the
    review loop whose fix run ended ``blocked`` must NOT be split into
    children. Look at the NEWEST decisive run outcome — a ``changes_requested``
    (or ``review_requested``) that has not since been closed by a
    ``completed`` means the card is mid-review. A trailing ``blocked`` run does
    not clear it (the blocked fix-round is the very case we want to resume, not
    fan out).

    Also covers the obvious fast path: the card's current status is
    ``review``/``changes_requested``.
    """
    try:
        with kbc.connect_closing() as conn:
            task = kb.get_task(conn, task_id)
            if task is None:
                return False
            if task.status in ("review", "changes_requested"):
                return True
            row = conn.execute(
                "SELECT outcome FROM task_runs "
                "WHERE task_id = ? AND outcome IS NOT NULL "
                "ORDER BY id DESC LIMIT 500",
                (task_id,),
            ).fetchall()
    except Exception:
        logger.warning(
            "decompose: could not check review cycle for %s (assuming none)", task_id,
        )
        return False
    decisive = [r["outcome"] for r in row if r["outcome"] in _REVIEW_CYCLE_TERMINAL]
    if not decisive:
        return False
    return decisive[0] in _REVIEW_CYCLE_OUTCOMES


@dataclass
class _Routing:
    """Config-derived routing context for one decomposition."""

    orchestrator: str
    default_assignee: str
    auto_promote: bool
    roster: list[dict]
    valid_names: set[str]


def _load_routing() -> _Routing:
    cfg = _load_config()
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    roster, valid_names = _build_roster()
    return _Routing(
        orchestrator=_resolve_profile_from_cfg(cfg, "orchestrator_profile"),
        default_assignee=_resolve_profile_from_cfg(cfg, "default_assignee"),
        auto_promote=bool(kanban_cfg.get("auto_promote_children", True)),
        roster=roster,
        valid_names=valid_names,
    )


def _apply_single(task: kb.Task, parsed: dict, routing: _Routing, author: str,
                  *, dry_run: bool = False) -> DecomposeOutcome:
    """``fanout=false``: single-task spec promotion (same effect as specify)."""
    title_val, body_val = _title_body(parsed)
    assignee_val = None
    if not task.assignee:
        assignee_val = _normalize_assignee_choice(
            parsed.get("assignee"), default_assignee=routing.default_assignee, valid_names=routing.valid_names,
        )
    if title_val is None and body_val is None:
        return DecomposeOutcome(task.id, False, "decomposer returned fanout=false with no title/body")
    if dry_run:
        return DecomposeOutcome(
            task.id, True, "dry-run: single task (no fanout) — nothing written",
            fanout=False, new_title=title_val,
            dry_run_plan=[{"title": title_val, "body": body_val, "assignee": assignee_val}],
        )
    with kbc.connect_closing() as conn:
        ok = kb.specify_triage_task(
            conn, task.id, title=title_val, body=body_val, assignee=assignee_val, author=author,
        )
    if not ok:
        return DecomposeOutcome(task.id, False, "task moved out of triage before promotion")
    return DecomposeOutcome(task.id, True, "single task (no fanout)", fanout=False, new_title=title_val)


def _clean_children(task_id: str, raw_tasks: list, routing: _Routing,
                    *, is_auto: bool = False) -> tuple[list[dict], str]:
    """Validate/normalise the LLM's ``tasks`` list; ``(children, "")`` or ``([], reason)``.
    Unknown assignees route to the default; never assignee=None."""
    children: list[dict] = []
    for idx, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict):
            return [], f"tasks[{idx}] is not an object"
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            return [], f"tasks[{idx}].title is missing or empty"
        body = entry.get("body")
        assignee = entry.get("assignee")
        chosen = _normalize_assignee_choice(
            assignee, default_assignee=routing.default_assignee, valid_names=routing.valid_names,
        )
# Deterministic role-map (platform fix-C1, t_bc08efc3): a LANE-shaped
        # child title (Implement/Build/FIX -> build; Review/Verify/... ) is
        # routed by the role-map, NEVER by the LLM's free-text roster pick — so
        # an ``Implement X`` child cannot land on rodge/steve-o. The role-map
        # profile wins as long as it is an installed profile; otherwise the
        # existing null/unknown -> default normalization stands.
        from hermes_cli.kanban_role_map import canonical_assignee_for_title

        _lane_profile = canonical_assignee_for_title(title, valid_names=valid_names)
        if _lane_profile is not None:
            chosen = _lane_profile
        if (
            isinstance(assignee, str)
            and assignee.strip()
            and assignee.strip() not in valid_names
        ):
            logger.info(
                "decompose: task %s child %d picked unknown assignee %r — "
                "routing to default_assignee %r",
                task_id, idx, assignee, routing.default_assignee,
            )
        parents = entry.get("parents") or []
        if not isinstance(parents, list):
            parents = []
        # FLEET: a decision-shaped child of an AUTO run parks in ``triage``
        # (never ``ready``) so a ghost PM run cannot self-complete it and lock a
        # decision the owner never signed. A manual run is already
        # owner-committed and keeps today's behaviour.
        #
        # Deploy/terminal children are the mirror case: anything whose
        # completion ships to production must be operator-HELD at creation. A
        # prose "HELD pending approval" in the body is not a gate —
        # recompute_ready auto-promotes such a card the moment its parent
        # completes (the 2026-09-04 GlobalAside incident). Over-holding is safe;
        # under-holding is the bug being closed.
        hold_flag = bool(entry.get("hold"))
        if is_auto and _is_deploy_shaped(title):
            hold_flag = True
        children.append({
            "title": title.strip()[:200],
            "body": body.strip() if isinstance(body, str) else "",
            "assignee": chosen,
            # Drop non-int, out-of-range and self parent indices.
            "parents": [p for p in parents if isinstance(p, int) and 0 <= p < len(raw_tasks) and p != idx],
            "triage": is_auto and _is_decision_shaped(title),
            "hold": hold_flag,
        })
    return children, ""


def _apply_fanout(task_id: str, parsed: dict, routing: _Routing, author: str,
                  *, dry_run: bool = False) -> DecomposeOutcome:
    raw_tasks = parsed.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return DecomposeOutcome(task_id, False, "decomposer returned fanout=true with empty tasks list")
    children, reason = _clean_children(
        task_id, raw_tasks, routing, is_auto=(author == AUTO_DECOMPOSER_AUTHOR),
    )
    if reason:
        return DecomposeOutcome(task_id, False, reason)
    if dry_run:
        return DecomposeOutcome(
            task_id, True,
            f"dry-run: would fan out into {len(children)} children — nothing written",
            fanout=True, child_ids=None,
            dry_run_plan=[
                {k: c[k] for k in ("title", "body", "assignee", "parents", "triage", "hold")}
                for c in children
            ],
        )
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
    except ValueError as exc:
        return DecomposeOutcome(task_id, False, f"DB rejected graph: {exc}")
    except Exception as exc:
        logger.exception("decompose: DB error on task %s", task_id)
        return DecomposeOutcome(task_id, False, f"DB error: {type(exc).__name__}")
    if child_ids is None:
        return DecomposeOutcome(task_id, False, "task already decomposed or moved out of triage")
    return DecomposeOutcome(
        task_id, True, f"decomposed into {len(child_ids)} children", fanout=True, child_ids=child_ids,
    )


def decompose_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
    dry_run: bool = False,
) -> DecomposeOutcome:
    """Decompose a triage task into a graph of child tasks. Expected failures
    (not in triage, no aux client, API error, malformed/empty reply) surface
    as ``ok=False``."""
    task, reason = _load_triage_task(task_id)
    if task is None:
        # FLEET: a BLOCKED root is a resume fan-out — an operator splitting a
        # stuck card into children. ``_load_triage_task`` only admits ``triage``,
        # so re-read and accept that one extra status here.
        with kbc.connect_closing() as conn:
            task = kb.get_task(conn, task_id)
        if task is None or task.status != "blocked":
            return DecomposeOutcome(task_id, False, reason)
    # FLEET guard (t_c52b9bc3): NO SPLIT MID-REVIEW. A card in an active review
    # cycle must be RESUMED in the same card and worktree, not fanned out — a
    # blocked fix-round is a resume trigger, never a fan-out trigger. Refusing
    # here rather than in the dispatcher tick makes the guard hold for the
    # manual `kanban decompose <id>` path too.
    if _has_active_review_cycle(task_id):
        return DecomposeOutcome(
            task_id, False,
            "task is in an active review cycle; resuming in same card + "
            "worktree, refusing to split (blocked fix-round is a resume, not a "
            "fan-out trigger)",
        )

    routing = _load_routing()
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
    if not parsed.get("fanout"):
        return _apply_single(task, parsed, routing, audit_author, dry_run=dry_run)
    return _apply_fanout(task_id, parsed, routing, audit_author, dry_run=dry_run)


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Return task ids currently in the triage column.

    FLEET: excludes cards the auto-decomposer itself parked in ``triage`` (the
    decision-shaped children it demoted for the PM). Without the exclusion the
    dispatcher's tick would re-decompose those parked decisions next tick,
    defeating the AC1 gate. Also excludes a card the loop breaker routed to
    triage (same-kind re-block at the recurrence limit): that is parked for a
    HUMAN, and auto-decomposing it would hand the escalation ceiling to the
    decomposer model. Genuine user-dropped triage is still returned.
    """
    with kbc.connect_closing() as conn:
        rows = kb.list_tasks(conn, status="triage", tenant=tenant, limit=1000)
    return [
        row.id for row in rows
        if (row.created_by or "") != AUTO_DECOMPOSER_AUTHOR
        # "decomposer" is the legacy fallback author decompose_triage_task
        # stamps when no author is supplied; filter it the same way rather than
        # stranding such a card forever.
        and (row.created_by or "") != "decomposer"
        and not (
            (row.block_kind or "") != ""
            and int(row.block_recurrences or 0) >= kb.BLOCK_RECURRENCE_LIMIT
        )
    ]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
