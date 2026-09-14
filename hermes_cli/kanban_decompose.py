"""Kanban decomposer — fan a triage task out into a graph of child tasks.

Invoked by ``hermes kanban decompose [task_id | --all]`` and the gateway
dispatcher's auto-decompose path. Reads the profile roster (with
descriptions), asks the auxiliary LLM for a task graph in JSON, then
atomically creates the children, links them under the root, and flips the
root ``triage -> todo``. The root stays alive as parent of every leaf child so
it wakes back up when the graph completes and its assignee (the orchestrator
profile) can judge completion and add more work.

Determinism guards (board-flow-convention: no self-reference, no duplicates,
never route write-phase work to a zero-write profile):
  - ``_validate_partitioning`` rejects a graph whose children restate the
    root's own scope (verbatim overlap or an action-verb chain split of the
    root's pipeline) instead of partitioning it — the decompose fails
    loudly and writes nothing, so the task stays visible in triage rather
    than stranding in todo.
  - ``kanban_db_graph.decompose_triage_task`` refuses to gate the root on
    its own duplicates: when every sink child belongs to the root's own
    assignee, the root is promoted straight to ``ready`` and audited with a
    DEGRADED DECOMPOSITION comment.
  - Read-only profiles (marked ``⚠ no-write`` in the roster from their own
    description) are flagged in the prompt and must never receive
    write/merge/deploy work.

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
  - NEVER assign a task to a profile marked "⚠ no-write" if the task
    requires writing, merging, deploying, or mutating production state —
    such a task can never be completed by that profile and will strand.
    Pick a write-capable profile or null instead.
  - Each child task body is what a fresh worker will read with no other
    context — be specific about goal, approach, and acceptance criteria.

Partitioning (hard constraints — a violation makes the whole decomposition
invalid, so the system will reject it):
  - CHILDREN PARTITION the original task; they never RESTATE it. A child
    whose title/body is the same job as the original (e.g. the original is
    "promote X to production" and a child is also "promote X") is FORBIDDEN.
    If the original task cannot be split into genuinely smaller units,
    return fanout=false instead.
  - Every child must be executable by its ASSIGNEE profile, judged by that
    profile's roster description. Never assign a task the profile cannot
    perform (e.g. a research/read-only profile must not receive a merge or
    deploy step). When no profile can perform a step, use null — do NOT
    force a mismatch.
  - Do not create a child that duplicates another child's job.

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


def _profile_author() -> str:
    """Mirror of ``hermes_cli.kanban._profile_author``."""
    return _specify_author("decomposer")


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


# Description fragments that mark a profile as unable to execute write-phase
# work (merge/deploy/production mutations). A decomposition must never route
# such a step to one of these — the task strands forever. Matched on the
# profile's own description so no per-install hardcoding of profile names.
_READONLY_DESCRIPTION_MARKERS: tuple[str, ...] = (
    "zero-write", "zero write", "read-only", "read only", "readonly",
    "no merge authority", "never merges", "never merge", "no deploy",
    "never deploys", "never deploy", "research lane", "read/sim",
    "never implements", "never hand-write", "scaling oracle",
)


def _roster_capability_note(description: str) -> str:
    """`` ⚠ no-write`` suffix when the description marks a read-only lane."""
    low = (description or "").lower()
    if any(marker in low for marker in _READONLY_DESCRIPTION_MARKERS):
        return " ⚠ no-write"
    return ""


def _format_roster(roster: list[dict]) -> str:
    if not roster:
        return "  (no profiles installed — decomposer cannot route work)"
    return "\n".join(
        f"  - {entry['name']}{'' if entry['has_description'] else ' ⚠ undescribed'}"
        f"{_roster_capability_note(entry['description'])}: {entry['description']}"
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


def _load_routing() -> _Routing:
    from hermes_cli.config import load_config_readonly
    try:
        cfg = load_config_readonly()
    except Exception:  # decompose_task promises ok=False, never a raise, on config trouble
        cfg = {}
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    roster, valid_names = _build_roster()
    return _Routing(
        orchestrator=_resolve_profile_from_cfg(cfg, "orchestrator_profile"),
        default_assignee=_resolve_profile_from_cfg(cfg, "default_assignee"),
        auto_promote=bool(kanban_cfg.get("auto_promote_children", True)),
        roster=roster,
        valid_names=valid_names,
    )


def _apply_single(task: kb.Task, parsed: dict, routing: _Routing, author: str) -> DecomposeOutcome:
    """``fanout=false``: single-task spec promotion (same effect as specify)."""
    title_val, body_val = _title_body(parsed)
    assignee_val = None
    if not task.assignee:
        assignee_val = _normalize_assignee_choice(
            parsed.get("assignee"), default_assignee=routing.default_assignee, valid_names=routing.valid_names,
        )
    if title_val is None and body_val is None:
        return DecomposeOutcome(task.id, False, "decomposer returned fanout=false with no title/body")
    with kbc.connect_closing() as conn:
        ok = kb.specify_triage_task(
            conn, task.id, title=title_val, body=body_val, assignee=assignee_val, author=author,
        )
    if not ok:
        return DecomposeOutcome(task.id, False, "task moved out of triage before promotion")
    return DecomposeOutcome(task.id, True, "single task (no fanout)", fanout=False, new_title=title_val)


def _clean_children(task_id: str, raw_tasks: list, routing: _Routing) -> tuple[list[dict], str]:
    """Validate/normalise the LLM's ``tasks`` list; ``(children, "")`` or ``([], reason)``.
    Unknown assignees route to the default; never assignee=None. Enforces the
    partitioning antipatterns the board-flow convention forbids: a child
    duplicating the parent's job, and (in ``routing``) leaves that would gate
    the root on its own duplicates."""
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
        if isinstance(assignee, str) and assignee.strip() and assignee.strip() not in routing.valid_names:
            logger.info(
                "decompose: task %s child %d picked unknown assignee %r — "
                "routing to default_assignee %r",
                task_id, idx, assignee, routing.default_assignee,
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
    reason = _validate_partitioning(task_id, children, routing)
    if reason:
        return [], reason
    return children, ""


_STOPWORDS = frozenset(
    "a an and as at for from in into of on onto or the to with".split()
)

# Common imperative action verbs: a root title that is itself a list of these
# (e.g. "merge dev into main and deploy") whose children merely split the same
# verbs into a chain is the parent's pipeline restated, not a decomposition.
_ACTION_VERBS = frozenset(
    "merge deploy ship release promote verify validate test build run install"
    " review approve implement write code refactor migrate restart roll rollback"
    " backfill publish tag push pull rebase sync commit upload download"
    .split()
)


def _job_tokens(text: str) -> set[str]:
    """Content-word set of a title/body, lowercased, stopwords dropped —
    cheap lexical fingerprint for duplicate-job detection."""
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _STOPWORDS}


def _validate_partitioning(task_id: str, children: list[dict], routing: _Routing) -> str:
    """Return a rejection reason when the proposed graph restates the root's
    job instead of partitioning it. Two checks, both cheap (no LLM round-trip):

    1. VERBATIM duplicate: a child whose title fingerprint is ≥ 70%
       token-overlap with the root title is the parent's job restated
       (the t_72cf3f84 shape: a "promote #260" root decomposing into
       another "promote #260" child).
    2. ACTION-SEQUENCE violation: a root whose title is itself a sequence
       of action verbs ("merge + deploy", "verify and promote") cannot be
       decomposed into a chain of children where each step re-uses one of
       the root's own action verbs — that is the root's pipeline restated
       one step per child, with the root left waiting on its own steps.
       Each child then only names ONE action; if EVERY child's action is
       drawn from the root's title and no child introduces a genuinely
       different deliverable, the "decomposition" is a fan-out of the
       parent's own verb list and is rejected (the root should have stayed
       one task or been handed to a single owner).
    """
    root_task, _ = _load_triage_task(task_id)
    root_title = (root_task.title or "") if root_task is not None else ""
    root_tokens = _job_tokens(root_title)
    if root_tokens:
        for idx, child in enumerate(children):
            # Title overlap is the duplicate signal; body words are ignored
            # (a child body legitimately restates its spec).
            title_overlap = _job_tokens(child["title"]) & root_tokens
            if title_overlap and len(title_overlap) / max(1, len(root_tokens)) >= 0.7:
                logger.warning(
                    "decompose: task %s child %d (%r) duplicates the root scope "
                    "(%d/%d root title tokens) — rejecting decomposition",
                    task_id, idx, child["title"], len(title_overlap), len(root_tokens),
                )
                return (
                    f"tasks[{idx}].title duplicates the root task's scope "
                    f"({sorted(title_overlap)}); children must PARTITION the "
                    "parent's job, not restate it — return fanout=false if the "
                    "task cannot be split"
                )
        # Action-sequence check: the root title lists multiple actions and the
        # children are just those actions split into a chain (every child's
        # distinguishing action token already appears in the root title).
        _ACTION_STOP = frozenset(
            "pr review gate test tests check verify approval criteria evidence"
            " record decision report production environment main dev fix"
            " readiness latest per usual anything anything stop instead"
            " step chain stage stages fix".split()
        )
        action_tokens = {
            w for w in root_tokens - _ACTION_STOP
            if w in _ACTION_VERBS
        } - {"promote"}  # "promote" is the root's framing verb, not a child step
        if len(action_tokens) >= 2:
            covered = set()
            for child in children:
                child_tokens = _job_tokens(child["title"]) - _ACTION_STOP
                overlap_actions = child_tokens & action_tokens
                if overlap_actions and len(child_tokens) <= len(overlap_actions) + 3:
                    covered |= overlap_actions
            if covered and covered == action_tokens:
                logger.warning(
                    "decompose: task %s children restate the root's own action "
                    "sequence %s one step per child — rejecting decomposition",
                    task_id, sorted(action_tokens),
                )
                return (
                    f"children restate the root's own action sequence "
                    f"({sorted(action_tokens)}) one step per child; the root's "
                    "pipeline is not a decomposition — return fanout=false and "
                    "let the root execute (or pick a single owner) instead"
                )
    return ""


def _apply_fanout(task_id: str, parsed: dict, routing: _Routing, author: str) -> DecomposeOutcome:
    raw_tasks = parsed.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return DecomposeOutcome(task_id, False, "decomposer returned fanout=true with empty tasks list")
    children, reason = _clean_children(task_id, raw_tasks, routing)
    if reason:
        return DecomposeOutcome(task_id, False, reason)
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
) -> DecomposeOutcome:
    """Decompose a triage task into a graph of child tasks. Expected failures
    (not in triage, no aux client, API error, malformed/empty reply) surface
    as ``ok=False``."""
    task, reason = _load_triage_task(task_id)
    if task is None:
        return DecomposeOutcome(task_id, False, reason)

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
        return _apply_single(task, parsed, routing, audit_author)
    return _apply_fanout(task_id, parsed, routing, audit_author)


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Return task ids currently in the triage column."""
    with kbc.connect_closing() as conn:
        rows = kb.list_tasks(conn, status="triage", tenant=tenant, limit=1000)
    return [row.id for row in rows]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
