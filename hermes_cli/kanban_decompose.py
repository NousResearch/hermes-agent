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

A proposed child that re-does the parent's own scope, or that lands on a file an
earlier sibling already edits, is dropped before creation (``_drop_duplicates``);
if every child is dropped the card stays in triage for a human. Without that
guard one parked defect card (t_2e0cd7e4) decomposed into three concurrent edits
of the same file, so the board manufactures duplicate work faster than anyone can
review it.
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


_DUPLICATE_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into", "after",
    "before", "when", "then", "than", "also", "make", "adds", "new",
})
#: Sibling titles at or above this Jaccard overlap are treated as the same work.
_SIBLING_OVERLAP = 0.75
#: Two children naming the same file are duplicates at this lower overlap.
_SAME_FILE_OVERLAP = 0.4
#: A child whose vocabulary is this covered by the parent is re-doing the parent.
_PARENT_COVERAGE = 0.8
_PATH_RE = re.compile(r"[\w./-]+\.(?:py|ts|tsx|js|jsx|json|md|sql|yaml|yml)\b")


def _work_fingerprint(*texts: object) -> frozenset:
    """Content words of ``texts``, lowercased; the unit of duplicate detection."""
    words: set[str] = set()
    for text in texts:
        if not isinstance(text, str):
            continue
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if len(word) > 2 and word not in _DUPLICATE_STOPWORDS:
                words.add(word)
    return frozenset(words)


def _stated_paths(*texts: object) -> set[str]:
    """File paths named in ``texts`` (``kanban_fleet_routing.py`` → the leaf)."""
    found: set[str] = set()
    for text in texts:
        if not isinstance(text, str):
            continue
        for match in _PATH_RE.findall(text):
            found.add(match.replace("\\", "/").rsplit("/", 1)[-1].lower())
    return found


def _overlap(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _duplicate_reason(
    child: dict, seen: dict, parent_fp: frozenset, parent_paths: set[str],
) -> tuple[str, Optional[int]]:
    """Why ``child`` must not be spawned, or ``("", None)`` when it is new work.

    A triage card is decomposed when one big request hides several jobs. It must
    never be decomposed into the same job more than once: the parent already
    owns its own scope (re-doing it makes the graph wait on work it is itself
    doing), and a second child for work an earlier child claims means two
    workers editing one file at the same time — the failure recorded on
    ``t_2e0cd7e4``, where a parked defect card decomposed into concurrent edits.
    """
    fp = _work_fingerprint(child["title"])
    if fp and len(fp) >= 3 and parent_fp and len(fp & parent_fp) / len(fp) >= _PARENT_COVERAGE:
        return "duplicates the parent's own scope", None
    paths = _stated_paths(child["title"], child.get("body"))
    if paths and parent_paths and paths & parent_paths and fp and len(fp & parent_fp) / len(fp) >= _SAME_FILE_OVERLAP:
        return "duplicates the parent's own scope on the same file", None
    for prior_index, prior in sorted(seen.items()):
        if _overlap(fp, prior["fp"]) >= _SIBLING_OVERLAP:
            return f"duplicates tasks[{prior_index}]", prior_index
        if paths and prior["paths"] and paths & prior["paths"] and _overlap(fp, prior["fp"]) >= _SAME_FILE_OVERLAP:
            return f"duplicates tasks[{prior_index}] on the same file", prior_index
    return "", None


def _drop_duplicates(
    task_id: str, children: list[dict], *,
    parent_title: object = None, parent_body: object = None,
) -> tuple[list[dict], str]:
    """Remove children that duplicate the parent's scope or an earlier child.

    Returns ``(children, "")`` or ``([], reason)`` when every proposal was a
    duplicate — the card then stays in triage for a human instead of silently
    creating nothing. Sibling ``parents`` indices are remapped to the survivors.

    A dependency edge must never lose its owner. Deleting an edge because its
    target was dropped lets a promoted child run without the prerequisite it
    declared (the review finding on ``t_2e0cd7e4``); pointing it at nothing
    stalls it forever, because the graph refuses to promote a child whose parent
    is not done. So a kept child that named a dropped SIBLING is repointed at the
    sibling that absorbed that work, and a kept child that named a child dropped
    for re-doing the PARENT's scope is dropped too: the root waits on every
    child (``kanban_db_graph.decompose_triage_task`` links each child as a parent
    of the root), so the parent's own scope can never run before its children and
    no edge can be satisfied by waiting on the root.
    """
    if not children:
        return children, ""
    parent_fp = _work_fingerprint(parent_title, parent_body)
    parent_paths = _stated_paths(parent_title, parent_body)
    keep = [True] * len(children)
    seen: dict[int, dict] = {}
    absorbed_by: dict[int, int] = {}
    dropped: list[str] = []
    for idx, child in enumerate(children):
        reason, owner = _duplicate_reason(child, seen, parent_fp, parent_paths)
        if not reason:
            seen[idx] = {
                "fp": _work_fingerprint(child["title"]),
                "paths": _stated_paths(child["title"], child.get("body")),
            }
            continue
        keep[idx] = False
        dropped.append(f"tasks[{idx}] {reason}")
        logger.warning("decompose: task %s dropped tasks[%d] — %s", task_id, idx, reason)
        if owner is not None:
            absorbed_by[idx] = owner

    def _owner(start: int) -> int | None:
        """The kept child that now owns ``start``'s work, or None if nobody does."""
        target = start
        while not keep[target]:
            if target not in absorbed_by:
                return None
            target = absorbed_by[target]
        return target

    # Cascade: a prerequisite dropped with no surviving owner takes its dependants
    # with it, transitively. ``absorbed_by`` points at kept siblings only, so each
    # hop is final and this terminates.
    changed = True
    while changed:
        changed = False
        for idx in range(len(children)):
            if not keep[idx]:
                continue
            for parent in children[idx]["parents"]:
                if _owner(parent) is None:
                    keep[idx] = False
                    reason = f"depends on tasks[{parent}], dropped with no surviving owner"
                    dropped.append(f"tasks[{idx}] {reason}")
                    logger.warning("decompose: task %s dropped tasks[%d] — %s", task_id, idx, reason)
                    changed = True
                    break

    if not any(keep):
        return [], "every proposed child duplicates work already owned: " + "; ".join(dropped)
    index_map = {old: new for new, old in enumerate(i for i in range(len(children)) if keep[i])}
    kept: list[dict] = []
    for idx, child in enumerate(children):
        if not keep[idx]:
            continue
        repointed: list[int] = []
        for parent in child["parents"]:
            owner = _owner(parent)
            mapped = index_map[owner] if owner is not None else None
            if mapped is not None and mapped != index_map[idx] and mapped not in repointed:
                repointed.append(mapped)
        child["parents"] = repointed
        kept.append(child)
    return kept, ""


def _clean_children(task_id: str, raw_tasks: list, routing: _Routing) -> tuple[list[dict], str]:
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
    return children, ""


def _apply_fanout(task: kb.Task, parsed: dict, routing: _Routing, author: str) -> DecomposeOutcome:
    task_id = task.id
    raw_tasks = parsed.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return DecomposeOutcome(task_id, False, "decomposer returned fanout=true with empty tasks list")
    children, reason = _clean_children(task_id, raw_tasks, routing)
    if reason:
        return DecomposeOutcome(task_id, False, reason)
    children, reason = _drop_duplicates(
        task_id, children, parent_title=getattr(task, "title", None),
        parent_body=getattr(task, "body", None),
    )
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
    if not parsed.get("fanout"):
        return _apply_single(task, parsed, routing, audit_author)
    return _apply_fanout(task, parsed, routing, audit_author)


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Return task ids currently in the triage column."""
    with kbc.connect_closing() as conn:
        rows = kb.list_tasks(conn, status="triage", tenant=tenant, limit=1000)
    return [row.id for row in rows]
