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
import time
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
_DEFAULT_SIBLING_OVERLAP = 0.75
#: Two children naming the same file are duplicates at this lower overlap.
_DEFAULT_SAME_FILE_OVERLAP = 0.4
#: A child that restates the parent's TITLE (Jaccard, both ways) is re-doing it.
#: Deliberately measured against the title, not title+body: a child's words
#: nearly always come from the parent's specification, so a body-inclusive
#: containment test dropped every genuinely narrower child (t_d97ab27d, 2026-11-08).
_DEFAULT_PARENT_OVERLAP = 0.75
#: Same stated file lowers the parent bar, but not to mere vocabulary overlap.
_SAME_FILE_PARENT_OVERLAP = 0.6
_PATH_RE = re.compile(r"[\w./-]+\.(?:py|ts|tsx|js|jsx|json|md|sql|yaml|yml)\b")


def _thresholds() -> tuple[float, float, float]:
    """``(sibling, same_file, parent)`` overlap thresholds from ``kanban.decompose``.

    Read fresh per decomposition so a threshold can be tuned on the live board
    without a gateway restart; a bad or missing value falls back to its default.
    """
    defaults = (_DEFAULT_SIBLING_OVERLAP, _DEFAULT_SAME_FILE_OVERLAP, _DEFAULT_PARENT_OVERLAP)
    try:
        from hermes_cli.config import load_config

        cfg = ((load_config() or {}).get("kanban") or {}).get("decompose") or {}
    except Exception:
        return defaults
    resolved: list[float] = []
    for key, default in zip(("sibling_overlap", "same_file_overlap", "parent_overlap"), defaults):
        try:
            resolved.append(float(cfg.get(key, default)))
        except (TypeError, ValueError):
            resolved.append(default)
    return resolved[0], resolved[1], resolved[2]


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
    thresholds: Optional[tuple[float, float, float]] = None,
) -> tuple[str, Optional[int]]:
    """Why ``child`` must not be spawned, or ``("", None)`` when it is new work.

    A triage card is decomposed when one big request hides several jobs. It must
    never be decomposed into the same job more than once: the parent already
    owns its own scope (re-doing it makes the graph wait on work it is itself
    doing), and a second child for work an earlier child claims means two
    workers editing one file at the same time — the failure recorded on
    ``t_2e0cd7e4``, where a parked defect card decomposed into concurrent edits.

    What it must NOT do is drop a child for being narrower than the parent. A
    child's words nearly always come from the parent's specification, so
    containment is not evidence of duplication; the parent test is therefore a
    two-way Jaccard measurement, which only fires when the child restates the
    parent's scope rather than carving a piece out of it.
    """
    sibling_overlap, same_file_overlap, parent_overlap = thresholds or _thresholds()
    fp = _work_fingerprint(child["title"])
    paths = _stated_paths(child["title"], child.get("body"))
    if fp and parent_fp:
        parent_sim = _overlap(fp, parent_fp)
        if parent_sim >= parent_overlap:
            return "duplicates the parent's own scope", None
        if paths and parent_paths and paths & parent_paths and parent_sim >= _SAME_FILE_PARENT_OVERLAP:
            return "duplicates the parent's own scope on the same file", None
    for prior_index, prior in sorted(seen.items()):
        if _overlap(fp, prior["fp"]) >= sibling_overlap:
            return f"duplicates tasks[{prior_index}]", prior_index
        if paths and prior["paths"] and paths & prior["paths"] and _overlap(fp, prior["fp"]) >= same_file_overlap:
            return f"duplicates tasks[{prior_index}] on the same file", prior_index
    return "", None


def _drop_duplicates(
    task_id: str, children: list[dict], *,
    parent_title: object = None, parent_body: object = None,
) -> tuple[list[dict], str]:
    """Remove children that duplicate the parent's scope or an earlier child.

    Returns ``(children, "")`` or ``([], reason)`` when every proposal was a
    duplicate — the caller then promotes the root as a single card, naming the
    drops, rather than leaving an orphan in triage. Sibling ``parents`` indices
    are remapped to the survivors.

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
    parent_fp = _work_fingerprint(parent_title)
    parent_paths = _stated_paths(parent_title, parent_body)
    thresholds = _thresholds()
    keep = [True] * len(children)
    seen: dict[int, dict] = {}
    absorbed_by: dict[int, int] = {}
    dropped: list[str] = []
    for idx, child in enumerate(children):
        reason, owner = _duplicate_reason(child, seen, parent_fp, parent_paths, thresholds)
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
        # Every proposal duplicated work already owned. Leaving the card in
        # triage made it an orphan: the dispatcher re-asks about it on every tick
        # and nothing ever runs it. Promote the root as ONE card instead — the
        # work then executes, and the comment says why it was not split.
        logger.warning("decompose: task %s — %s; promoting the root as one card", task_id, reason)
        try:
            with kbc.connect_closing() as conn:
                kb.add_comment(
                    conn, task_id, author or "auto-decomposer",
                    f"Auto-decomposer: {reason} Promoted as a single card so the work still "
                    "runs; re-specify or archive it if the split was wanted.",
                )
                promoted = kb.specify_triage_task(conn, task_id, author=author)
        except Exception:
            logger.exception("decompose: could not promote %s after dropping every child", task_id)
            return DecomposeOutcome(task_id, False, reason)
        if not promoted:
            return DecomposeOutcome(task_id, False, reason)
        return DecomposeOutcome(
            task_id, True, f"{reason} — promoted as a single card", fanout=False)
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


#: A card that has failed to decompose this many times is left alone for the window.
_DECOMPOSE_FAILURE_LIMIT = 3
_DECOMPOSE_FAILURE_WINDOW_SECONDS = 6 * 3600


def _decomposed_children(conn, task_id: str) -> list[str]:
    """Child ids for an already-decomposed card: the event first, then the links.

    ``decompose_triage_task`` refuses a second fan-out by looking for the
    ``decomposed`` event. Asking the same question here, before the aux LLM is
    called, is what stops a card that has already decomposed being re-planned on
    every dispatcher tick (t_d97ab27d: 111 paid calls on 2026-11-08).
    """
    for event in reversed(kb.list_events(conn, task_id)):
        if event.kind != "decomposed":
            continue
        payload = event.payload if isinstance(event.payload, dict) else {}
        ids = payload.get("child_ids")
        if isinstance(ids, list) and ids:
            return [str(i) for i in ids]
    try:
        rows = conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY rowid", (task_id,),
        ).fetchall()
    except Exception:  # a board without links must not break decomposition
        return []
    return [str(row[0]) for row in rows]


def _recent_decompose_failures(conn, task_id: str) -> int:
    """How many ``decompose_failed`` events this card has inside the window."""
    cutoff = int(time.time()) - _DECOMPOSE_FAILURE_WINDOW_SECONDS
    return sum(
        1 for event in kb.list_events(conn, task_id)
        if event.kind == "decompose_failed" and int(event.created_at or 0) >= cutoff
    )


def _note_already_decomposed(conn, task_id: str, children: list[str]) -> None:
    """Leave one comment explaining why the card is parked in triage. Idempotent."""
    marker = "Auto-decomposer: already decomposed into"
    for comment in kb.list_comments(conn, task_id):
        if (comment.body or "").startswith(marker):
            return
    kb.add_comment(
        conn, task_id, "auto-decomposer",
        f"{marker} {', '.join(children)}. Triage is its resting place until a human "
        "decides whether to run the children, re-specify this card, or archive it. "
        "Decomposition is not re-attempted while those children exist.",
    )


def _record_decompose_failure(conn, task_id: str, reason: str) -> None:
    """Count one failed plan so a repeat failure cannot loop unbounded."""
    try:
        kb._append_event(conn, task_id, "decompose_failed", {"reason": reason[:400]})
    except Exception:
        logger.debug("decompose: could not record failure for %s", task_id, exc_info=True)


def decompose_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
) -> DecomposeOutcome:
    """Decompose a triage task into a graph of child tasks. Expected failures
    (not in triage, no aux client, API error, malformed/empty reply) surface
    as ``ok=False``.

    Two guards run before the aux LLM is touched, because the dispatcher asks
    about every triage card on every tick:

    * a card that already decomposed is reported, not re-planned;
    * a card that has failed ``_DECOMPOSE_FAILURE_LIMIT`` times in the window is
      left for a human rather than retried.
    """
    task, reason = _load_triage_task(task_id)
    if task is None:
        return DecomposeOutcome(task_id, False, reason)

    with kbc.connect_closing() as conn:
        children = _decomposed_children(conn, task_id)
        if children:
            _note_already_decomposed(conn, task_id, children)
            return DecomposeOutcome(
                task_id, False, f"already decomposed into {', '.join(children)}")
        failures = _recent_decompose_failures(conn, task_id)

    if failures >= _DECOMPOSE_FAILURE_LIMIT:
        return DecomposeOutcome(
            task_id, False,
            f"decomposition failed {failures}x in the last "
            f"{_DECOMPOSE_FAILURE_WINDOW_SECONDS // 3600}h; left for a human")

    outcome = _decompose_task_unplanned(task_id, author=author, timeout=timeout)
    if not outcome.ok and outcome.reason:
        with kbc.connect_closing() as conn:
            _record_decompose_failure(conn, task_id, outcome.reason)
    return outcome


def _decompose_task_unplanned(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
) -> DecomposeOutcome:
    """The original plan-and-fan-out path, with the pre-checks already done."""
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
