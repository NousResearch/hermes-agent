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
(tighten + promote, no children), making ``decompose`` a strict superset;
``question=true`` parks the card itself as a single question card
(``blocked`` / ``needs_input``, no assignee, no children) — an open premise
is a decision for a human, never a work graph. A deterministic marker check
backs the prompt rule: an entry whose own words record an undecided premise
can never be turned into spawned work, whatever the model answers.
Unknown assignees are rewritten to ``default_assignee`` — a child NEVER ends
up with ``assignee=None``.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task, question_triage_task
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
  - An entry whose premise rests on a decision that has NOT been made is
    an open question, not work — use the question shape below, never a
    work graph and never a single work item.

STOP — an open question is not work. When the task's premise depends on a
decision nobody has made yet — the entry says the work "depends on the
choice of" or "depends on a decision", records that there is "no decision"
or "aucune décision", or carries an explicit "undecided" / "to be decided"
/ "à arbitrer" / "à décider" / "question ouverte" / "TBD" marker — do NOT
decompose it into implementation tasks and do NOT return it as a single
work item. Return instead:

  {
    "fanout": false,
    "question": true,
    "rationale": "<one sentence: which premise is open, which decision is missing>",
    "title": "<the open question itself, <= 80 chars>",
    "body":  "<question card body, see structure below>"
  }

The question body MUST include these sections, each prefixed with a bold
markdown heading, in this order:

  **Question** — the decision that must be made, one sentence.
  **Options** — the concrete options exactly as the entry records them
      (quote the entry; if only one direction is recorded, say so). Never
      invent options the entry does not hint at.
  **Impact** — the work that stays blocked until the decision is made.
  **Source** — the entry line(s) that record the open premise, verbatim.

A question card is never assigned and never spawns children — the decision
belongs to a human. A decided task (a concrete goal with no open premise)
decomposes normally: never manufacture a question out of decided work.

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


def _strip_accents(text: str) -> str:
    """Lowercase, accent-stripped form for marker matching: ``dépend du choix``
    must match regardless of accents or case, without normalizing the card."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


# Open-premise markers: phrases by which an entry records that its premise
# rests on a decision nobody has made. Matched against the accent-stripped
# lowercase text, so they are deliberately NARROW — a bare "open" never
# matches ("open source", "open the port" are technical, not undecided) and
# neither does a bare "depends on"/"dépend de" ("depends on the provider's
# API" is a settled technical dependency, not an open product decision).
_OPEN_QUESTION_MARKER_RES = tuple(re.compile(p) for p in (
    # The work depends on a choice that has not been made.
    r"\bdepend du choix\b",
    r"\bdepends? on the choice\b",
    r"\bdepend d[']?une decision\b",
    r"\bdepend de la decision\b",
    r"\bdepends? on (?:a|the) decision\b",
    # No recorded decision.
    r"\baucune decision\b",
    r"\bno decision (?:has been |is |was )?(?:yet )?recorded\b",
    r"\bno decision (?:has been |is |was )?(?:yet )?made\b",
    # An explicit to-be-decided marker.
    r"\ba arbitrer\b",
    r"\bto arbitrate\b",
    r"\ba decider\b",
    r"\ba trancher\b",
    r"\bto be decided\b",
    r"\bdecision non (?:prise|tranchee|arrete|enregistree)\b",
    r"\bdecision not (?:taken|made|recorded)\b",
    r"\bpas encore decide(?:e|es|s)?\b",
    r"\bnot yet decided\b",
    r"\bnon decide(?:e|es|s)?\b",
    r"\bundecided\b",
    # An explicit open-question marker.
    r"\bquestion ouverte\b",
    r"\bopen question\b",
    r"\btbd\b",
))


def _open_question_marker(task: kb.Task) -> Optional[str]:
    """First open-premise marker phrase found in the task's title/body
    (accent-stripped, lowercase), else ``None``.

    The deterministic backstop behind the prompt rule: the prompt teaches the
    model that an open premise is a question card, this check enforces it with
    the entry's own words — the model never gets to fan an entry out into
    spawned work when the entry itself records that its premise is undecided.
    """
    norm = _strip_accents(f"{task.title or ''}\n{task.body or ''}")
    for pattern in _OPEN_QUESTION_MARKER_RES:
        match = pattern.search(norm)
        if match:
            return match.group(0)
    return None


@dataclass
class DecomposeOutcome:
    """Result of decomposing a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    fanout: bool = False
    question: bool = False
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


def _apply_question(
    task: kb.Task, parsed: dict, author: str, *, forced_marker: Optional[str] = None,
) -> DecomposeOutcome:
    """``question=true``: park the triage card itself as the question card —
    ``blocked`` / ``needs_input``, no assignee, no children. An open premise
    is a decision for a human, never a work graph.

    ``forced_marker`` is the deterministic backstop firing: the entry carries
    an open-premise marker but the LLM answered work (a fanout graph or a
    single work item). Its drafted spec describes work, not the question, so
    the question body falls back to the entry verbatim — the options the
    entry records ARE the question — under a header naming the marker.
    """
    title_val, body_val = _title_body(parsed)
    rationale = parsed.get("rationale")
    reason = (
        rationale.strip() if isinstance(rationale, str) and rationale.strip()
        else f"open premise in the entry ({forced_marker!r}) — decision required"
        if forced_marker is not None
        else "open premise — decision required"
    )
    if forced_marker is not None:
        entry_title = (task.title or "").strip()
        title_val = f"Open question: {entry_title}" if entry_title else "Open question"
        body_val = (
            "Open question — the premise of this entry is not decided "
            f"(marker: {forced_marker!r}); a decision is required before any work.\n\n"
            + (task.body or "").strip()
        )
    if body_val is None:
        return DecomposeOutcome(task.id, False, "decomposer returned question=true with no body")
    if title_val is None:
        title_val = task.title
    with kbc.connect_closing() as conn:
        ok = question_triage_task(
            conn, task.id, title=title_val, body=body_val, reason=reason, author=author,
        )
    if not ok:
        return DecomposeOutcome(task.id, False, "task moved out of triage before parking")
    return DecomposeOutcome(task.id, True, "question card (open premise)", question=True)


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
    if parsed.get("question"):
        return _apply_question(task, parsed, audit_author)
    marker = _open_question_marker(task)
    if marker is not None:
        # Deterministic backstop: the entry's own words record an undecided
        # premise; a work answer (fanout graph or single work item) contradicts
        # them. The prompt rule is the primary detector, this is the
        # enforcement — the model never gets to fan an undecided premise out
        # into spawned work.
        logger.warning(
            "decompose: task %s carries open-premise marker %r but the LLM "
            "answered work (%s) — parking as a question card instead",
            task_id, marker, "fanout" if parsed.get("fanout") else "single",
        )
        return _apply_question(task, parsed, audit_author, forced_marker=marker)
    if not parsed.get("fanout"):
        return _apply_single(task, parsed, routing, audit_author)
    return _apply_fanout(task_id, parsed, routing, audit_author)


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Return task ids currently in the triage column."""
    with kbc.connect_closing() as conn:
        rows = kb.list_tasks(conn, status="triage", tenant=tenant, limit=1000)
    return [row.id for row in rows]
