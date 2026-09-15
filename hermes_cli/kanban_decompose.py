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

Rule 6 — EARS gate: this module's aux-LLM call is the one real choke point
where a triage task becomes concrete child tasks, so this is where the
EARS-restatement gate lives (not decision-hud, not the dispatch loop). The
same JSON response that produces each child's title/body/assignee is asked
to also produce an ``ears_sentence`` — a one-sentence EARS-pattern
(Ubiquitous/Event-driven-When/State-driven-While/Unwanted-behavior-If-then/
Optional-Where) restatement of that child's requirement. Two independent
checks guard against the LLM "grading its own homework": (1) the LLM may
self-report ``ears_refusal`` instead of guessing when the task doesn't give
it enough to restate without inventing details; (2) even a claimed
``ears_sentence`` is mechanically re-validated against the five templates by
:func:`validate_ears_sentence` — a plausible-but-malformed sentence is
treated exactly like an explicit refusal, never trusted at face value.

OWNER CHOICE (design decision, preserved from the originating plan): on
refusal/failure, this module does **not** block task creation. It calls
:func:`hermes_cli.plugin_bridges.decision_hud.push_problem_report`
(fail-open — logged, never raised, mirroring ``check_batch_approval``'s
contract) and creates the child with ``ears_sentence=None``. Rationale: (a)
Rule 6 is a pre-dispatch SCOPING aid, not (yet) a dispatch gate — nothing
downstream reads ``ears_sentence`` to permit/deny work, so blocking here
would strand triage work behind a field with no enforcement consumer; (b)
the existing F1 batch-approval gate already owns the "some work needs PO
sign-off before running" mechanism, and duplicating a second blocking gate
on the same choke point stacks two different escalation semantics onto one
call path. The problem report gives a human (or a future gate) full
visibility into every failure without adding a new hard stop today. If
``ears_sentence`` is later wired into enforcement, this default should be
revisited.
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
from hermes_cli.plugin_bridges import decision_hud as _dh_bridge

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
        "parents": [<int>, ...],
        "ears_sentence": "<one EARS-pattern sentence restating this child's requirement, or omit/null>",
        "ears_refusal": "<one sentence on what's missing, ONLY if you cannot EARS-ify without inventing details>"
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
  - "ears_sentence": restate the child's requirement as EXACTLY ONE
    sentence in one of these five EARS patterns:
      Ubiquitous:        "The <system> shall <response>."
      Event-driven:      "When <trigger>, the <system> shall <response>."
      State-driven:      "While <state>, the <system> shall <response>."
      Unwanted behavior: "If <condition>, then the <system> shall <response>."
      Optional feature:  "Where <feature is included>, the <system> shall <response>."
    Use ONLY facts present in the title/body/original task — never invent
    a trigger, state, or system behavior that isn't there. If the task
    doesn't give you enough to do this honestly, do NOT guess: omit
    "ears_sentence" and instead set "ears_refusal" to one sentence
    explaining what's missing.

When the task is genuinely a single unit of work (no useful decomposition),
return:

  {
    "fanout": false,
    "rationale": "<one sentence>",
    "title": "<tightened title>",
    "body":  "<concrete spec for a single worker>",
    "assignee": "<profile name from the roster, or null for default>",
    "ears_sentence": "<one EARS-pattern sentence, or omit/null>",
    "ears_refusal": "<one sentence on what's missing, ONLY if you cannot EARS-ify>"
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

# --- Rule 6: mechanical EARS-shape validator ---------------------------------
# One regex per template (case-insensitive "shall"/"the system" wording is
# lenient — the LLM names the real system/worker, not a literal "system").
# This is intentionally independent of the LLM call that produced the
# sentence: it never trusts the LLM's own claim of success (see module
# docstring "grading its own homework").
_EARS_PATTERNS = (
    # Event-driven: "When <trigger>, the <system> shall <response>."
    re.compile(r"^when\b.+,\s*(the\s+)?\S.*\bshall\b.+\.$", re.IGNORECASE),
    # State-driven: "While <state>, the <system> shall <response>."
    re.compile(r"^while\b.+,\s*(the\s+)?\S.*\bshall\b.+\.$", re.IGNORECASE),
    # Unwanted behavior: "If <condition>, then the <system> shall <response>."
    re.compile(r"^if\b.+,\s*then\s+(the\s+)?\S.*\bshall\b.+\.$", re.IGNORECASE),
    # Optional feature: "Where <feature>, the <system> shall <response>."
    re.compile(r"^where\b.+,\s*(the\s+)?\S.*\bshall\b.+\.$", re.IGNORECASE),
    # Ubiquitous: "The <system> shall <response>." (checked last — the most
    # permissive shape, so trigger/state/condition words above match first).
    re.compile(r"^the\s+\S.*\bshall\b.+\.$", re.IGNORECASE),
)


def validate_ears_sentence(sentence: object) -> bool:
    """True iff ``sentence`` is a single, non-empty string that mechanically
    matches one of the five EARS templates (Ubiquitous/Event-driven-When/
    State-driven-While/Unwanted-behavior-If-then/Optional-Where). Shape-only
    — does not (cannot) verify the restatement is faithful to the original
    task, only that it has the grammatical form a human reviewer expects.
    """
    if not isinstance(sentence, str):
        return False
    text = sentence.strip()
    if not text:
        return False
    return any(p.match(text) for p in _EARS_PATTERNS)


def _resolve_ears_sentence(
    entry: dict, *, task_id: str, child_title: str,
) -> tuple[Optional[str], Optional[dict]]:
    """Validate ``entry.get("ears_sentence")``; returns
    ``(ears_sentence, pending_report)`` — never invents a sentence.
    ``pending_report`` is a kwargs dict for
    :func:`hermes_cli.plugin_bridges.decision_hud.push_problem_report`, or
    ``None`` when validation succeeded.

    IMPORTANT: this function no longer pushes the report itself. Adversarial
    review found the report was previously filed eagerly, before the
    caller's DB write (``specify_triage_task``/``decompose_triage_task``)
    had even been attempted — a subsequent write failure (a race under
    concurrent decompose sweeps, a DB error, "already decomposed") left
    orphaned problem reports referencing tasks/children that were never
    actually created. Callers must call
    :func:`_flush_pending_ears_reports` only AFTER their write succeeds.
    See the module docstring's OWNER CHOICE section for the fail-open
    (proceed with ``ears_sentence=None``) rationale — that design is
    unchanged; only the report's timing moved.
    """
    candidate = entry.get("ears_sentence")
    if validate_ears_sentence(candidate):
        return candidate.strip(), None  # type: ignore[union-attr]

    refusal = entry.get("ears_refusal")
    if isinstance(refusal, str) and refusal.strip():
        problem = (
            f"Decomposer could not EARS-ify child task {child_title!r} "
            f"(root task {task_id}): {refusal.strip()}"
        )
    else:
        problem = (
            f"Decomposer produced no valid EARS restatement for child task "
            f"{child_title!r} (root task {task_id}); LLM did not explicitly "
            f"refuse, but the returned sentence (if any) failed the mechanical "
            f"EARS-shape check: {candidate!r}"
        )
    pending_report = {
        "project": "kanban", "problem": problem,
        "context": f"task_id={task_id}", "reporter": "decomposer",
    }
    return None, pending_report


def _flush_pending_ears_reports(pending_reports: list[dict], *, task_id: str) -> None:
    """Push every pending EARS problem report — call ONLY after the
    caller's DB write for these entries has actually succeeded."""
    for pending_report in pending_reports:
        ok, detail = _dh_bridge.push_problem_report(**pending_report)
        if not ok:
            logger.warning(
                "decompose: EARS problem report failed for task %s: %s", task_id, detail,
            )


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
    mode_title = title_val if title_val is not None else task.title
    # Rule 6: resolve the EARS restatement (validate/refuse+report; never
    # invent) — see module docstring for the fail-open owner choice.
    ears_sentence_val, pending_report = _resolve_ears_sentence(
        parsed, task_id=task.id, child_title=mode_title,
    )
    with kbc.connect_closing() as conn:
        ok = kb.specify_triage_task(
            conn, task.id, title=title_val, body=body_val, assignee=assignee_val, author=author,
            ears_sentence=ears_sentence_val,
        )
    if not ok:
        return DecomposeOutcome(task.id, False, "task moved out of triage before promotion")
    # Flush only after the write is confirmed — a report referencing a
    # promotion that never happened would be misleading.
    if pending_report is not None:
        _flush_pending_ears_reports([pending_report], task_id=task.id)
    return DecomposeOutcome(task.id, True, "single task (no fanout)", fanout=False, new_title=title_val)


def _clean_children(
    task_id: str, raw_tasks: list, routing: _Routing,
) -> tuple[list[dict], str, list[dict]]:
    """Validate/normalise the LLM's ``tasks`` list; ``(children, "", pending_reports)``
    or ``([], reason, [])``. Unknown assignees route to the default; never
    assignee=None. ``pending_reports`` are EARS problem reports to flush
    ONLY after the caller's DB write for these children succeeds."""
    children: list[dict] = []
    pending_reports: list[dict] = []
    for idx, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict):
            return [], f"tasks[{idx}] is not an object", []
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            return [], f"tasks[{idx}].title is missing or empty", []
        title_clean = title.strip()[:200]
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
        # Rule 6: resolve the EARS restatement (validate/refuse+report;
        # never invent) — see module docstring for the fail-open owner
        # choice. Report is deferred, not fired here — see
        # _flush_pending_ears_reports.
        ears_sentence_val, pending_report = _resolve_ears_sentence(
            entry, task_id=task_id, child_title=title_clean,
        )
        if pending_report is not None:
            pending_reports.append(pending_report)
        children.append({
            "title": title_clean,
            "body": body.strip() if isinstance(body, str) else "",
            "assignee": chosen,
            "ears_sentence": ears_sentence_val,
            # Drop non-int, out-of-range and self parent indices.
            "parents": [p for p in parents if isinstance(p, int) and 0 <= p < len(raw_tasks) and p != idx],
        })
    return children, "", pending_reports


def _apply_fanout(task_id: str, parsed: dict, routing: _Routing, author: str) -> DecomposeOutcome:
    raw_tasks = parsed.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return DecomposeOutcome(task_id, False, "decomposer returned fanout=true with empty tasks list")
    children, reason, pending_reports = _clean_children(task_id, raw_tasks, routing)
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
    # Flush only after the fan-out write is confirmed — reports filed
    # before this point would reference children that were never created
    # if the write above had failed or no-opped.
    if pending_reports:
        _flush_pending_ears_reports(pending_reports, task_id=task_id)
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
