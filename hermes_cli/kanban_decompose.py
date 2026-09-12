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
        "parents": [<int>, ...],
        "hold": false
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
  - Set "hold": true on the FINAL terminal child whose completion ships to
    production / releases to a human (deploy, release, rollout, go-live). The
    system creates it as a hard operator_hold — it will NOT dispatch until an
    owner unblocks it after review. This is mandatory for deploy-shaped tasks:
    a deploy card without a real hold can auto-run past an owner approval gate
    and is a serious safety hole. When in doubt, hold.

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


def _is_decision_shaped(title: str) -> bool:
    """True when a child title reads as a product decision, not a task."""
    if not title:
        return False
    return bool(_DECISION_TITLE_RE.search(title))


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


def _is_deploy_shaped(title: str) -> bool:
    """True when a child title names a deploy/release/rollout to production.

    Used to force a real ``operator_hold`` on auto-decomposed deploy children
    (see ``decompose_task``), so a deployment gate is machine-enforced rather
    than advisory prose.
    """
    if not title:
        return False
    return bool(_DEPLOY_TITLE_RE.search(title))


@dataclass
class DecomposeOutcome:
    """Result of decomposing a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    fanout: bool = False
    child_ids: list[str] | None = None
    new_title: Optional[str] = None
    dry_run_plan: Optional[list[dict]] = None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"

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
        with kb.connect_closing() as conn:
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
    dry_run: bool = False,
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
    if task.status not in ("triage", "blocked"):
        return DecomposeOutcome(
            task_id, False, f"task is not in triage/blocked (status={task.status!r})"
        )
    # Guard 1 (auto-decomposer fix, t_c52b9bc3): NO SPLIT MID-REVIEW. A card in
    # an active review cycle (status review/changes_requested, OR its newest
    # decisive run outcome is review_requested/changes_requested with no
    # intervening completion) must be RESUMED in the same card + worktree, not
    # fanned out. A blocked fix-round is a resume trigger, never a fan-out
    # trigger. Refusing here (rather than in the tick) makes the guard hold for
    # both the auto-decompose tick and the manual `kanban decompose <id>` CLI.
    if _has_active_review_cycle(task_id):
        return DecomposeOutcome(
            task_id,
            False,
            "task is in an active review cycle; resuming in same card + "
            "worktree, refusing to split (blocked fix-round is a resume, not a "
            "fan-out trigger)",
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
        # Fall back to single-task spec promotion (same effect as specify).
        new_title = parsed.get("title")
        new_body = parsed.get("body")
        title_val = new_title.strip() if isinstance(new_title, str) and new_title.strip() else None
        body_val = new_body if isinstance(new_body, str) and new_body.strip() else None
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
        if dry_run:
            return DecomposeOutcome(
                task_id, True, "dry-run: single task (no fanout) — nothing written",
                fanout=False, new_title=title_val,
                dry_run_plan=[{
                    "title": title_val,
                    "body": body_val,
                    "assignee": assignee_val,
                }],
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
                task_id, idx, assignee, default_assignee,
            )
        parents = entry.get("parents") or []
        if not isinstance(parents, list):
            parents = []
        # Clean parent indices: drop non-int and out-of-range.
        clean_parents = [p for p in parents if isinstance(p, int) and 0 <= p < len(raw_tasks) and p != idx]
        is_auto = (audit_author == AUTO_DECOMPOSER_AUTHOR)
        # Decision-shaped children park in triage for the PM. Deploy/terminal
        # children — anything whose completion ships to production or releases
        # to a human — are the opposite: they MUST be operator-held at creation,
        # never dispatched toward `done` until an owner unblocks them. A prose
        # "HELD pending approval" in the body is NOT a gate (recompute_ready
        # auto-promotes it the moment its parent completes — the 2026-09-04
        # GlobalAside incident). So any auto-decomposed child whose title is
        # deploy/release/rollout-shaped (or explicitly flagged `hold`) is
        # created as a real DB operator_hold block. Over-holding is safe (it
        # just waits for an unblock); under-holding is the bug we are closing.
        hold_flag = bool(entry.get("hold"))
        if is_auto and _is_deploy_shaped(title):
            hold_flag = True
        children.append({
            "title": title.strip()[:200],
            "body": body.strip(),
            "assignee": chosen,
            "parents": clean_parents,
            # AC1: an auto-decomposer-spawned child whose title is a product
            # decision must land in ``triage`` (never ``ready``) so a ghost PM
            # run cannot self-complete it. Manually-specified (non auto) runs
            # keep current behavior: they are already owner-committed.
            "triage": is_auto and _is_decision_shaped(title),
            # Terminal/deploy child -> hard operator_hold so it can never
            # auto-promote past an owner approval gate.
            "hold": hold_flag,
        })

    if dry_run:
        plan = [
            {
                "title": c["title"],
                "body": c["body"],
                "assignee": c["assignee"],
                "parents": c["parents"],
                "triage": c["triage"],
                "hold": c["hold"],
            }
            for c in children
        ]
        return DecomposeOutcome(
            task_id, True,
            f"dry-run: would fan out into {len(children)} children — nothing written",
            fanout=True, child_ids=None,
            dry_run_plan=plan,
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
    """Return task ids currently in the triage column.

    Excludes cards that the auto-decomposer itself created in ``triage``
    (i.e. the decision-shaped children it demoted so the PM can accept
    them). Without this exclusion the dispatcher's auto-decompose tick
    would re-decompose those parked decisions on the next tick, defeating
    the AC1/AC2 gate. Genuine user-dropped triage (``created_by`` a real
    profile/user) is still returned and decomposed as normal.
    """
    with kb.connect_closing() as conn:
        rows = kb.list_tasks(
            conn,
            status="triage",
            tenant=tenant,
            limit=1000,
        )
    return [
        row.id for row in rows
        if (row.created_by or "") != AUTO_DECOMPOSER_AUTHOR
        # "decomposer" is a legacy fallback author, never produced by the live
        # dispatcher: decompose_triage_task stamps it ("decompose_triage_task"
        # in kanban_db.py:7452 via `author or "decomposer"`) only when no author
        # is supplied. Filter it like the auto-decomposer author so such triage
        # can still be re-decomposed instead of being silently stranded.
        and (row.created_by or "") != "decomposer"
        # 2026-09-03 (charter §6): a card that block_task routed to triage as a
        # LOOP BREAKER (same-kind re-block, block_recurrences >= limit) is parked
        # for a HUMAN decision. Auto-decomposing it would hand the escalation
        # ceiling to the decomposer model and re-run the card without Richie.
        and not (
            (row.block_kind or "") != ""
            and int(row.block_recurrences or 0) >= kb.BLOCK_RECURRENCE_LIMIT
        )
    ]
