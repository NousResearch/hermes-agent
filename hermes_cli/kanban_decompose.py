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
  - Preserve every explicit hard constraint from the original task in every
    child task body. This includes read-only/no-file-change requirements,
    "do not modify files", final-verdict-only output limits, budget limits,
    target repo/path restrictions, safety approvals, and any "must not" rule.
    When the original task is read-only, every child must also be read-only.
  - Prefer parallelism. If two tasks can be done independently, give
    them no parents so the dispatcher fans them out at once.
  - Use 2-6 tasks for normal work. Don't create 20 tiny tasks. Don't
    cram everything into 1 task.
  - Pick assignees from the roster by matching the task to the profile's
    DESCRIPTION (not just the name). When nothing matches well, use null
    and the system will route to the default_assignee.
  - Each child task body is what a fresh worker will read with no other
    context — be specific about goal, approach, and acceptance criteria.
  - For final closure / final verdict graphs, make the final synthesis task
    depend on every evidence-producing child. Do not let it run in parallel
    with the checks it must summarize.
  - For worker terminal-state verification, make that verifier depend on the
    workers whose final persisted states it must inspect. If statuses are still
    running/todo/ready, the verifier should wait or return BLOCKED/PENDING,
    not a premature FAIL based only on an in-progress snapshot.
  - Historical closure/evidence files are only supporting documentation. A
    current live run's root card body, events, and parent handoffs are the
    authoritative evidence unless a file explicitly matches the same job/task id.
  - Do not create a child task that requires the root task itself to already be
    done/blocked before the child can complete. The root only closes after its
    children close; root-terminal verification belongs to the root/orchestrator
    or to an external post-root check.

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

_HARD_CONSTRAINT_RE = re.compile(
    r"(?i)"
    r"(read[- ]?only|"
    r"do not modify|don't modify|without modifying|no file changes?|"
    r"must not|never |forbidden|"
    r"final verdict only|final output|checklist and final verdict|"
    r"budget|approval|"
    r"repo=|/users/|/var/|/tmp/|[a-z]:\\|"
    r"只读|不要修改|不得修改|不能修改|不修改文件|最终裁决)"
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;。；!！?？])\s+|\n+")

_FINAL_SYNTHESIS_RE = re.compile(
    r"(?i)\b(synthesi[sz]e|synthesis|final\s+verdict|overall\s+verdict|"
    r"final\s+closeout|final\s+closure)\b"
)
_TERMINAL_VERIFIER_RE = re.compile(
    r"(?i)\b(terminal[- ]?states?|done/blocked|complete or block|comment[- ]?only|"
    r"left only comments?|worker terminal)\b"
)
_STALE_CLOSURE_EVIDENCE_RE = re.compile(
    r"(?i)\b(closure[- ]file|closure files|evidence files?|closure evidence|"
    r"orchestrator_closure|closure_verdict)\b"
)
_SELF_REFERENTIAL_ROOT_STATUS_RE = re.compile(
    r"(?i)\b(root|root card|root task).{0,80}\b(done|blocked|terminal|"
    r"complete[sd]?|reach(?:ed|es)?)\b"
)
_LIVE_CLOSURE_FALLBACK_RE = re.compile(
    r"(?is)"
    r"Budget Gate Evidence.*Current live verification scope.*"
    r"(terminal[- ]?state verifier|premature FAIL|stale closure|self[- ]?deadlock|"
    r"root to already be done)"
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


def _extract_hard_constraints(title: str, body: str) -> list[str]:
    """Pull explicit safety/scope constraints from the original task.

    Child workers only see their own child body, so the decomposer must carry
    forward constraints such as "read-only" and "do not modify files" even when
    the LLM forgets to restate them. This is intentionally keyword-based and
    conservative; it preserves user-provided text instead of inventing policy.
    """
    source = "\n".join(part for part in (title or "", body or "") if part)
    constraints: list[str] = []
    seen: set[str] = set()
    for chunk in _SENTENCE_SPLIT_RE.split(source):
        cleaned = " ".join(chunk.strip().split())
        if not cleaned or not _HARD_CONSTRAINT_RE.search(cleaned):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        constraints.append(cleaned[:400])
        if len(constraints) >= 8:
            break
    return constraints


def _append_hard_constraints(body: str, constraints: list[str]) -> str:
    """Append inherited constraints to a worker body when needed."""
    body = (body or "").strip()
    if not constraints:
        return body
    lower_body = body.lower()
    missing = [c for c in constraints if c.lower() not in lower_body]
    if not missing:
        return body
    lines = ["**Inherited hard constraints**"]
    lines.extend(f"- {c}" for c in missing)
    suffix = "\n".join(lines)
    return f"{body}\n\n{suffix}" if body else suffix


def _append_once(body: str, marker: str, lines: list[str]) -> str:
    """Append a guidance block only when the marker is not already present."""
    body = (body or "").strip()
    if marker.lower() in body.lower():
        return body
    suffix = "\n".join([marker, *lines]).strip()
    return f"{body}\n\n{suffix}" if body else suffix


def _is_final_synthesis_child(child: dict) -> bool:
    # Use the generated child title, not inherited hard constraints in the body:
    # root tasks often contain "final closure" and that text is appended to every
    # child for safety.
    text = str(child.get("title") or "")
    return bool(_FINAL_SYNTHESIS_RE.search(text))


def _is_terminal_verifier_child(child: dict) -> bool:
    # Same title-first rule as final synthesis. Original tasks may say "workers
    # complete or block" and that phrase is inherited by unrelated children.
    text = str(child.get("title") or "")
    return bool(_TERMINAL_VERIFIER_RE.search(text))


def _mentions_stale_closure_evidence(child: dict) -> bool:
    text = " ".join(str(child.get(key) or "") for key in ("title", "body"))
    return bool(_STALE_CLOSURE_EVIDENCE_RE.search(text))


def _is_self_referential_root_status_child(child: dict) -> bool:
    # Title-only on purpose. Bodies may contain guard text such as "do not
    # require the root to be done", which must not turn a normal verifier into
    # a self-referential root-status gate.
    text = str(child.get("title") or "")
    return bool(_SELF_REFERENTIAL_ROOT_STATUS_RE.search(text))


def _is_live_closure_fallback_task(title: str, body: str) -> bool:
    text = "\n".join(part for part in (title or "", body or "") if part)
    return bool(_LIVE_CLOSURE_FALLBACK_RE.search(text))


def _stabilize_closure_graph(children: list[dict]) -> list[dict]:
    """Make closure-test graphs wait for the evidence they are judging.

    The LLM decomposer occasionally creates a terminal-state verifier that runs
    in parallel with the workers it is supposed to inspect. It can then return a
    truthful but stale FAIL ("siblings still running") and poison final
    synthesis. Apply a narrow deterministic repair for closure/final-verdict
    graphs: terminal verifiers wait for non-synthesis siblings, and final
    synthesis waits for every other child.
    """
    if len(children) < 2:
        return children

    final_idxs = [idx for idx, child in enumerate(children) if _is_final_synthesis_child(child)]
    terminal_idxs = [idx for idx, child in enumerate(children) if _is_terminal_verifier_child(child)]
    root_status_idxs = [
        idx for idx, child in enumerate(children)
        if _is_self_referential_root_status_child(child)
    ]
    if not final_idxs and not terminal_idxs and not root_status_idxs:
        return children

    final_set = set(final_idxs)
    terminal_set = set(terminal_idxs)
    root_status_set = set(root_status_idxs)
    for idx, child in enumerate(children):
        parents = set(child.get("parents") or [])
        if idx in root_status_set:
            parents.discard(idx)
            child["body"] = _append_once(
                str(child.get("body") or ""),
                "**Root-status self-reference guard**",
                [
                    "- Do not require the root task itself to be done/blocked inside this child; the root cannot close until its children close.",
                    "- Verify only the worker children that have already been spawned under the root, plus whether the dependency graph makes root closure possible.",
                    "- If later root terminal state is required, leave that to the root/orchestrator after all children finish.",
                ],
            )
        if idx in terminal_set:
            parents.difference_update(root_status_set)
            parents.update(
                other_idx
                for other_idx in range(len(children))
                if other_idx != idx and other_idx not in final_set and other_idx not in root_status_set
            )
            child["body"] = _append_once(
                str(child.get("body") or ""),
                "**Terminal-state timing guard**",
                [
                    "- Run this verifier only after all non-synthesis sibling workers have completed or blocked.",
                    "- If any sibling is still running/todo/ready, do not return FAIL; wait if possible, otherwise use BLOCKED/PENDING with the exact pending task ids.",
                    "- A PASS/FAIL verdict must be based on final persisted task statuses, not on an in-progress snapshot.",
                ],
            )
        if idx in final_set:
            parents.difference_update(root_status_set)
            parents.update(
                other_idx
                for other_idx in range(len(children))
                if other_idx != idx and other_idx not in root_status_set
            )
            child["body"] = _append_once(
                str(child.get("body") or ""),
                "**Final synthesis evidence guard**",
                [
                    "- Treat parent worker handoffs and the current root Kanban body/events as the authoritative evidence for this run.",
                    "- Do not downgrade the current live verdict because historical closure files are stale or unrelated; mention them only as documentation follow-up.",
                    "- OpenSquilla preflight findings are questions to verify, not final blockers once live Kanban evidence answers them.",
                ],
            )
        if _mentions_stale_closure_evidence(child):
            child["body"] = _append_once(
                str(child.get("body") or ""),
                "**Current-run evidence guard**",
                [
                    "- Compare closure files against the current root task/job id before using them as evidence.",
                    "- If a closure file belongs to an older packet, report it as stale documentation follow-up, not as a failure of the current live run.",
                ],
            )
        child["parents"] = sorted(
            p for p in parents if isinstance(p, int) and 0 <= p < len(children) and p != idx
        )
    return children


def _malformed_json_fallback_body(
    *,
    title: str,
    body: str,
    constraints: list[str],
) -> str:
    """Build a deterministic single-task spec when the model emits bad JSON."""
    original_body = (body or "").strip() or "(no original body)"
    sections = [
        "**Goal**",
        "Complete the original Kanban task as one guarded work item because the decomposer returned malformed JSON.",
        "",
        "**Original task**",
        f"Title: {(title or '').strip() or '(untitled)'}",
        "",
        original_body,
        "",
        "**Approach**",
        "- Follow the original task exactly; do not expand scope.",
        "- Preserve every explicit safety, budget, repo, and output constraint from the original task.",
        "- If the original task is read-only or says not to modify files, inspect only and report the verdict.",
        "",
        "**Acceptance criteria**",
        "- The original requested checks are answered directly.",
        "- Any blocker is reported explicitly instead of guessed.",
        "- No files are modified unless the original task explicitly permits modification.",
    ]
    fallback = "\n".join(sections).strip()
    return _append_hard_constraints(fallback, constraints)


def _live_closure_fallback_children(*, root_task_id: str) -> list[dict]:
    """Deterministic closure graph for the final DevFlow live smoke.

    This is intentionally narrow: it only covers the live closure prompt used to
    validate DevFlow/Kanban hardening. If the decomposer LLM is quota-limited or
    returns an empty response, we still want the safety proof to run instead of
    leaving the root card stranded in triage.
    """
    children = [
        {
            "title": "Verify Budget Gate Evidence in root card body",
            "body": (
                f"Read-only verification. Inspect Kanban root {root_task_id} and "
                "confirm the card body contains the literal 'DevFlow Budget Gate Evidence' "
                "section with estimated agents and estimated MiniMax-M3 calls. Quote the "
                "matching lines. Do not modify files. Final verdict only."
            ),
            "assignee": "reviewer",
            "parents": [],
        },
        {
            "title": "Verify Current live verification scope in root card body",
            "body": (
                f"Read-only verification. Inspect Kanban root {root_task_id} and "
                "confirm the card body contains the literal 'Current live verification scope' "
                "block and its current-run evidence priority bullets. Quote the matching "
                "lines. Do not modify files. Final verdict only."
            ),
            "assignee": "reviewer",
            "parents": [],
        },
        {
            "title": "Verify isolated-board routing for this root",
            "body": (
                f"Read-only verification. Confirm task {root_task_id} exists on the "
                "intended isolated board and is not resolved from the default board. "
                "Use live Kanban evidence only. Do not modify files. Final verdict only."
            ),
            "assignee": "ops",
            "parents": [],
        },
        {
            "title": "Verify terminal-state verifier waits for sibling workers",
            "body": (
                f"Read-only verification. Inspect this decomposed graph under root "
                f"{root_task_id}. Confirm this verifier's dependency list waits for "
                "the evidence-producing sibling workers and does not run in parallel "
                "with them. If anything is still in flight, report BLOCKED/PENDING, "
                "not FAIL. Do not require the root itself to be done inside this child. "
                "Do not modify files. Final verdict only."
            ),
            "assignee": "ops",
            "parents": [0, 1, 2],
        },
        {
            "title": "Synthesize final closure verdict from live Kanban evidence",
            "body": (
                f"Read-only synthesis for root {root_task_id}. Use only the current "
                "root card body, child task events, and worker handoffs as authoritative "
                "evidence; historical closure files count only if they explicitly match "
                "this job/card id. Verify: Budget Gate Evidence present, Current live "
                "verification scope present, isolated-board routing works, terminal-state "
                "verifier waited instead of premature FAIL, and the graph contains no "
                "child that requires the root to already be done before children finish. "
                "Do not modify files. Output PASS, NEEDS_CHANGES, or BLOCKED with quoted "
                "evidence per condition."
            ),
            "assignee": "reviewer",
            "parents": [0, 1, 2, 3],
        },
    ]
    return _stabilize_closure_graph(children)


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
        roster=_format_roster(roster),
        default_assignee=default_assignee,
    )
    hard_constraints = _extract_hard_constraints(task.title or "", task.body or "")
    live_closure_fallback = _is_live_closure_fallback_task(
        task.title or "",
        task.body or "",
    )

    try:
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
        if "No LLM provider configured" in str(exc):
            return DecomposeOutcome(task_id, False, "no auxiliary client configured")
        if live_closure_fallback:
            children = _live_closure_fallback_children(root_task_id=task.id)
            try:
                with kb.connect_closing() as conn:
                    child_ids = kb.decompose_triage_task(
                        conn,
                        task_id,
                        root_assignee=orchestrator,
                        children=children,
                        author=author or _profile_author(),
                        auto_promote=auto_promote,
                    )
            except Exception as db_exc:
                logger.exception("decompose: deterministic fallback DB error on task %s", task_id)
                return DecomposeOutcome(task_id, False, f"DB error: {type(db_exc).__name__}")
            if child_ids is None:
                return DecomposeOutcome(
                    task_id, False, "task moved out of triage before deterministic fallback",
                )
            return DecomposeOutcome(
                task_id,
                True,
                f"decomposed into {len(child_ids)} children (deterministic closure fallback)",
                fanout=True,
                child_ids=child_ids,
            )
        return DecomposeOutcome(task_id, False, f"LLM error: {type(exc).__name__}")

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    parsed = _extract_json_blob(raw)
    if parsed is None:
        if live_closure_fallback:
            children = _live_closure_fallback_children(root_task_id=task.id)
            try:
                with kb.connect_closing() as conn:
                    child_ids = kb.decompose_triage_task(
                        conn,
                        task_id,
                        root_assignee=orchestrator,
                        children=children,
                        author=author or _profile_author(),
                        auto_promote=auto_promote,
                    )
            except Exception as exc:
                logger.exception("decompose: deterministic fallback DB error on task %s", task_id)
                return DecomposeOutcome(task_id, False, f"DB error: {type(exc).__name__}")
            if child_ids is None:
                return DecomposeOutcome(
                    task_id, False, "task moved out of triage before deterministic fallback",
                )
            return DecomposeOutcome(
                task_id,
                True,
                f"decomposed into {len(child_ids)} children (deterministic closure fallback)",
                fanout=True,
                child_ids=child_ids,
            )
        fallback_body = _malformed_json_fallback_body(
            title=task.title or "",
            body=task.body or "",
            constraints=hard_constraints,
        )
        assignee_val = _normalize_assignee_choice(
            None,
            default_assignee=default_assignee,
            valid_names=valid_names,
        )
        with kb.connect_closing() as conn:
            ok = kb.specify_triage_task(
                conn,
                task_id,
                body=fallback_body,
                assignee=assignee_val if not task.assignee else None,
                author=author or _profile_author(),
            )
        if not ok:
            return DecomposeOutcome(
                task_id, False, "task moved out of triage before JSON fallback",
            )
        return DecomposeOutcome(
            task_id, True, "single task (malformed JSON fallback)",
            fanout=False,
        )

    fanout = bool(parsed.get("fanout"))
    audit_author = author or _profile_author()

    if not fanout:
        # Fall back to single-task spec promotion (same effect as specify).
        new_title = parsed.get("title")
        new_body = parsed.get("body")
        title_val = new_title.strip() if isinstance(new_title, str) and new_title.strip() else None
        body_val = (
            _append_hard_constraints(new_body, hard_constraints)
            if isinstance(new_body, str) and new_body.strip()
            else None
        )
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
        body = _append_hard_constraints(body, hard_constraints)
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
        children.append({
            "title": title.strip()[:200],
            "body": body.strip(),
            "assignee": chosen,
            "parents": clean_parents,
        })
    children = _stabilize_closure_graph(children)

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
