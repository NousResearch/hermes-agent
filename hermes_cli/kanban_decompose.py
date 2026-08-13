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
  - FULL-TIER TASKS ONLY: if the parent task has tier=full, every child
    body MUST include both of these exact section headers:
    ## Acceptance Criteria
    followed by bullet-pointed criteria.
    ## Test Plan
    followed by the testing strategy.
    This is required for the contract gate (validate_task_contract).
    Fast-tier tasks do NOT need these sections.

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
    """Lenient JSON extraction. Returns None if nothing parses.

    Delegates to the shared ``hermes_cli.llm_json.parse_llm_json``
    (JSON-1 consolidation).
    """
    from hermes_cli.llm_json import parse_llm_json
    return parse_llm_json(raw, raise_on_failure=False)


def _profile_author() -> str:
    """Best-effort author name. Delegates to ``hermes_cli.llm_json``
    (I-3: single source, breaks circular import with kanban.py)."""
    from hermes_cli.llm_json import _profile_author as _pa
    return _pa()


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
    """Resolve which profile catches child tasks the orchestrator can't route.

    FIX 2026-08-13 (Option A routing directive): work routes to LEAD profiles
    first; leads execute and re-delegate to specialists. The previous
    fallback used ``get_active_profile_name()`` — which returned the
    dispatcher gateway's own profile (e.g. ``sirvir`` when the sirvir
    gateway holds the dispatcher lock), silently assigning unrelated
    implementation work to the model-fleet manager.

    New resolution:
      1. explicit ``kanban.default_assignee`` config (unchanged)
      2. the ROOT kensei profile (operator; routes work correctly itself)
      3. literal ``default`` as last resort
    """
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    explicit = (kanban_cfg.get("default_assignee") or "").strip()
    if explicit:
        try:
            if profiles_mod.profile_exists(explicit):
                return explicit
        except Exception:
            pass
    # Root operator profile — never a gateway-incidental profile.
    try:
        if profiles_mod.profile_exists("kensei"):
            return "kensei"
    except Exception:
        pass
    return "default"


def _structured_output_enabled(cfg: dict) -> bool:
    """Return True only when config explicitly opts in to native structured
    output for kanban auxiliary tasks.

    Default OFF. This is the gate that keeps the free-first constraint safe:
    structured output is only ever requested when the user has explicitly
    enabled it AND the model is proven capable (see ``_aux_supports_json_schema``).
    """
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    return bool(kanban_cfg.get("structured_output", False))


def _aux_supports_json_schema(model: str) -> bool:
    """Strict capability probe: True ONLY when the model is PROVEN to support
    ``response_format=json_schema``.

    Unlike tool-calling (where a missing ``supported_parameters`` field is
    treated as permissive), structured output must be proven — we never send
    ``response_format`` to a model we cannot confirm supports it, because that
    would cause an HTTP 400 and regress the decomposer on free-tier models
    (2026-07-19 free-first constraint).

    Resolution order:
      1. Known-good allowlist of models that reliably support json_schema.
      2. OpenRouter-style ``supported_parameters`` containing ``response_format``.
    Anything else (unknown model, local/Ollama without proof) returns False.
    """
    if not model:
        return False
    m = model.lower()

    # 1. Known-good allowlist (extend deliberately; never assume).
    _KNOWN_JSON_SCHEMA_MODELS = (
        "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.5",
        "o1", "o3", "o3-mini", "o4-mini",
        "claude-3-5-sonnet", "claude-3-7-sonnet", "claude-4", "claude-opus-4",
        "claude-sonnet-4", "claude-3-5-haiku",
        "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro",
        "deepseek-chat", "deepseek-v3", "deepseek-v4",
        "qwen2.5-72b", "qwen3-235b", "llama-3.3-70b", "mistral-large",
    )
    if any(m.startswith(k) for k in _KNOWN_JSON_SCHEMA_MODELS):
        return True

    # 2. OpenRouter-style catalog: only confirm when response_format is
    #    explicitly advertised. No field => NOT proven => False.
    try:
        from hermes_cli.models import fetch_openrouter_models
        models = fetch_openrouter_models()
        for item in models:
            if not isinstance(item, dict):
                continue
            if (item.get("id") or "").lower() == m:
                params = item.get("supported_parameters")
                if isinstance(params, list):
                    return "response_format" in params
                return False  # explicit catalog entry, no proof of support
    except Exception:
        pass
    return False


def _emit_redacted_telemetry(*, profile: str, model: str, task_class: str, mismatch: str) -> None:
    """Item A (F001 hardening): record a redacted validation-failure event.

    Wraps ``hermes_cli.kanban_telemetry.record_validation_failure`` so callers
    stay lean. Never raises; telemetry must not affect control flow. The
    provider is intentionally omitted-at-call-site and defaulted to 'unknown'
    here because the decomposer resolves the client lazily and the provider
    slug is not always available without a network round-trip.
    """
    try:
        from hermes_cli.kanban_telemetry import record_validation_failure
        record_validation_failure(
            profile=profile,
            provider="unknown",
            model=model,
            task_class=task_class,
            schema_or_version_mismatch=mismatch,
        )
    except Exception:
        pass


# ROUTING MODEL (Sahil directive, 2026-08-13, Option A):
#  - Work goes to the LEAD profile first; the lead executes and
#    re-delegates specialist sub-work via delegate_task / kanban_create.
#    Lead profiles (remii, octacon, quan, wesker, ceecee, gojo, light) stay
#    IN the decomposer roster — they are the preferred recipients.
#  - sirvir (model-fleet manager) receives ONLY tasks about running models
#    locally, reviewing/benchmarking LLM models, or model infrastructure.
#    The decomposer LLM must never be offered sirvir for general work —
#    he stays out of the roster entirely (positive routing to sirvir is a
#    manual/lead decision, not an auto-decompose one).
_DECOMPOSER_DOMAIN_EXCLUDED = frozenset({
    "sirvir",          # model fleet / inference ops — manual routing only
    "market-scanner",  # breadth-first signal scanning only
    "moss",            # upstream vanilla Hermes work only
    "misa-misa",       # voice intake only
    "denji",           # governance only
    "orchestrator",    # virtual name, not a worker
})


def _build_roster() -> tuple[list[dict], set[str]]:
    """Return (roster_for_prompt, valid_assignee_names).

    Each roster entry is ``{name, description, has_description}``. The
    valid-set is used after the LLM responds to rewrite invalid
    assignees to the default fallback.

    Only SPAWNABLE profiles are offered. A profile the dispatcher would
    refuse (a lead profile in ``kanban.nonspawnable_profiles``, or one
    with no directory) must never be handed a child task: the dispatcher
    would bucket it as ``skipped_nonspawnable`` and the task would strand
    in ``ready`` forever. We reuse the dispatcher's own predicate so the
    two can never drift.
    """
    roster: list[dict] = []
    valid: set[str] = set()
    try:
        all_profiles = profiles_mod.list_profiles()
    except Exception as exc:
        logger.warning("decompose: failed to list profiles: %s", exc)
        return roster, valid
    for p in all_profiles:
        try:
            if not kb._is_profile_spawnable(p.name):
                continue
        except Exception:
            # Fail closed (see docstring): exclude rather than risk offering a
            # non-spawnable profile.
            logger.warning(
                "decompose: spawnability check failed for %s, excluding it",
                p.name,
            )
            continue
        # FIX 2026-08-13: infra-only profiles are not decomposer-routable
        # (see _DECOMPOSER_DOMAIN_EXCLUDED below). Excluding them here keeps
        # the LLM from ever being offered a mismatched choice.
        if p.name in _DECOMPOSER_DOMAIN_EXCLUDED:
            continue
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

    FIX 2026-08-13: domain-excluded profiles are removed from the valid set
    — a decomposer LLM choice of an infra-only profile (e.g. sirvir) is
    rewritten to the default assignee instead of being honoured.
    """
    valid_names = valid_names - _DECOMPOSER_DOMAIN_EXCLUDED
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
    # FIX 2026-08-13 (Spectator Mode incident t_9df6f54b): feature-pipeline
    # tasks (pipeline_mode set) are owned by the gated pipeline
    # (research→prd→spec→council...). The auto-decomposer must never fan them
    # out into a workgraph — that hijacks them off the pipeline track.
    # Defense-in-depth alongside the kanban_watchers gate: this also covers
    # direct callers of decompose_task (e.g. the dashboard decompose endpoint).
    pipeline_mode = getattr(task, "pipeline_mode", None)
    if pipeline_mode:
        return DecomposeOutcome(
            task_id, False,
            f"task is pipeline-owned (pipeline_mode={pipeline_mode!r}); "
            "decompose is a pipeline stage, not an auto-decompose target",
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

    # Resolve the task's configured model for the capability probe and
    # telemetry below. call_llm resolves provider+model internally, but the
    # structured-output gate (_aux_supports_json_schema) and the redacted
    # telemetry event still need a model string. We read it from the same
    # auxiliary task config call_llm uses, without constructing a client.
    model: Optional[str] = None
    try:
        from agent.auxiliary_client import _resolve_task_provider_model  # type: ignore
        _rp = _resolve_task_provider_model("kanban_decomposer")
        # Returns (provider, model, base_url, api_key, api_mode); model may be None.
        model = _rp[1]
    except Exception:
        pass

    # Bounded retry: a transient model hiccup (prose, truncated JSON,
    # fenced-but-broken JSON, empty content) must not permanently strand a
    # triage task. We re-call call_llm with the SAME task config (never a
    # different/unauthorised provider, never a silent downgrade) up to
    # DECOMPOSE_MAX_RETRIES + 1 attempts. Each failed attempt carries a
    # sanitised, structured diagnostic — we never echo raw model output.
    DECOMPOSE_MAX_RETRIES = 2
    last_diagnostic = "no attempt made"

    # Item B (F001 hardening): optionally request native structured output.
    # Capability-gated and OFF by default — we only send response_format when
    # (a) config explicitly enables kanban.structured_output AND (b) the
    # resolved aux model is PROVEN to support json_schema. Free-first
    # constraint (2026-07-19): many free models lack json_schema, so an
    # unproven model must fall back to the lenient+retried path — never force
    # it (would cause HTTP 400 and regress the decomposer).
    structured_kwargs: dict = {}
    if _structured_output_enabled(cfg) and _aux_supports_json_schema(model):
        structured_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "kanban_decomposition",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "fanout": {"type": "boolean"},
                        "rationale": {"type": "string"},
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "assignee": {"type": ["string", "null"]},
                        "tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "body": {"type": "string"},
                                    "assignee": {"type": ["string", "null"]},
                                    "parents": {"type": "array", "items": {"type": "integer"}},
                                },
                                "required": ["title", "body", "assignee", "parents"],
                            },
                        },
                    },
                    "required": ["fanout", "rationale"],
                },
            },
        }

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    for attempt in range(DECOMPOSE_MAX_RETRIES + 1):
        if attempt > 0:
            # Structured recovery nudge — tells the model *why* the prior
            # response was rejected without leaking prior raw output.
            messages.append({
                "role": "assistant",
                "content": "<response was not valid JSON and could not be parsed>",
            })
            messages.append({
                "role": "user",
                "content": "Your previous response was not valid JSON. Output ONLY the JSON object, no prose, no code fences.",
            })
        try:
            # Route through call_llm so auxiliary.kanban_decomposer.* config
            # (provider/model/base_url, extra_body, reasoning_effort, retries)
            # all apply — the previous direct client.chat.completions.create()
            # path dropped auxiliary.<task>.extra_body entirely (#35566).
            # Structured-output (response_format) is carried inside extra_body
            # so it propagates through call_llm's _build_call_kwargs merge.
            _call_extra_body = dict(structured_kwargs) or None
            resp = call_llm(
                task="kanban_decomposer",
                messages=messages,
                temperature=0.3,
                max_tokens=4000,
                timeout=timeout or 180,
                extra_body=_call_extra_body,
            )
        except Exception as exc:
            logger.info(
                "decompose: API call failed for %s (attempt %d/%d, %s)",
                task_id, attempt + 1, DECOMPOSE_MAX_RETRIES + 1, type(exc).__name__,
            )
            last_diagnostic = f"LLM error: {type(exc).__name__}"
            continue

        try:
            raw = resp.choices[0].message.content or ""
        except Exception:
            raw = ""

        parsed = _extract_json_blob(raw)
        if parsed is not None:
            break  # success — proceed to fanout/spec handling below
        # Parsed as None: record a sanitised class of failure, retry.
        if not raw.strip():
            last_diagnostic = "LLM returned empty content"
        else:
            last_diagnostic = "LLM returned malformed JSON"
        logger.info(
            "decompose: parse failed for %s (attempt %d/%d, %s)",
            task_id, attempt + 1, DECOMPOSE_MAX_RETRIES + 1, last_diagnostic,
        )
    else:
        # All attempts exhausted without a parseable response.
        # Item A (F001 hardening): redacted telemetry — never raw output.
        _emit_redacted_telemetry(
            profile="kanban_decomposer", model=model,
            task_class="aux_malformed_json" if "malformed" in last_diagnostic else "aux_empty_response",
            mismatch=last_diagnostic,
        )
        return DecomposeOutcome(
            task_id, False, f"decomposer failed after {DECOMPOSE_MAX_RETRIES + 1} attempts: {last_diagnostic}"
        )

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
