"""Kanban triage specifier — flesh out a one-liner into a real spec.

``hermes kanban specify [task_id | --all]`` asks the auxiliary LLM for a
tightened title + concrete body for a Triage task, then flips it
``triage -> todo`` via ``kanban_db.specify_triage_task``.

Mirrors ``hermes_cli/goals.py``: same aux-client pattern, same "empty config
=> skip, don't crash" tolerance. One shot, no retry loop. JSON mode is not
requested (works on providers without it); the parse is lenient and falls
back to "whole reply is the body" so a malformed reply never strands a task.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

from utils import env_int

HERMES_KANBAN_SPECIFY_MAX_TOKENS = max(1500, env_int("HERMES_KANBAN_SPECIFY_MAX_TOKENS", 6000))

# The specifier must SEE the whole brief to preserve it: real triage cards on
# the shared board have been authored at 8 KB - 49 KB, and a 4,000-char clamp
# silently halved every long brief before the model ever saw it (the model
# then faithfully rewrote the half it was given; the other half was gone from
# the card). The defaults below are far above any realistic card; an operator
# running a small-context auxiliary model can lower them via env. When a bound
# DOES bite, the user prompt carries an explicit truncation marker (see
# ``_task_prompt_fields``) so the model knows the brief is partial instead of
# silently summarising half of it.
HERMES_KANBAN_SPECIFY_MAX_BODY_CHARS = max(1, env_int("HERMES_KANBAN_SPECIFY_MAX_BODY_CHARS", 100_000))
HERMES_KANBAN_SPECIFY_MAX_TITLE_CHARS = max(1, env_int("HERMES_KANBAN_SPECIFY_MAX_TITLE_CHARS", 2_000))

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are the Kanban triage specifier for the Hermes Agent board.
A user dropped an idea into the Triage column. Your job is to make sure the
task is concrete, actionable, and executable by an autonomous worker without
further clarification - WITHOUT destroying anything the author already wrote.

Output a single JSON object with exactly two keys:

  {
    "title": "<tightened task title, <= 80 chars, imperative voice>",
    "body":  "<the spec>"
  }

How to build the body - preserving detail is the priority:

  - If the incoming body is ALREADY a complete or detailed spec (it names
    files, paths, ids, measurements, acceptance criteria, steps, or
    decisions), it is the deliverable. Keep ALL of its text: copy it into
    your reply essentially verbatim, preserving every anchor, id, path,
    number, and list. Do not summarise it, do not reshape it into a
    different structure, and do not drop anything. You may only ADD what is
    genuinely missing. Prefer keeping the author's structure; add the
    sections below only where they are absent, using bold markdown headings:

      **Goal** - one sentence, user-facing outcome.
      **Approach** - 2-5 bullets on how a worker should tackle it.
      **Acceptance criteria** - checklist of concrete, verifiable conditions.
      **Out of scope** - short list of things NOT to touch.

  - If the incoming body is a rough one-liner or vague idea, restructure it
    into the sections above, inventing nothing the author did not imply.

Rules:
  - When in doubt, keep the original wording over a rewrite. A longer body
    that preserves the author's meaning beats a shorter, cleaner-looking one.
  - Keep the tightened title close in meaning to the original idea - do
    NOT invent a different project.
  - Never add invented requirements the user didn't hint at.
  - No preamble, no closing remarks, no code fences around the JSON.
  - Output only the JSON object and nothing else.
"""


_USER_TEMPLATE = """Task id: {task_id}
Current title: {title}
Current body:
{body}
"""


# Marker the model sees when the body/title above was clipped before it got
# here. Explicit and unmissable: a silent cap is what this replaces.
_TRUNCATION_MARKER = "[... TRUNCATED BY THE BOARD: this field is {total} chars and only the first {kept} are shown above. The rest is NOT included - base your rewrite only on the text you can see and keep everything shown, in full. The full original text is preserved on the card in a comment.]"


@dataclass
class SpecifyOutcome:
    """Result of specifying a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    new_title: Optional[str] = None


def _clamp_field(text: str, limit: int) -> str:
    """Clamp a prompt field with an explicit, unmissable marker.

    Unlike a bare ellipsis (only suitable for short human-facing previews),
    the marker states in the prompt that the brief is partial - the silent
    half-brief is what let the specifier rewrite away the other half. The
    full original body survives on the card regardless (see
    ``kb.specify_triage_task``, which snapshots it as a comment).
    """
    if len(text) <= limit:
        return text
    marker = _TRUNCATION_MARKER.format(total=len(text), kept=limit)
    return text[:limit] + "\n\n" + marker


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_json_blob(raw: str, fence_re: re.Pattern = _FENCE_RE) -> Optional[dict]:
    """Lenient JSON object extraction: strip code fences, take the first ``{``
    to the last ``}``. None if nothing parses to a dict."""
    if not raw:
        return None
    stripped = fence_re.sub("", raw.strip())
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        val = json.loads(stripped[first : last + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    return val if isinstance(val, dict) else None


def _nonblank(v) -> Optional[str]:
    return v if isinstance(v, str) and v.strip() else None


def _title_body(parsed: dict) -> tuple[Optional[str], Optional[str]]:
    """``(title, body)`` from an LLM reply: title stripped, body verbatim,
    either None when missing/blank."""
    title = _nonblank(parsed.get("title"))
    return (title.strip() if title else None), _nonblank(parsed.get("body"))


def _profile_author(default: str = "specifier") -> str:
    """Same identity contract as ``hermes_cli.kanban._profile_author``; ``$USER`` as the last
    resort for a human running the CLI outside any profile."""
    from hermes_cli.profiles import current_profile_name
    return current_profile_name() or os.environ.get("USER") or default


def _load_triage_task(task_id: str) -> tuple[Optional[kb.Task], str]:
    """``(task, "")`` when the task exists and is in triage, else ``(None, reason)``."""
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    if task is None:
        return None, "unknown task id"
    if task.status != "triage":
        return None, f"task is not in triage (status={task.status!r})"
    return task, ""


def _task_prompt_fields(task: kb.Task) -> dict[str, str]:
    """Bounded ``task_id``/``title``/``body`` for the user prompt templates.

    The bounds are deliberately far above any realistic card (100 KB body /
    2 KB title, env-overridable) because the specifier's whole job is to
    preserve what it sees. A brief longer than the bound still arrives whole
    unless it is truly enormous, and when a clamp does bite the prompt carries
    an explicit ``TRUNCATED BY THE BOARD`` marker (``_clamp_field``) instead of
    silently shipping half the brief.
    """
    return {
        "task_id": task.id,
        "title": _clamp_field(task.title or "", HERMES_KANBAN_SPECIFY_MAX_TITLE_CHARS),
        "body": _clamp_field(
            task.body or "(no body)", HERMES_KANBAN_SPECIFY_MAX_BODY_CHARS
        ),
    }


def _call_aux(verb: str, task_id: str, *, aux_task: str, system: str, user: str,
              max_tokens: int, timeout: int, log: logging.Logger = logger) -> tuple[Optional[str], str]:
    """One auxiliary LLM call; ``(reply_text, "")`` or ``(None, reason)``.

    ``call_llm`` applies all ``auxiliary.<aux_task>.*`` config (provider/model/
    base_url, extra_body, reasoning_effort, retries). Imported lazily so a
    missing aux client degrades to a skip instead of an import-time crash.
    """
    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:  # pragma: no cover — import smoke test
        log.debug("%s: auxiliary client import failed: %s", verb, exc)
        return None, "auxiliary client unavailable"
    # Specify/decompose run outside any agent turn (CLI, dashboard route, gateway watcher), so no
    # conversation affinity scope is bound and the relay-affinity headers (x-opencode-session, the
    # OpenRouter/Portal sticky key) are omitted — the OpenCode Go relay rejects that with 400
    # MissingSessionID (#112043). Declare a per-task scope, but only when none is already bound so an
    # in-turn caller keeps its conversation's key.
    from agent.portal_tags import get_affinity_scope, reset_affinity_scope, set_affinity_scope
    affinity_token = None if get_affinity_scope() else set_affinity_scope(f"kanban:{task_id}")
    try:
        # Route through call_llm so auxiliary.triage_specifier.* config (provider/model/base_url,
        # extra_body, reasoning_effort, retries) all apply — the direct-create path dropped extra_body
        # (#35566).
        resp = call_llm(
            task=aux_task,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except Exception as exc:
        suffix = " — skipping" if verb == "specify" else ""
        log.info("%s: API call failed for %s (%s)%s", verb, task_id, exc, suffix)
        return None, f"LLM error: {type(exc).__name__}"
    finally:
        if affinity_token is not None:
            reset_affinity_scope(affinity_token)
    try:
        return resp.choices[0].message.content or "", ""
    except Exception:
        return "", ""


def specify_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
) -> SpecifyOutcome:
    """Specify one triage task and promote it to ``todo``. Expected failures
    (not in triage, no aux client, API error, malformed reply) surface as
    ``ok=False`` so an ``--all`` sweep continues."""
    task, reason = _load_triage_task(task_id)
    if task is None:
        return SpecifyOutcome(task_id, False, reason)

    raw, reason = _call_aux(
        "specify", task_id, aux_task="triage_specifier", system=_SYSTEM_PROMPT,
        user=_USER_TEMPLATE.format(**_task_prompt_fields(task)),
        max_tokens=HERMES_KANBAN_SPECIFY_MAX_TOKENS, timeout=timeout or 120,
    )
    if raw is None:
        return SpecifyOutcome(task_id, False, reason)
    raw = raw.strip()

    parsed = _extract_json_blob(raw)
    if parsed is None:
        # Whole reply becomes the body; the user can edit afterward.
        if not raw:
            return SpecifyOutcome(task_id, False, "LLM returned an empty response")
        new_title, new_body = None, raw
    else:
        new_title, new_body = _title_body(parsed)
        if new_body is None and new_title is None:
            return SpecifyOutcome(task_id, False, "LLM response missing title and body")

    with kbc.connect_closing() as conn:
        ok = kb.specify_triage_task(
            conn,
            task_id,
            title=new_title,
            body=new_body,
            author=author or _profile_author(),
        )
    if not ok:
        # Race: promoted/archived between our read and the write.
        return SpecifyOutcome(task_id, False, "task moved out of triage before promotion")
    return SpecifyOutcome(task_id, True, "specified", new_title=new_title)


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Task ids in the triage column; ``tenant`` narrows the sweep."""
    with kbc.connect_closing() as conn:
        tasks = kb.list_tasks(conn, status="triage", tenant=tenant, include_archived=False)
    return [t.id for t in tasks]
