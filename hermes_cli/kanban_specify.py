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

logger = logging.getLogger(__name__)

# Hard bound on the task body handed to the specifier prompt. The model never
# sees more than this many characters of the body; ``specify_task`` refuses a
# longer body unless ``force`` is set (the refusal writes nothing), and on the
# forced path the unseen tail is appended verbatim to the returned body.
PROMPT_BODY_LIMIT = 4000


_SYSTEM_PROMPT = """You are the Kanban triage specifier for the Hermes Agent board.
A user dropped a rough idea into the Triage column. Your job is to turn it
into a concrete, actionable task spec that an autonomous worker can pick up
and execute without further clarification.

Output a single JSON object with exactly two keys:

  {
    "title": "<tightened task title, <= 80 chars, imperative voice>",
    "body":  "<multi-line spec, see structure below>"
  }

The body MUST include these sections, each prefixed with a bold markdown
heading, in this order:

  **Goal** — one sentence, user-facing outcome.
  **Approach** — 2-5 bullets on how a worker should tackle it.
  **Acceptance criteria** — checklist of concrete, verifiable conditions.
  **Out of scope** — short list of things NOT to touch (omit if nothing
      obvious; never invent scope creep).

Rules:
  - Keep the tightened title close in meaning to the original idea — do
    NOT invent a different project.
  - If the original idea is already detailed, preserve its substance and
    just reformat into the sections above.
  - Never add invented requirements the user didn't hint at.
  - No preamble, no closing remarks, no code fences around the JSON.
  - Output only the JSON object and nothing else.
"""


_USER_TEMPLATE = """Task id: {task_id}
Current title: {title}
Current body:
{body}
"""


@dataclass
class SpecifyOutcome:
    """Result of specifying a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    new_title: Optional[str] = None
    # True only when the stored body carries an appended verbatim tail because
    # the specifier prompt only saw the first ``PROMPT_BODY_LIMIT`` chars.
    body_truncated: bool = False
    # Real length of the task body as read (0 when unknown / never loaded).
    body_length: int = 0
    prompt_body_limit: int = PROMPT_BODY_LIMIT


def _truncate(text: str, limit: int) -> str:
    # Plain length clamp for LLM prompt fields; these never reach a terminal, so no escape stripping here.
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


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


def _body_over_bound(body: Optional[str]) -> tuple[int, bool]:
    """``(len(body or ""), len > PROMPT_BODY_LIMIT)`` — the body-length guard
    shared by ``specify_task`` and the sibling decompose fix."""
    length = len(body or "")
    return length, length > PROMPT_BODY_LIMIT


def _with_unseen_tail(new_body: str, original: str, *, actor: str = "specifier") -> str:
    """Append the tail the model never saw (the prompt clamped at
    ``PROMPT_BODY_LIMIT - 1`` + '…'), verbatim, for the forced path."""
    return (
        new_body
        + "\n\n---\n\n## Original body tail (verbatim; the {} saw only the first {} chars)\n\n".format(actor, PROMPT_BODY_LIMIT - 1)
        + (original or "")[PROMPT_BODY_LIMIT - 1:]
    )


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
    """Bounded ``task_id``/``title``/``body`` for the user prompt templates."""
    return {
        "task_id": task.id,
        "title": _truncate(task.title or "", 400),
        "body": _truncate(task.body or "(no body)", PROMPT_BODY_LIMIT),
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
    force: bool = False,
) -> SpecifyOutcome:
    """Specify one triage task and promote it to ``todo``. Expected failures
    (not in triage, no aux client, API error, malformed reply) surface as
    ``ok=False`` so an ``--all`` sweep continues.

    A body longer than ``PROMPT_BODY_LIMIT`` is refused without any DB write
    unless ``force`` is set — the specifier prompt only ever sees the first
    ``PROMPT_BODY_LIMIT`` characters, so silently replacing the body would drop
    the unseen tail. ``force`` proceeds with the truncated prompt and appends
    the unseen tail verbatim to the returned body; on the forced path a
    malformed reply promotes the card unchanged rather than writing a body the
    model never fully saw.
    """
    task, reason = _load_triage_task(task_id)
    if task is None:
        return SpecifyOutcome(task_id, False, reason)

    body_len, over = _body_over_bound(task.body)
    if over and not force:
        return SpecifyOutcome(
            task_id, False,
            f"task body is {body_len} chars, over PROMPT_BODY_LIMIT="
            f"{PROMPT_BODY_LIMIT} (the specifier would only see the first "
            f"{PROMPT_BODY_LIMIT} chars); pass --force to specify anyway (the "
            f"unseen tail is appended verbatim to the result)",
            body_length=body_len, prompt_body_limit=PROMPT_BODY_LIMIT,
        )

    raw, reason = _call_aux(
        "specify", task_id, aux_task="triage_specifier", system=_SYSTEM_PROMPT,
        user=_USER_TEMPLATE.format(**_task_prompt_fields(task)),
        max_tokens=HERMES_KANBAN_SPECIFY_MAX_TOKENS, timeout=timeout or 120,
    )
    if raw is None:
        return SpecifyOutcome(task_id, False, reason)
    raw = raw.strip()

    promote_reason = "specified"
    parsed = _extract_json_blob(raw)
    if parsed is None:
        # Whole reply becomes the body; the user can edit afterward.
        if not raw:
            return SpecifyOutcome(task_id, False, "LLM returned an empty response")
        if over:
            # Never overwrite a body the model only partly saw with the raw
            # reply: promote the card unchanged instead.
            new_title, new_body = None, None
            promote_reason = "promoted unchanged: reply malformed"
        else:
            new_title, new_body = None, raw
    else:
        new_title, new_body = _title_body(parsed)
        if new_body is None and new_title is None:
            return SpecifyOutcome(task_id, False, "LLM response missing title and body")

    # Forced path with a fresh body: re-attach the tail the model never saw.
    # _truncate cuts at PROMPT_BODY_LIMIT - 1, so the unseen complement starts
    # there (slicing at PROMPT_BODY_LIMIT would drop one character).
    body_truncated = False
    if over and new_body is not None:
        new_body = _with_unseen_tail(new_body, task.body or "")
        body_truncated = True

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
    return SpecifyOutcome(
        task_id, True, promote_reason, new_title=new_title,
        body_truncated=body_truncated, body_length=body_len,
        prompt_body_limit=PROMPT_BODY_LIMIT,
    )


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Task ids in the triage column; ``tenant`` narrows the sweep."""
    with kbc.connect_closing() as conn:
        tasks = kb.list_tasks(conn, status="triage", tenant=tenant, include_archived=False)
    return [t.id for t in tasks]
