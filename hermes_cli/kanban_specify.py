"""Kanban triage specifier — flesh out a one-liner into a real spec.

Used by ``hermes kanban specify [task_id | --all]``. Takes a task that
lives in the Triage column (a rough idea, typically only a title), calls
the auxiliary LLM to produce:

  * A tightened title (optional — only replaces if the model proposes a
    materially different one)
  * A concrete body: goal, proposed approach, acceptance criteria

and then flips the task ``triage -> todo`` via
``kanban_db.specify_triage_task``. The dispatcher promotes it to
``ready`` on its next tick (or immediately if there are no open parents).

Design notes
------------

* This module intentionally mirrors ``hermes_cli/goals.py`` — same aux
  client pattern, same "empty config => skip, don't crash" tolerance.
  Keeps the surface area tiny and the failure modes predictable.

* The prompt is a short system + user pair. We ask for JSON with
  ``{title, body}``; if parsing fails, we fall back to treating the
  whole response as the body and leave the title untouched. No
  retry loop — one shot, keep cost bounded.

* Structured output / JSON mode is not requested explicitly so the
  specifier works on providers that don't implement it. The parse
  is lenient (tolerates markdown code fences around the JSON).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli.config import load_config

HERMES_KANBAN_SPECIFY_MAX_TOKENS = max(
    1500,
    int(os.getenv("HERMES_KANBAN_SPECIFY_MAX_TOKENS", "6000")),
)

logger = logging.getLogger(__name__)


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


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_json_blob(raw: str) -> Optional[dict]:
    """Lenient JSON extraction — tolerates fenced code blocks and
    leading/trailing whitespace. Returns None if nothing parses."""
    if not raw:
        return None
    stripped = _FENCE_RE.sub("", raw.strip())
    # Greedy: find the first `{` and last `}` and try that slice.
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
    """Mirror of ``hermes_cli.kanban._profile_author``. Kept local to
    avoid a circular import when kanban.py imports this module."""
    return (
        os.environ.get("HERMES_PROFILE")
        or os.environ.get("USER")
        or "specifier"
    )


def _specifier_uses_lmstudio() -> bool:
    """Return True when the triage specifier routes to LM Studio.

    ``get_text_auxiliary_client("triage_specifier")`` resolves ``auto`` to the
    main chat provider before returning the OpenAI-compatible client, so the
    returned client alone does not tell us whether the configured provider was
    LM Studio.  Check the same config knobs here so we can send LM Studio's
    top-level ``reasoning_effort`` override only when it is safe to do so.
    """
    try:
        cfg = load_config()
    except Exception:
        return False
    aux = (cfg.get("auxiliary") or {}).get("triage_specifier") or {}
    model_cfg = cfg.get("model") or {}
    provider = str(aux.get("provider") or "auto").strip().lower()
    if provider in {"", "auto", "main"}:
        provider = str(model_cfg.get("provider") or "").strip().lower()
    if provider == "lmstudio":
        return True
    base_url = str(aux.get("base_url") or model_cfg.get("base_url") or "").lower()
    return "127.0.0.1:1234" in base_url or "localhost:1234" in base_url


def specify_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
) -> SpecifyOutcome:
    """Specify a single triage task and promote it to ``todo``.

    Returns an outcome describing what happened. Never raises for expected
    failure modes (task not in triage, no aux client configured, API
    error, malformed response) — those surface via ``ok=False`` so the
    ``--all`` sweep can continue past individual failures.
    """
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
    if task is None:
        return SpecifyOutcome(task_id, False, "unknown task id")
    if task.status != "triage":
        return SpecifyOutcome(
            task_id, False, f"task is not in triage (status={task.status!r})"
        )

    try:
        from agent.auxiliary_client import get_auxiliary_extra_body
        from hermes_cli.kanban_auxiliary import (
            kanban_auxiliary_timeout,
            kanban_card_auxiliary_client,
        )
    except Exception as exc:  # pragma: no cover — import smoke test
        logger.debug("specify: auxiliary client import failed: %s", exc)
        return SpecifyOutcome(task_id, False, "auxiliary client unavailable")

    with kanban_card_auxiliary_client(task_id, "triage_specifier") as (client, model):
        if client is None or not model:
            return SpecifyOutcome(
                task_id, False, "no auxiliary client configured"
            )

        user_msg = _USER_TEMPLATE.format(
            task_id=task.id,
            title=_truncate(task.title or "", 400),
            body=_truncate(task.body or "(no body)", 4000),
        )

        request_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.3,
            "max_tokens": HERMES_KANBAN_SPECIFY_MAX_TOKENS,
            "timeout": kanban_auxiliary_timeout(
                "triage_specifier",
                client,
                explicit=timeout,
                default=180.0,
            ),
            "extra_body": get_auxiliary_extra_body() or None,
        }
        if _specifier_uses_lmstudio():
            # LM Studio thinking models can spend the whole completion budget in
            # reasoning_content and leave message.content empty. Specify needs the
            # final JSON in content, so disable reasoning for this formatting step.
            request_kwargs["reasoning_effort"] = "none"

        from hermes_cli.kanban_worker_log import emit_task_worker_log, task_worker_log

        try:
            with task_worker_log(task_id, operation="specify", model=model or ""):
                emit_task_worker_log("Calling auxiliary LLM…\n")
                resp = client.chat.completions.create(**request_kwargs)
                try:
                    raw = (resp.choices[0].message.content or "").strip()
                except Exception:
                    raw = ""
                try:
                    from agent.cursor_auxiliary_client import CursorAuxiliaryClient

                    streamed = isinstance(client, CursorAuxiliaryClient)
                except ImportError:
                    streamed = False
                if not streamed:
                    emit_task_worker_log("\n--- LLM response ---\n")
                    if raw:
                        emit_task_worker_log(raw)
                        if not raw.endswith("\n"):
                            emit_task_worker_log("\n")
        except Exception as exc:
            logger.info(
                "specify: API call failed for %s (%s) — skipping",
                task_id, exc,
            )
            detail = str(exc).strip() or type(exc).__name__
            return SpecifyOutcome(
                task_id, False, f"LLM error: {detail}"
            )

    parsed = _extract_json_blob(raw)

    new_title: Optional[str]
    new_body: Optional[str]
    if parsed is None:
        # Fall back: treat the whole reply as the body, leave title as-is.
        # Worst case the user edits afterward — still better than stranding
        # the task in triage on a malformed LLM reply.
        stripped_raw = raw.strip()
        if not stripped_raw:
            return SpecifyOutcome(
                task_id, False, "LLM returned an empty response"
            )
        new_title = None
        new_body = stripped_raw
    else:
        title_val = parsed.get("title")
        body_val = parsed.get("body")
        new_title = (
            title_val.strip()
            if isinstance(title_val, str) and title_val.strip()
            else None
        )
        new_body = (
            body_val if isinstance(body_val, str) and body_val.strip() else None
        )
        if new_body is None and new_title is None:
            return SpecifyOutcome(
                task_id, False, "LLM response missing title and body"
            )

    with kb.connect() as conn:
        ok = kb.specify_triage_task(
            conn,
            task_id,
            title=new_title,
            body=new_body,
            author=author or _profile_author(),
        )
    if not ok:
        # Race: someone else promoted / archived the task between our
        # read above and the write. Report, don't crash.
        return SpecifyOutcome(
            task_id, False, "task moved out of triage before promotion"
        )
    return SpecifyOutcome(task_id, True, "specified", new_title=new_title)


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Return task ids currently in the triage column.

    ``tenant`` narrows the sweep; ``None`` returns every triage task.
    """
    with kb.connect() as conn:
        tasks = kb.list_tasks(
            conn,
            status="triage",
            tenant=tenant,
            include_archived=False,
        )
    return [t.id for t in tasks]
