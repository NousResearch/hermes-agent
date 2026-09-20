"""Kanban triage router — a fast/cheap Jev pre-check ahead of the full
``hermes kanban specify`` LLM call.

See docs/superpowers/specs/2026-09-20-triage-jev-router-design.md.

Off by default: ``auxiliary.triage_router.model`` must be set explicitly.
Every failure mode (unconfigured, over the size threshold, timeout, error,
any reply other than a clean "trivial") falls through to the caller's
existing full-specify path — this module never blocks or slows down that
path, it only sometimes lets the caller skip it.
"""

from __future__ import annotations

import logging
from typing import Optional

from hermes_cli import kanban_db as kb

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BODY_CHARS = 300

_ROUTER_SYSTEM_PROMPT = """You are a triage router for the Hermes Agent Kanban board.
Decide whether a rough task idea is trivial enough to auto-promote with a
minimal spec, or needs the full Goal/Approach/Acceptance-criteria treatment.

Respond with exactly one word, nothing else:
  TRIVIAL     - a small, self-contained, unambiguous change (e.g. fix a typo,
                bump a version string, update one config value).
  NEEDS_SHAPE - anything with real design decisions, multiple files/systems
                involved, or unclear scope.
  UNSURE      - you are not confident either way.

When in doubt, answer UNSURE or NEEDS_SHAPE, never TRIVIAL."""

_ROUTER_USER_TEMPLATE = """Task id: {task_id}
Title: {title}
Body:
{body}
"""

# Reply -> verdict. Matched against resp.content.strip().upper(); anything
# not a key here (blank, multi-word, punctuation, unparseable) is None.
_VERDICTS = {"TRIVIAL": "trivial", "NEEDS_SHAPE": "needs_shape", "UNSURE": "unsure"}


def _router_task_config() -> dict:
    from agent.auxiliary_client import _get_auxiliary_task_config
    return _get_auxiliary_task_config("triage_router")


def configured_model() -> Optional[str]:
    """``auxiliary.triage_router.model``, or ``None`` when unset/blank."""
    model = _router_task_config().get("model")
    model = str(model).strip() if model else ""
    return model or None


def router_configured() -> bool:
    """The router is opt-in: it never runs unless a model is explicitly set."""
    return configured_model() is not None


def _max_body_chars() -> int:
    raw = _router_task_config().get("max_body_chars")
    try:
        return int(raw) if raw is not None else _DEFAULT_MAX_BODY_CHARS
    except (TypeError, ValueError):
        return _DEFAULT_MAX_BODY_CHARS


def eligible(task: "kb.Task") -> bool:
    """Hard length/complexity pre-filter, independent of what Jev says —
    bounds the blast radius of a wrong verdict to tasks that were already
    short. Jev narrows *within* this set; it never expands it."""
    length = len(task.title or "") + len(task.body or "")
    return length <= _max_body_chars()


def _ask_jev(task: "kb.Task") -> Optional[str]:
    """One classification call; the normalized verdict string, or ``None``
    on any failure (import, timeout, API error, empty/unparseable reply)."""
    try:
        from agent.auxiliary_client import _get_task_timeout, call_llm
    except Exception as exc:  # pragma: no cover - import smoke test
        logger.debug("triage router: auxiliary client import failed: %s", exc)
        return None
    try:
        resp = call_llm(
            task="triage_router",
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": _ROUTER_USER_TEMPLATE.format(
                    task_id=task.id, title=task.title or "", body=task.body or "(no body)",
                )},
            ],
            temperature=0, max_tokens=8,
            timeout=_get_task_timeout("triage_router"),
        )
    except Exception as exc:
        logger.info(
            "triage router: API call failed for %s (%s) — falling through to full specify",
            task.id, exc,
        )
        return None
    try:
        raw = (resp.choices[0].message.content or "").strip().upper()
    except Exception:
        return None
    return _VERDICTS.get(raw)


def is_trivial(task: "kb.Task") -> bool:
    """True only when the task is short enough AND Jev cleanly answers
    "trivial". Every other outcome is False, which callers treat as "fall
    through to the full specify path" — never a distinct error case."""
    if not router_configured():
        return False
    if not eligible(task):
        return False
    return _ask_jev(task) == "trivial"


_MINIMAL_APPROACH = (
    "- Make the smallest change that satisfies the goal above.\n"
    "- If the change turns out to need touching more files/systems than the "
    "title implies, stop and request-changes/comment instead of expanding scope."
)
_MINIMAL_ACCEPTANCE = (
    "- [ ] The change matches the stated goal.\n"
    "- [ ] If the actual scope turned out larger than the one-liner implied, "
    "this was flagged via request-changes/comment rather than silently expanded."
)


def build_minimal_spec(task: "kb.Task") -> tuple[str, str]:
    """``(title, body)`` for an auto-promoted task — template-filled from the
    raw title, no additional LLM call. Title truncation mirrors the 80-char
    limit ``kanban_specify.py``'s full LLM path enforces."""
    title = (task.title or "").strip()
    if len(title) > 80:
        title = title[:79] + "…"
    body = (
        f"**Goal**\n{task.title or ''}\n\n"
        f"**Approach**\n{_MINIMAL_APPROACH}\n\n"
        f"**Acceptance criteria**\n{_MINIMAL_ACCEPTANCE}\n"
    )
    return title, body
