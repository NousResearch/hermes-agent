"""web_gemini worker: executes ledger-bound actions parsed from a Gemini reply.

The conversation half (``hermes_cli.web_gemini``) binds a Gemini conversation to
a kanban task and parses its replies. This module is the executing half: it
re-validates the execution scope, scrubs the owner environment so a browser
session can never fall back to an API-key model, trims Gemini UI chrome from
raw payloads, and applies parsed actions through :class:`ActionLedger` so every
board write is intent-durable, idempotent, and replay-refusing.

Execution scope is explicit: a task opts into this worker via
``execution_scope="web_gemini"`` at creation. The scope is never inferred from
an assignee or profile alias, and ``web_gemini`` is deliberately not a valid
profile name (the dispatcher routes the lane without profile resolution).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_cli.web_gemini import (
    MAX_ACTIONS,
    ActionLedger,
    ParsedResponse,
    ProtocolError,
)

WEB_GEMINI_SCOPE = "web_gemini"

# Keys scrubbed from the owner environment: the web_gemini worker must act only
# through the bound Gemini web conversation, never by silently degrading to an
# API-key provider/model.
_SCRUBBED_ENV_KEYS = frozenset(
    {
        "OPENROUTER_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "HERMES_MODEL",
        "HERMES_PROVIDER",
    }
)


def validate_execution_scope(scope: Optional[str]) -> str:
    """Require the explicit ``web_gemini`` execution scope.

    Profile aliases (``default``, ``gemini``, ``coder``, ...) and empty values
    are rejected: this worker only ever executes for tasks explicitly created
    with ``execution_scope="web_gemini"``.
    """
    if scope is None:
        raise ProtocolError("execution_scope is required for web_gemini actions")
    normalized = str(scope).strip().lower()
    if not normalized:
        raise ProtocolError("execution_scope is required for web_gemini actions")
    if normalized != WEB_GEMINI_SCOPE:
        raise ProtocolError(
            f"execution_scope must be the explicit {WEB_GEMINI_SCOPE!r} scope, "
            f"got {scope!r} (profile aliases are not accepted)"
        )
    return normalized


def build_owner_env(env: Mapping[str, str], *, profile_dir: Optional[Path] = None) -> dict:
    """Build the browser-owner child environment for a web_gemini session.

    Scrubs model/provider API keys and overrides so the browser session cannot
    fall back to an API-key model, and stamps :data:`WEB_GEMINI_SCOPE` so the
    child can re-assert its scope. ``profile_dir`` isolates the browser profile
    away from the user's real one when provided.
    """
    child: dict[str, str] = {k: v for k, v in env.items() if k not in _SCRUBBED_ENV_KEYS}
    child["WEB_GEMINI_SCOPE"] = WEB_GEMINI_SCOPE
    if profile_dir is not None:
        child["HERMES_WEB_GEMINI_PROFILE_DIR"] = str(profile_dir)
    return child


def _trim_gemini_chrome(payload: str) -> str:
    """Trim Gemini UI chrome (footer/status snippets) from a raw reply.

    The web UI appends trailing boilerplate (model-status lines, privacy
    notices, "Gemini replied" markers) after the model's payload. Only
    *known* chrome fragments are removed — arbitrary model prose after the
    payload is preserved so nothing meaningful is silently dropped.
    """
    if not payload:
        return payload
    text = payload.rstrip()
    chrome_markers = (
        "Gemini replied",
        "Gemini is AI and can make mistakes",
        "Your privacy & Gemini",
        "chats aren\u2019t used to improve our models",
        "chats aren't used to improve our models",
        "Check your internet connection and try again",
        "Opens in a new window",
        "Gemini Opens in a new window",
        "Flash Your n-gineers.com",
    )
    changed = True
    while changed:
        changed = False
        for marker in chrome_markers:
            idx = text.rfind(marker)
            if idx != -1:
                candidate = text[:idx].rstrip()
                # Removing a known fragment that leaves nothing or shortens is
                # fine; keep looping so stacked chrome collapses.
                if candidate != text:
                    text = candidate
                    changed = True
    return text


def _action_mode_for_task(task: Any) -> str:
    """The task's web_gemini action mode; legacy/unset rows default to ``full``."""
    mode = getattr(task, "web_gemini_action_mode", None)
    if not mode:
        return "full"
    return str(mode)


def execute_actions(
    conn,
    ledger: ActionLedger,
    parsed: ParsedResponse,
    *,
    author: str = "web_gemini",
    comment_only: bool = False,
) -> list:
    """Apply parsed actions through the ledger, bound to the parsed binding.

    Every action goes intent -> side effect -> applied -> finalized, so a crash
    between the external write and the board update is a visible blocker rather
    than a silent duplication. Under ``comment_only`` (or a ``comment_only``
    task), terminal actions and multi-action batches are rejected before any
    side effect.

    ``conn`` is the kanban DB connection; the ledger keeps its own connection.
    """
    from hermes_cli import kanban_db as kb

    binding = parsed.binding
    validate_execution_scope(getattr(binding, "execution_scope", None) or WEB_GEMINI_SCOPE)

    task = kb.get_task(conn, binding.task_id)
    if task is None:
        raise ProtocolError(f"unknown task for binding: {binding.task_id}")
    mode = _action_mode_for_task(task)
    if comment_only or mode == "comment_only":
        if len(parsed.actions) > 1:
            raise ProtocolError("comment_only tasks reject multi-action batches")
        for action in parsed.actions:
            if str(action.get("kind")) != "comment":
                raise ProtocolError(
                    f"comment_only tasks reject terminal action {action.get('kind')!r}"
                )
    if len(parsed.actions) > MAX_ACTIONS:
        raise ProtocolError(f"response exceeds MAX_ACTIONS={MAX_ACTIONS}")

    results = []
    for action in parsed.actions:
        index = int(action.get("index", 0))
        # The parsed actions already validated index/kind against the binding;
        # derive the per-action binding copy carrying this index.
        from dataclasses import replace

        action_binding = replace(binding, action_index=index)
        kind = str(action.get("kind"))
        if kind == "comment":
            body = str(action.get("body", "")).strip()
            if not body:
                raise ProtocolError("comment action requires a non-empty body")

            def _apply_comment(action_binding=action_binding, body=body):
                comment_id = kb.add_comment(conn, action_binding.task_id, author, body)
                return {"comment_id": comment_id}

            results.append(ledger.apply_once(action_binding, _apply_comment))
        elif kind == "complete":
            result_text = str(action.get("result", "")).strip()

            def _apply_complete(action_binding=action_binding, result_text=result_text):
                kb.complete_task(conn, action_binding.task_id, result=result_text or None)
                return {"completed": action_binding.task_id}

            results.append(ledger.apply_once(action_binding, _apply_complete))
        elif kind == "block":
            reason = str(action.get("reason", "")).strip() or None

            def _apply_block(action_binding=action_binding, reason=reason):
                kb.block_task(conn, action_binding.task_id, reason=reason)
                return {"blocked": action_binding.task_id, "reason": reason}

            results.append(ledger.apply_once(action_binding, _apply_block))
        else:
            raise ProtocolError(f"unsupported action kind: {kind!r}")
    return results
