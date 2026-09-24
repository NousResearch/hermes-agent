"""The worker outcome plugin: a task whose model call failed ends *blocked, with the reason*.

Copied verbatim into each agent's profile as ``plugins/nova-outcome/__init__.py``, with a
verbatim copy of :mod:`nova.runtime.model_errors` beside it as ``_model_errors.py``.

**The defect it closes.** A dispatched worker runs ``hermes chat -q`` in display mode, and
that path exits 0 whatever the turn did (``cli.py::_run_single_query_mode`` returns after
``cli.chat``; only the quiet ``-Q`` path turns a failed turn into a non-zero exit). So when the
model refuses every call — Bedrock's ``NOT_AUTHORIZED`` on a live deployment — the worker
exits cleanly without calling ``kanban_block``, and the dispatcher books a *protocol
violation*: "the work probably succeeded, only the paperwork was skipped". It re-runs the
task with that advice, three times, and then blocks it with the same words. The provider's
actual error is in no field of the task. On the live board, seven runs across two tasks
said nothing but "protocol violation".

**What it does.** It watches the runtime's own model-call hooks and, when the process ends
with the last model call failed and the task still running *for this run*:

- a failure a person has to fix (access, credentials, a model that does not exist,
  billing) **blocks the task** with the provider's message and what to do about it, through
  the runtime's own ``block_task`` — the call its goal loop makes. The dispatcher then leaves
  it alone instead of spending retries that cannot succeed;
- a failure that time may fix (throttling, an overloaded or unreachable provider) is left to
  the dispatcher's retry budget, with the real error **added as a comment** so the retry
  worker and the operator see the cause instead of "protocol violation".

Nothing here decides whether a *tool* may run — that is the policy plugin's job. This one
only makes sure a run that failed says why.
"""

from __future__ import annotations

import atexit
import os
from typing import Any, Dict, Optional

try:  # installed layout
    from ._model_errors import classify_model_error
except ImportError:  # in-tree layout, for tests that import this module directly
    from nova.runtime.model_errors import classify_model_error

#: The last model-call failure this process saw, cleared by a later success: a fallback
#: model that answered means the run did not fail on the model.
_STATE: Dict[str, Any] = {"error": None, "registered": False}


def _needs_a_person(reason: str, status_code: Optional[int], retryable: Optional[bool]) -> bool:
    """Should this model failure stop the task for a person, rather than be retried?

    ``reason`` is the runtime's classification (``agent/error_classifier.py::FailoverReason``
    values such as ``auth``, ``auth_permanent``, ``billing``, ``rate_limit``, ``overloaded``,
    ``server_error``, ``timeout``, ``model_not_found``, ``content_policy_blocked``,
    ``unknown``), ``status_code`` the provider's HTTP status when there was one, and
    ``retryable`` the runtime's own verdict for the call.
    """
    reason = (reason or "").strip()
    if reason in _NEEDS_A_PERSON:
        return True
    if reason in _TIME_MAY_FIX:
        return False
    # Plain ``auth`` is "refresh and retry" to the runtime, but a provider that answered 401
    # or 403 after that refresh is refusing the account, not the moment — Bedrock's
    # NOT_AUTHORIZED is exactly this, and it did not fix itself in four retries.
    if reason == "auth" or status_code in (401, 402, 403, 404):
        return True
    # Anything else: the runtime's own verdict, and when it has none, retry — a task that
    # would have recovered is worth more than the few runs the retry budget caps.
    return retryable is False


#: Failures no retry of the same request can fix: someone has to change an account, a
#: credential, a model id, a bill, or the prompt itself.
_NEEDS_A_PERSON = frozenset({
    "auth_permanent", "billing", "model_not_found", "provider_policy_blocked",
    "content_policy_blocked", "ssl_cert_verification",
})

#: Failures that clear on their own; the dispatcher's bounded retry is the right answer.
_TIME_MAY_FIX = frozenset({
    "rate_limit", "upstream_rate_limit", "overloaded", "server_error", "timeout",
})


def on_api_request_error(**kwargs: Any) -> None:
    error = kwargs.get("error") or {}
    _STATE["error"] = {
        "message": str(error.get("message") or error.get("type") or "")[:1500],
        "reason": str(kwargs.get("reason") or ""),
        "status_code": kwargs.get("status_code"),
        "retryable": kwargs.get("retryable"),
        "provider": kwargs.get("provider") or "",
        "model": kwargs.get("model") or "",
    }


def on_post_api_request(**_: Any) -> None:
    _STATE["error"] = None


def _outcome_text(error: Dict[str, Any]) -> tuple[str, str]:
    """``(one line for the board, the provider's own words)``."""
    explained = classify_model_error(error["message"])
    where = " ".join(part for part in (error["provider"], error["model"]) if part)
    status = f"HTTP {error['status_code']}, " if error.get("status_code") else ""
    evidence = f"{error['message']} ({status}{error['reason'] or 'unclassified'}{', ' + where if where else ''})"
    return f"{explained.headline}. {explained.remedy}", evidence


def settle() -> Optional[str]:
    """Close out the task if the run ended on a model failure. Returns what it did.

    Runs at process exit. Never raises: the worker is already on its way out, and an
    outcome that cannot be recorded must not turn a clean exit into a crash.
    """
    error = _STATE.get("error")
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if not error or not task_id:
        return None
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        raw_run = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
        run_id = int(raw_run) if raw_run.isdigit() else None
        with kbc.connect_closing() as connection:
            task = kb.get_task(connection, task_id)
            # Only a task this run still holds: one the agent already completed or blocked,
            # or one another run has since claimed, is not ours to change.
            if task is None or task.status != "running":
                return None
            if run_id is not None and getattr(task, "current_run_id", None) not in (None, run_id):
                return None
            summary, evidence = _outcome_text(error)
            if _needs_a_person(error["reason"], error.get("status_code"), error.get("retryable")):
                kb.block_task(
                    connection, task_id,
                    reason=f"The AI model refused the request — {summary} Provider said: {evidence}",
                    kind="needs_input",
                    expected_run_id=run_id,
                )
                return "blocked"
            kb.add_comment(
                connection, task_id, author="nova-outcome",
                body=f"This run ended because the AI model call failed: {evidence}. {summary}",
            )
            return "commented"
    except Exception:  # noqa: BLE001 — see docstring
        return None


def register(ctx: Any) -> None:
    ctx.register_hook("api_request_error", on_api_request_error)
    ctx.register_hook("post_api_request", on_post_api_request)
    # At exit rather than at the end of the turn: a turn that fails early returns before the
    # runtime's session-end hook, and a worker makes exactly one run, so its exit is the
    # one moment every path reaches.
    if os.environ.get("HERMES_KANBAN_TASK") and not _STATE["registered"]:
        _STATE["registered"] = True
        atexit.register(settle)
