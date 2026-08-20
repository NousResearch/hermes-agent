"""Hermes lifecycle dispatch for first-party observers and plugins."""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


def _observe_run_usage(hook_name: str, event: dict[str, Any]) -> None:
    """Persist real runtime usage without making plugins or cards a dependency."""
    if hook_name not in {
        "on_session_start",
        "post_api_request",
        "post_tool_call",
        "api_request_error",
        "on_session_finalize",
    }:
        return
    try:
        import os

        from agent.run_usage_ledger import (
            bind_session,
            current_source_profile,
            current_task_run_id,
            process_invocation_id,
            process_ledger,
            run_id_for_session,
        )

        ledger = process_ledger()
        session_id = str(event.get("session_id") or os.environ.get("HERMES_SESSION_ID") or "") or None
        bind_session(session_id)
        run_id = str(event.get("run_id") or run_id_for_session(session_id))
        task_id = str(event.get("task_id") or os.environ.get("HERMES_KANBAN_TASK") or "") or None
        board = str(event.get("board") or os.environ.get("HERMES_KANBAN_BOARD") or "") or None
        task_run_id = event.get("task_run_id")
        if task_run_id is None:
            task_run_id = current_task_run_id()
        model = str(event.get("model") or "") or None
        provider = str(event.get("provider") or "") or None
        common = {
            "run_id": run_id,
            "process_id": process_invocation_id(),
            "task_run_id": task_run_id,
            "session_id": session_id,
            "task_id": task_id,
            "board": board,
            "model": model,
            "provider": provider,
        }
        if hook_name == "on_session_start":
            # The start marker is the crash-survival anchor.  It must be
            # durable before the first provider request, otherwise SIGKILL can
            # erase the only evidence that a direct run existed.
            ledger.start_run(**common)
        elif hook_name == "post_api_request":
            api_request_id = str(event.get("api_request_id") or "")
            if not api_request_id:
                return
            usage = event.get("usage") or {}
            usage_common = dict(common)
            usage_common["model"] = event.get("response_model") or model
            ledger.queue_model_usage(
                **usage_common,
                event_id=f"{run_id}:api:{api_request_id}",
                turn_id=event.get("turn_id"),
                input_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                output_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)),
                cost_usd=event.get("cost_usd", event.get("estimated_cost_usd", 0.0)) or 0.0,
                retry_count=0,
            )
        elif hook_name == "post_tool_call":
            tool_call_id = str(event.get("tool_call_id") or "")
            if tool_call_id:
                ledger.queue_tool_call(
                    run_id=run_id,
                    event_id=f"{run_id}:tool:{tool_call_id}",
                    session_id=session_id,
                    process_id=process_invocation_id(),
                    task_id=task_id,
                    board=board,
                    task_run_id=task_run_id,
                )
        elif hook_name == "api_request_error":
            if event.get("retryable") is not False:
                request_id = str(event.get("api_request_id") or "")
                if not request_id:
                    return
                retry_count = int(event.get("retry_count") or 0)
                error_payload = event.get("error") or {}
                error_type = error_payload.get("type", "") if isinstance(error_payload, dict) else str(error_payload)
                ledger.queue_retry(
                    run_id=run_id,
                    event_id=f"{run_id}:retry:{request_id}:{retry_count}:{error_type}",
                    session_id=session_id,
                    model=model,
                    provider=provider,
                    process_id=process_invocation_id(),
                    task_id=task_id,
                    board=board,
                    task_run_id=task_run_id,
                )
        elif hook_name == "on_session_finalize":
            if event.get("completed") is True:
                outcome = "completed"
            elif event.get("failed") is True:
                outcome = "failed"
            elif event.get("interrupted") is True:
                outcome = "interrupted"
            else:
                outcome = str(event.get("reason") or "unknown")
            reason = event.get("failure_reason") or event.get("reason")
            finalized = ledger.finalize_run(run_id=run_id, outcome=outcome, failure_reason=reason)
            if finalized and task_run_id is not None:
                ledger.link_kanban_run(
                    task_run_id=int(task_run_id),
                    usage_run_id=run_id,
                    kanban_db=os.environ.get("HERMES_KANBAN_DB"),
                    source_profile=current_source_profile(),
                    board=board,
                )
    except Exception:
        # Usage must never break a model/tool turn or session shutdown.
        logger.debug("Run usage ledger hook failed", exc_info=True)


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Notify first-party observers, then invoke compatibility plugin hooks."""
    _observe_run_usage(hook_name, kwargs)
    try:
        from hermes_cli.observability import observe_lifecycle

        observe_lifecycle(hook_name, **kwargs)
    except Exception:
        logger.warning("Built-in observability hook failed", exc_info=True)

    from hermes_cli import plugins

    return plugins.invoke_hook(hook_name, **kwargs)


def has_hook(hook_name: str) -> bool:
    """Return whether a first-party observer or plugin consumes a hook."""
    if hook_name in {"on_session_start", "post_api_request", "post_tool_call", "api_request_error", "on_session_finalize"}:
        return True
    try:
        from hermes_cli.observability import handles_hook

        if handles_hook(hook_name):
            return True
    except Exception:
        logger.warning("Unable to inspect built-in observability hooks", exc_info=True)

    from hermes_cli import plugins

    return plugins.has_hook(hook_name)


def finalize_session(**kwargs: Any) -> List[Any]:
    """Notify observers and hard-close one core-owned Relay conversation."""
    _observe_run_usage("on_session_finalize", kwargs)
    try:
        from hermes_cli.observability import observe_lifecycle

        observe_lifecycle("on_session_finalize", **kwargs)
    except Exception:
        logger.warning("Built-in observability hook failed", exc_info=True)

    session_id = str(kwargs.get("session_id") or "")
    if session_id:
        try:
            from agent import relay_runtime

            relay_runtime.SESSION_COORDINATOR.finalize_conversation(
                profile_key=relay_runtime.current_profile_key(),
                session_id=session_id,
            )
        except Exception:
            logger.warning("Core Relay session finalization failed", exc_info=True)

    from hermes_cli import plugins

    return plugins.invoke_hook("on_session_finalize", **kwargs)
