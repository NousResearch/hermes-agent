"""Fail-closed plugin admission and completion hooks for one agent turn."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _workspace_candidate(agent) -> dict[str, str] | None:
    """Expose only the host-selected workspace authority for this surface."""
    try:
        from gateway.session_context import get_session_desktop_project, get_session_env
        project = get_session_desktop_project()
        if project is not None:
            primary = project.get("primary_path")
            if isinstance(primary, str) and primary.strip():
                return {"source": "desktop_project_root", "path": primary}

        source = str(get_session_env("HERMES_SESSION_SOURCE", "") or "").strip()
        platform = str(getattr(agent, "platform", None) or "").strip()
        if source in {"gui", "desktop"} or platform in {"gui", "desktop"}:
            return None
        if source in {"cli", "tui"} or platform in {"cli", "tui"}:
            return {"source": f"{source or platform}_process_cwd", "path": str(Path.cwd())}

        from agent.runtime_cwd import scoped_session_cwd
        cwd = scoped_session_cwd()
        return {"source": "session_context_cwd", "path": cwd} if cwd else None
    except Exception:
        logger.exception("turn workspace authority unavailable")
        return None


def begin_agent_turn_hooks(agent, *, session_id: str, turn_id: str, task_id: str) -> tuple[bool, str | None]:
    """Run pre-agent-turn policy hooks before any model or tool execution.

    A plugin opts into the matching completion hook by returning ``{"action": "allow"}``. A block,
    malformed decision, callback failure, or hook-dispatch failure refuses the turn.
    """
    try:
        from gateway.session_context import get_session_desktop_project
        from hermes_cli.plugins import invoke_hook

        payload = {
            "session_id": session_id,
            "turn_id": turn_id,
            "task_id": task_id,
            "platform": str(getattr(agent, "platform", None) or ""),
            "model": str(getattr(agent, "model", None) or ""),
            "desktop_project": get_session_desktop_project(),
            "workspace_candidate": _workspace_candidate(agent),
        }
        decisions = invoke_hook("pre_agent_turn", **payload)
    except Exception as exc:
        logger.exception("pre_agent_turn hook dispatch failed")
        return False, f"Turn admission hook failed ({type(exc).__name__}); the turn was not started."

    if not isinstance(decisions, list):
        return False, "Turn admission hook dispatch returned an invalid result; the turn was not started."

    opted_in = False
    for decision in decisions:
        if not isinstance(decision, dict):
            return opted_in, "Turn admission hook returned an invalid decision; the turn was not started."
        action = decision.get("action")
        if action == "allow":
            opted_in = True
        elif action == "block":
            message = decision.get("message")
            return opted_in, message if isinstance(message, str) and message.strip() else (
                "A plugin blocked this turn."
            )
        else:
            return opted_in, "Turn admission hook returned an unsupported decision; the turn was not started."
    return opted_in, None


def finish_agent_turn_hooks(
    agent,
    *,
    enabled: bool = True,
    session_id: str,
    turn_id: str,
    task_id: str,
    outcome: str,
    error_type: str | None = None,
) -> None:
    """Notify opted-in plugins after a turn, without passing prompt, answer, or exception text."""
    if not enabled:
        return
    try:
        from gateway.session_context import get_session_desktop_project
        from hermes_cli.plugins import invoke_hook

        invoke_hook(
            "post_agent_turn",
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            platform=str(getattr(agent, "platform", None) or ""),
            desktop_project=get_session_desktop_project(),
            outcome=outcome,
            error_type=error_type,
        )
    except Exception:
        logger.exception("post_agent_turn hook dispatch failed")
