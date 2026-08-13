"""Control Room V1 creation flows (CR-206): New Task, New Message, New Agent Run.

Each flow is a two-step contract: ``preview()`` returns the concise scope /
impact summary the renderer must show, and ``submit(confirmed=True)`` executes
through the action router. Renderers may style the preview but cannot omit it
(CR-207). Every flow shows target/profile/scope before confirmation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .actions import ControlRoomActionRouter
from .contract import (
    ActionTarget,
    ControlRoomAction,
    ControlRoomActionResult,
)


class FlowResult:
    def __init__(self, action_result: ControlRoomActionResult, preview: str = ""):
        self.action_result = action_result
        self.preview = preview


class NewTaskFlow:
    """Create a Kanban task (New Task). Routes through kanban_create executor."""

    def __init__(self, router: ControlRoomActionRouter, *, profile: str = "default"):
        self.router = router
        self.profile = profile

    def preview(self, **kwargs: Any) -> str:
        title = str(kwargs.get("title") or "").strip()
        assignee = str(kwargs.get("assignee") or "").strip()
        scope = f"assignee={assignee}" if assignee else "unassigned"
        return f"New Task: {title!r} ({scope}, profile={self.profile})"

    def submit(self, confirmed: bool = False, **kwargs: Any) -> FlowResult:
        title = str(kwargs.get("title") or "").strip()
        if not title:
            return FlowResult(
                ControlRoomActionResult(status="failed", message="New Task requires a title"),
                preview=self.preview(**kwargs),
            )
        action = ControlRoomAction(
            id=kwargs.get("action_id") or "new-task",
            target=ActionTarget(kind="kanban_create", id="new", profile=self.profile),
            parameters={
                "title": title,
                "body": kwargs.get("body"),
                "assignee": kwargs.get("assignee"),
                "created_by": kwargs.get("created_by") or "control-room",
                "tenant": kwargs.get("tenant"),
                "priority": kwargs.get("priority", 0),
                "initial_status": kwargs.get("initial_status", "running"),
            },
            confirmation="required",
        )
        result = self.router.dispatch(action, confirmed=confirmed)
        return FlowResult(result, preview=self.preview(**kwargs))


class NewMessageFlow:
    """Send a Hermes Peer message (New Message). Routes through peer_send."""

    def __init__(self, router: ControlRoomActionRouter, *, profile: str = "default"):
        self.router = router
        self.profile = profile

    def preview(self, **kwargs: Any) -> str:
        target = str(kwargs.get("target") or "")
        message = str(kwargs.get("message") or "")
        snippet = message[:60] + ("..." if len(message) > 60 else "")
        return f"New Message to {target}: {snippet!r} (profile={self.profile})"

    def submit(self, confirmed: bool = False, **kwargs: Any) -> FlowResult:
        target = str(kwargs.get("target") or "").strip()
        message = str(kwargs.get("message") or "").strip()
        if not target or not message:
            return FlowResult(
                ControlRoomActionResult(
                    status="failed",
                    message="New Message requires target and message",
                ),
                preview=self.preview(**kwargs),
            )
        action = ControlRoomAction(
            id=kwargs.get("action_id") or "new-message",
            target=ActionTarget(kind="peer_send", id=target, profile=self.profile),
            parameters={"message": message, "reply_to": kwargs.get("reply_to")},
            confirmation="required",
        )
        result = self.router.dispatch(action, confirmed=confirmed)
        return FlowResult(result, preview=self.preview(**kwargs))


class NewAgentRunFlow:
    """Start a new agent run (New Agent Run). Uses the existing background /
    delegation route; always shows target/profile/scope before confirmation."""

    def __init__(self, router: ControlRoomActionRouter, *, profile: str = "default"):
        self.router = router
        self.profile = profile

    def preview(self, **kwargs: Any) -> str:
        prompt = str(kwargs.get("prompt") or "")
        snippet = prompt[:60] + ("..." if len(prompt) > 60 else "")
        target = str(kwargs.get("target") or "background")
        return f"New Agent Run ({target}): {snippet!r} (profile={self.profile})"

    def submit(self, confirmed: bool = False, **kwargs: Any) -> FlowResult:
        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            return FlowResult(
                ControlRoomActionResult(status="failed", message="New Agent Run requires a prompt"),
                preview=self.preview(**kwargs),
            )
        action = ControlRoomAction(
            id=kwargs.get("action_id") or "new-agent-run",
            target=ActionTarget(kind="agent_run", id="new", profile=self.profile),
            parameters={
                "prompt": prompt,
                "target": kwargs.get("target") or "background",
                "profile": self.profile,
            },
            confirmation="required",
        )
        result = self.router.dispatch(action, confirmed=confirmed)
        return FlowResult(result, preview=self.preview(**kwargs))
