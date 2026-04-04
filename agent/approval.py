"""HITL Approval Workflow — pausable execution with user confirmation.

Provides RequiredAction objects that the agent loop checks before executing
tools marked with requires_confirmation=True. Integrates with:
- CLI: uses existing clarify_callback for interactive approval
- Gateway: emits ApprovalRequired events for push notification
- Audit: logs all approval decisions to session state

Inspired by agno's @approval decorator and RequiredAction system.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default timeout for approval requests (seconds)
DEFAULT_APPROVAL_TIMEOUT = 300  # 5 minutes
# Default action when approval times out
DEFAULT_TIMEOUT_ACTION = "abort"  # "abort" or "proceed"


@dataclass
class RequiredAction:
    """An action requiring user approval before execution."""
    tool_name: str = ""
    tool_call_id: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    approved: bool = False
    resolved_at: float = 0.0
    timeout_seconds: float = DEFAULT_APPROVAL_TIMEOUT
    timeout_action: str = DEFAULT_TIMEOUT_ACTION

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.timeout_seconds

    def approve(self) -> None:
        self.resolved = True
        self.approved = True
        self.resolved_at = time.time()

    def deny(self, reason: str = "") -> None:
        self.resolved = True
        self.approved = False
        self.resolved_at = time.time()
        if reason:
            self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ApprovalDecision:
    """Audit record of an approval decision."""
    tool_name: str
    tool_call_id: str
    approved: bool
    reason: str = ""
    response_time_ms: float = 0.0
    auto_resolved: bool = False  # True if resolved by timeout
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ApprovalManager:
    """Manages approval requests for the agent loop.

    The agent loop calls check_approval() before executing any tool.
    If the tool requires confirmation, the manager creates a RequiredAction,
    requests user input, and returns the decision.
    """

    def __init__(
        self,
        clarify_callback: Optional[Callable] = None,
        event_bus: Any = None,
        session_db: Any = None,
        session_id: str = None,
    ):
        self.clarify_callback = clarify_callback
        self.event_bus = event_bus
        self.session_db = session_db
        self.session_id = session_id
        self._audit_log: List[ApprovalDecision] = []

    def check_approval(
        self,
        tool_name: str,
        tool_call_id: str,
        args: Dict[str, Any],
        requires_confirmation: bool = False,
    ) -> Optional[RequiredAction]:
        """Check if a tool call requires approval. Returns RequiredAction if so.

        Call resolve_action() on the returned object to get the decision.
        Returns None if no approval needed.
        """
        if not requires_confirmation:
            return None

        action = RequiredAction(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            args=args,
            reason=f"Tool '{tool_name}' requires confirmation before execution.",
        )

        # Emit event
        if self.event_bus:
            try:
                from agent.events import ApprovalRequired as _AR
                self.event_bus.emit(_AR(
                    session_id=self.session_id,
                    tool_name=tool_name,
                    args=args,
                    reason=action.reason,
                ))
            except Exception:
                pass

        return action

    def resolve_action(self, action: RequiredAction) -> bool:
        """Resolve a RequiredAction by requesting user input.

        Returns True if approved, False if denied.
        Uses clarify_callback in CLI mode, or auto-resolves on timeout.
        """
        if action.resolved:
            return action.approved

        # Check timeout
        if action.is_expired:
            auto_decision = action.timeout_action == "proceed"
            if auto_decision:
                action.approve()
            else:
                action.deny("Approval timed out")
            self._record_decision(action, auto_resolved=True)
            return action.approved

        # CLI mode: use clarify callback
        if self.clarify_callback:
            try:
                args_preview = json.dumps(action.args, ensure_ascii=False)
                if len(args_preview) > 200:
                    args_preview = args_preview[:197] + "..."

                question = (
                    f"Tool '{action.tool_name}' requires approval.\n"
                    f"Arguments: {args_preview}\n"
                    f"Allow this tool to execute?"
                )
                response = self.clarify_callback(question, ["Yes, proceed", "No, skip"])

                if response and "yes" in response.lower():
                    action.approve()
                else:
                    action.deny("User denied")
            except Exception as e:
                logger.debug("Clarify callback failed: %s", e)
                action.deny(f"Approval request failed: {e}")
        elif os.environ.get("HERMES_GATEWAY_SESSION"):
            # Gateway mode: bridge to submit_pending so the API server's
            # SSE approval flow can handle it via the companion app.
            try:
                from tools.approval import submit_pending
                session_key = os.environ.get("HERMES_SESSION_KEY", "default")
                args_preview = json.dumps(action.args, ensure_ascii=False)[:200]
                submit_pending(session_key, {
                    "command": f"{action.tool_name}({args_preview})",
                    "pattern_key": f"tool:{action.tool_name}",
                    "pattern_keys": [f"tool:{action.tool_name}"],
                    "description": f"Tool '{action.tool_name}' requires confirmation",
                })
                # Return denied — the api_server SSE writer will detect
                # the pending approval and handle it via the companion.
                action.deny("Routed to gateway approval gate")
            except Exception as e:
                logger.debug("Gateway approval bridge failed: %s", e)
                action.deny(f"Approval request failed: {e}")
        else:
            # No callback and not gateway — use timeout action as default
            if action.timeout_action == "proceed":
                action.approve()
            else:
                action.deny("No approval mechanism available")

        self._record_decision(action)

        # Emit resolution event
        if self.event_bus:
            try:
                from agent.events import ApprovalResolved as _ARes
                self.event_bus.emit(_ARes(
                    session_id=self.session_id,
                    tool_name=action.tool_name,
                    approved=action.approved,
                    reason=action.reason,
                ))
            except Exception:
                pass

        return action.approved

    def _record_decision(self, action: RequiredAction, auto_resolved: bool = False) -> None:
        """Record an approval decision to the audit log and session state."""
        response_time = (action.resolved_at - action.created_at) * 1000 if action.resolved_at else 0

        decision = ApprovalDecision(
            tool_name=action.tool_name,
            tool_call_id=action.tool_call_id,
            approved=action.approved,
            reason=action.reason,
            response_time_ms=response_time,
            auto_resolved=auto_resolved,
        )
        self._audit_log.append(decision)

        # Persist to session state
        if self.session_db and self.session_id:
            try:
                state = self.session_db.get_session_state(self.session_id)
                log = state.get("_approval_audit", [])
                log.append(decision.to_dict())
                # Keep last 50 decisions
                if len(log) > 50:
                    log = log[-50:]
                self.session_db.update_session_state(self.session_id, {"_approval_audit": log})
            except Exception:
                pass

    @property
    def audit_log(self) -> List[ApprovalDecision]:
        return list(self._audit_log)
