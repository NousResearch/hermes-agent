"""Integration hooks for hermes-agent components."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _get_audit_context_from_session(
    session_id: str,
    user: str = "",
    channel: str = "",
    platform: str = "",
) -> Dict[str, Any]:
    """
    Build a context dict from session/platform info.
    Used by integration hooks to create OperationContext.
    """
    from audit.events import OperationContext

    return {
        "session_id": session_id,
        "user": user,
        "channel": channel,
        "platform": platform,
    }


def audit_tool_call(
    tool_name: str,
    tool_args: Dict[str, Any],
    session_id: str,
    user: str = "",
    channel: str = "",
    platform: str = "",
    operation_type: str = "Mutate",
) -> Optional[str]:
    """
    Emit RequestReceived audit event for a tool call.

    Returns:
        audit_id for correlation if audit is enabled, None otherwise
    """
    from audit import get_audit_engine
    from audit.events import OperationContext

    engine = get_audit_engine()
    if not engine:
        return None

    context = OperationContext(
        user=user,
        verb="execute",
        resource=tool_name,
        operation_type=operation_type,
        platform=platform,
        channel=channel,
        session_id=session_id,
        source="agent",
    )

    engine.emit_request_received(context, request_body=tool_args)
    return None  # audit_id not needed for now


def audit_tool_result(
    tool_name: str,
    result: Any,
    session_id: str,
    user: str = "",
    channel: str = "",
    platform: str = "",
    operation_type: str = "Mutate",
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    """
    Emit ResponseComplete audit event for a tool result.
    """
    from audit import get_audit_engine
    from audit.events import OperationContext

    engine = get_audit_engine()
    if not engine:
        return

    context = OperationContext(
        user=user,
        verb="execute",
        resource=tool_name,
        operation_type=operation_type,
        platform=platform,
        channel=channel,
        session_id=session_id,
        source="agent",
    )

    response_body = {"success": success}
    if error:
        response_body["error"] = error
    else:
        response_body["result"] = str(result)[:1000] if result else None

    engine.emit_response_complete(
        context,
        response_body=response_body,
        response_code=0 if success else 1,
    )


def audit_message_received(
    message_type: str,
    sender: str,
    session_id: str,
    channel: str,
    platform: str,
    content: str = "",
) -> None:
    """
    Emit RequestReceived audit event for an incoming message.
    """
    from audit import get_audit_engine
    from audit.events import OperationContext

    engine = get_audit_engine()
    if not engine:
        return

    context = OperationContext(
        user=sender,
        verb="receive",
        resource=message_type,
        operation_type="Read",
        platform=platform,
        channel=channel,
        session_id=session_id,
        source="gateway",
    )

    engine.emit_request_received(
        context,
        request_body={"content": content[:500]} if content else None,
    )


def audit_message_sent(
    recipient: str,
    session_id: str,
    channel: str,
    platform: str,
    message_type: str = "message",
    success: bool = True,
) -> None:
    """
    Emit ResponseComplete audit event for an outgoing message.
    """
    from audit import get_audit_engine
    from audit.events import OperationContext

    engine = get_audit_engine()
    if not engine:
        return

    context = OperationContext(
        user="agent",
        verb="send",
        resource=message_type,
        operation_type="Mutate",
        platform=platform,
        channel=channel,
        session_id=session_id,
        source="gateway",
    )

    engine.emit_response_complete(
        context,
        response_body={"recipient": recipient, "success": success},
        response_code=0 if success else 1,
    )