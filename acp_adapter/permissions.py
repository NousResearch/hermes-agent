"""ACP permission bridging for Hermes dangerous-command approvals."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import TimeoutError as FutureTimeout
from itertools import count
from typing import Callable

from acp.schema import AllowedOutcome, PermissionOption

logger = logging.getLogger(__name__)

# ACP permission option id -> Hermes approval result. Ids are stable across the
# ``allow_permanent=True`` and ``False`` paths even though the option list differs.
_OPTION_ID_TO_HERMES = {
    "allow_once": "once", "allow_session": "session", "allow_always": "always", "deny": "deny", "deny_always": "deny"
}

_PERMISSION_REQUEST_IDS = count(1)


def _permission_option_supports_kind(kind: str) -> bool:
    """Return whether the installed ACP SDK accepts a permission option kind."""
    try:
        PermissionOption(option_id="__probe__", kind=kind, name="probe")
        return True
    except Exception:
        return False


def _build_permission_options(
    *, allow_permanent: bool, allow_session: bool = True, smart_denied: bool = False,
) -> list[PermissionOption]:
    """Return ACP options that match Hermes approval semantics."""
    # A gate that re-asks every time (allow_session=False, e.g. protected
    # agent-instruction writes) collapses to the same two options as a Smart
    # DENY override — offering a scope Hermes discards would re-prompt every write.
    # See #81887.
    once_only = smart_denied or not allow_session
    options = [PermissionOption(option_id="allow_once", kind="allow_once", name="Allow once")]
    if not once_only:
        # ACP has no session-scoped kind: closest persistent hint, Hermes semantics in the id.
        options.append(PermissionOption(option_id="allow_session", kind="allow_always", name="Allow for session"))
        if allow_permanent:
            options.append(PermissionOption(option_id="allow_always", kind="allow_always", name="Allow always"))
    options.append(PermissionOption(option_id="deny", kind="reject_once", name="Deny"))
    if not once_only and _permission_option_supports_kind("reject_always"):
        options.append(PermissionOption(option_id="deny_always", kind="reject_always", name="Deny always"))
    return options


def _build_permission_tool_call(command: str, description: str):
    """Return the ``ToolCallUpdate`` (not ``ToolCallStart``) payload attached to a
    permission request; unique ``perm-check-N`` ids keep concurrent requests apart."""
    import acp as _acp

    content_text = f"{description}\n$ {command}" if description else f"$ {command}"
    return _acp.update_tool_call(
        f"perm-check-{next(_PERMISSION_REQUEST_IDS)}", title=f"{description}: {command}" if description else command,
        kind="execute", status="pending", content=[_acp.tool_content(_acp.text_block(content_text))],
        raw_input={"command": command, "description": description},
    )


def _map_outcome_to_hermes(outcome: object, *, allowed_option_ids: set[str]) -> str:
    """Map an ACP permission outcome into Hermes approval strings."""
    if not isinstance(outcome, AllowedOutcome):
        return "deny"
    if outcome.option_id not in allowed_option_ids:
        logger.warning("Permission request returned unknown option_id: %s", outcome.option_id)
        return "deny"
    return _OPTION_ID_TO_HERMES.get(outcome.option_id, "deny")


def await_permission(
    request_permission_fn: Callable, loop: asyncio.AbstractEventLoop, session_id: str, *,
    tool_call, options: list[PermissionOption], timeout: float, what: str,
    send_update: Callable[[object], None] | None = None,
) -> tuple[object | None, bool]:
    """Schedule ``request_permission`` on ``loop`` from a worker thread and block for the answer.
    Returns ``(response, timed_out)``; ``(None, False)`` when scheduling or the request failed.

    Clients materialise the request's ``tool_call`` as a pending bubble, so once the answer is
    in, ``send_update`` (when given) closes it: ``completed`` for an allow, ``failed`` otherwise."""
    from agent.async_utils import safe_schedule_threadsafe

    coro = request_permission_fn(session_id=session_id, tool_call=tool_call, options=options)
    future = safe_schedule_threadsafe(coro, loop, logger=logger, log_message=f"{what}: failed to schedule on loop")
    if future is None:
        return None, False
    response, timed_out = None, False
    try:
        response = future.result(timeout=timeout)
    except FutureTimeout:
        future.cancel()
        logger.warning("%s timed out after %ss", what, timeout)
        timed_out = True
    except Exception as exc:
        future.cancel()
        logger.warning("%s failed: %s", what, exc)
    if send_update is not None:
        import acp as _acp

        # Duck-typed like the callers' own allow checks (``outcome == "selected"`` is the wire
        # discriminator), so the bubble's terminal status always matches the decision taken.
        outcome = getattr(response, "outcome", None)
        allowed = getattr(outcome, "outcome", None) == "selected" and any(
            option.option_id == getattr(outcome, "option_id", None) and option.kind.startswith("allow")
            for option in options
        )
        send_update(_acp.update_tool_call(tool_call.tool_call_id, status="completed" if allowed else "failed"))
    return response, timed_out


def make_approval_callback(request_permission_fn: Callable, loop: asyncio.AbstractEventLoop,
                           session_id: str, timeout: float = 60.0,
                           send_update: Callable[[object], None] | None = None) -> Callable[..., str]:
    """Return a Hermes approval callback (``command, description, **kw`` as used by
    ``tools.approval.prompt_dangerous_approval()``) that bridges to the ACP
    connection's ``request_permission`` coroutine on ``loop``; auto-denies after ``timeout`` s."""

    def _callback(command: str, description: str, *, allow_permanent: bool = True,
                  allow_session: bool = True, smart_denied: bool = False, **_: object) -> str:
        options = _build_permission_options(allow_permanent=allow_permanent, allow_session=allow_session,
                                            smart_denied=smart_denied)
        response, timed_out = await_permission(
            request_permission_fn, loop, session_id, tool_call=_build_permission_tool_call(command, description),
            options=options, timeout=timeout, what="Permission request", send_update=send_update,
        )
        if timed_out:
            # Distinct from an explicit deny: tools.approval reports "timed out
            # without user response" instead of a user denial.
            return "timeout"
        if response is None:
            return "deny"
        return _map_outcome_to_hermes(response.outcome, allowed_option_ids={option.option_id for option in options})

    return _callback
