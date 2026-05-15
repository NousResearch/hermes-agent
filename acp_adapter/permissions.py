"""ACP permission bridging for Hermes dangerous-command approvals."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import TimeoutError as FutureTimeout
from itertools import count
from typing import Callable

from acp.schema import (
    AllowedOutcome,
    PermissionOption,
)

logger = logging.getLogger(__name__)

# Maps ACP permission option ids to Hermes approval result strings.
# Option ids are stable across both the ``allow_permanent=True`` and
# ``allow_permanent=False`` paths even though the option list differs.
_OPTION_ID_TO_HERMES = {
    "allow_once": "once",
    "allow_session": "session",
    "allow_always": "always",
    "deny": "deny",
}

_PERMISSION_REQUEST_IDS = count(1)


def _build_permission_options(*, allow_permanent: bool) -> list[PermissionOption]:
    """Return ACP options that match Hermes approval semantics."""
    options = [
        PermissionOption(option_id="allow_once", kind="allow_once", name="Allow once"),
        PermissionOption(
            option_id="allow_session",
            # ACP has no session-scoped kind, so use the closest persistent
            # hint while keeping Hermes semantics in the option id.
            kind="allow_always",
            name="Allow for session",
        ),
    ]
    if allow_permanent:
        options.append(
            PermissionOption(
                option_id="allow_always",
                kind="allow_always",
                name="Allow always",
            ),
        )
    options.append(PermissionOption(option_id="deny", kind="reject_once", name="Deny"))
    return options


def _build_permission_tool_call(command: str, description: str):
    """Return the ACP tool-call update attached to a permission request.

    ``request_permission`` expects a ``ToolCallUpdate`` payload — produced
    by ``_acp.update_tool_call`` — not a ``ToolCallStart``. Each request
    gets a unique ``perm-check-N`` id so concurrent requests don't collide.
    """
    import acp as _acp

    tool_call_id = f"perm-check-{next(_PERMISSION_REQUEST_IDS)}"
    return _acp.update_tool_call(
        tool_call_id,
        title=description,
        kind="execute",
        status="pending",
        content=[_acp.tool_content(_acp.text_block(f"$ {command}"))],
        raw_input={"command": command, "description": description},
    )


def _map_outcome_to_hermes(outcome: object, *, allowed_option_ids: set[str]) -> str:
    """Map an ACP permission outcome into Hermes approval strings."""
    if not isinstance(outcome, AllowedOutcome):
        return "deny"

    option_id = outcome.option_id
    if option_id not in allowed_option_ids:
        logger.warning("Permission request returned unknown option_id: %s", option_id)
        return "deny"
    return _OPTION_ID_TO_HERMES.get(option_id, "deny")


def make_approval_callback(
    request_permission_fn: Callable,
    loop: asyncio.AbstractEventLoop,
    session_id: str,
    timeout: float = 60.0,
) -> Callable[..., str]:
    """
    Return a Hermes-compatible approval callback that bridges to ACP.

    The callback accepts ``command`` and ``description`` plus optional
    keyword arguments such as ``allow_permanent`` used by
    ``tools.approval.prompt_dangerous_approval()``.

    Args:
        request_permission_fn: The ACP connection's ``request_permission`` coroutine.
        loop: The event loop on which the ACP connection lives.
        session_id: Current ACP session id.
        timeout: Seconds to wait for a response before auto-denying.
    """

    def _callback(
        command: str,
        description: str,
        *,
        allow_permanent: bool = True,
        **_: object,
    ) -> str:
        from agent.async_utils import safe_schedule_threadsafe

        options = _build_permission_options(allow_permanent=allow_permanent)

        tool_call = _build_permission_tool_call(command, description)
        coro = request_permission_fn(
            session_id=session_id,
            tool_call=tool_call,
            options=options,
        )
        future = safe_schedule_threadsafe(
            coro, loop,
            logger=logger,
            log_message="Permission request: failed to schedule on loop",
        )
        if future is None:
            return "deny"

        try:
            response = future.result(timeout=timeout)
        except (FutureTimeout, Exception) as exc:
            future.cancel()
            logger.warning("Permission request timed out or failed: %s", exc)
            return "deny"

        if response is None:
            return "deny"

        allowed_option_ids = {option.option_id for option in options}
        return _map_outcome_to_hermes(
            response.outcome,
            allowed_option_ids=allowed_option_ids,
        )

    return _callback
