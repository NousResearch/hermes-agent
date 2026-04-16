"""ACP permission bridging — maps ACP approval requests to hermes approval callbacks."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable

from acp.schema import (
    AllowedOutcome,
    PermissionOption,
)

logger = logging.getLogger(__name__)

# Maps ACP PermissionOptionKind -> hermes approval result strings
_KIND_TO_HERMES = {
    "allow_once": "once",
    "allow_always": "always",
    "reject_once": "deny",
    "reject_always": "deny",
}


def make_approval_callback(
    request_permission_fn: Callable,
    loop: asyncio.AbstractEventLoop,
    session_id: str,
    timeout: float = 60.0,
) -> Callable[[str, str], str]:
    """
    Return a hermes-compatible ``approval_callback(command, description) -> str``
    that bridges to the ACP client's ``request_permission`` call.

    Args:
        request_permission_fn: The ACP connection's ``request_permission`` coroutine.
        loop: The event loop on which the ACP connection lives.
        session_id: Current ACP session id.
        timeout: Seconds to wait for a response before auto-denying.
    """

    def _callback(command: str, description: str) -> str:
        options = [
            PermissionOption(option_id="allow_once", kind="allow_once", name="Allow once"),
            PermissionOption(option_id="allow_always", kind="allow_always", name="Allow always"),
            PermissionOption(option_id="deny", kind="reject_once", name="Deny"),
        ]
        import acp as _acp

        tool_call = _acp.start_tool_call("perm-check", command, kind="execute")

        coro = request_permission_fn(
            session_id=session_id,
            tool_call=tool_call,
            options=options,
        )

        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            response = future.result(timeout=timeout)
        except (FutureTimeout, Exception) as exc:
            logger.warning("Permission request timed out or failed: %s", exc)
            return "deny"

        outcome = response.outcome
        if isinstance(outcome, AllowedOutcome):
            option_id = outcome.option_id
            # Look up the kind from our options list
            for opt in options:
                if opt.option_id == option_id:
                    return _KIND_TO_HERMES.get(opt.kind, "deny")
            return "once"  # fallback for unknown option_id
        else:
            return "deny"

    return _callback
