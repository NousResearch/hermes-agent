"""Plugin context of the API server request whose agent turn is running.

``api_server_request`` hooks run once per authenticated API server request that starts an agent turn
(``gateway/platforms/api_server.py`` ``_admit_api_agent_request``), with the request's headers minus the
server's own credentials. The string dict they return is that request's context: it rides the request's
admission in RAM (``gateway/session_api_turn.py``), is bound while that admission executes and reaches
middleware as ``request_context``. It is never persisted, so a turn recovered after a restart, or one no
API request started, runs with an empty context: a consumer that authorizes on it refuses when it is
empty.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any, Mapping

HOOK = "api_server_request"
# The API server's own credentials never reach a plugin.
_WITHHELD_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie"})
_EMPTY: Mapping[str, str] = MappingProxyType({})
_bound: ContextVar[Mapping[str, str]] = ContextVar("request_context", default=_EMPTY)


def current() -> dict[str, str]:
    """The running turn's request context; ``{}`` outside one."""
    return dict(_bound.get())


@contextmanager
def bound(context: Mapping[str, str]):
    token = _bound.set(MappingProxyType(dict(context)))
    try:
        yield
    finally:
        _bound.reset(token)


async def _from_hooks(request: Any) -> Mapping[str, str]:
    """The ``api_server_request`` hooks' string dicts for *request*, merged in registration order."""
    from hermes_cli.lifecycle import ainvoke_hook, has_hook, hook_known_unused

    # On the API server's loop: answering has_hook may load plugins, so it runs off the loop.
    if hook_known_unused(HOOK) or not await asyncio.to_thread(has_hook, HOOK):
        return {}
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _WITHHELD_HEADERS
    }
    merged: dict[str, str] = {}
    for result in await ainvoke_hook(
        HOOK, headers=headers, method=request.method, path=request.path
    ):
        if isinstance(result, dict):
            merged.update(
                (k, v)
                for k, v in result.items()
                if isinstance(k, str) and isinstance(v, str)
            )
    return merged


@asynccontextmanager
async def captured(request: Any):
    """Bind *request*'s context for the handler that admits its turn."""
    with bound(await _from_hooks(request)):
        yield
