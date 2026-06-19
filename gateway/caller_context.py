"""
gateway/caller_context.py

A provider-agnostic ContextVar carrying the (provider, external_id) of the
HUMAN who triggered the current agent turn.

Each gateway (Slack, Telegram, Discord, ...) populates this when an inbound
message starts a session. Downstream code — particularly the MCP tool layer
in ``tools/mcp_tool.py`` — reads it and forwards the caller identity to MCP
servers as part of the ``tools/call`` request metadata, so the server can
authorize the call as the right user.

Why a ContextVar (not a thread-local, not a global):
  - ContextVars propagate naturally across asyncio.create_task() boundaries
    and to executors that copy the calling context (the pattern Hermes
    already uses for ``_slash_user_id`` in gateway/platforms/slack.py and
    HERMES_HOME override in tools/mcp_tool.py).
  - One MCP server can be talking to many concurrent agent sessions (e.g.
    on a desktop with multiple users via web). Globals would cross-talk.
  - reset() with the token gives clean unwinding even when an exception
    fires inside the agent loop.

Why provider-agnostic (not ``_slack_user_id`` style):
  - The same MCP server may be reached from Slack, Telegram, the web
    composer, etc. The downstream resolver (e.g. an external auth-broker
    MCP server) needs to know WHICH provider the external_id belongs to.
  - Keeps the gateway change small (Slack writes the ContextVar; other
    gateways copy the same one-liner) and the MCP-side reader trivial.

This module deliberately has NO Hermes-internal dependencies. It can be
imported from ``gateway/`` and from ``tools/`` without creating import
cycles.
"""

from __future__ import annotations

import contextvars
from typing import NamedTuple, Optional


class CallerIdentity(NamedTuple):
    """Identity of the human who triggered the current agent turn.

    ``provider`` is a short lowercase tag identifying the gateway
    (``"slack"``, ``"telegram"``, ``"discord"``, ``"web"``, ...).
    ``external_id`` is the provider's stable user id (NOT a username or
    display name — those can change). Examples:
      - ``CallerIdentity("slack", "U07TCQBDPMJ")``
      - ``CallerIdentity("telegram", "12345678")``
      - ``CallerIdentity("discord", "987654321098765432")``
    """

    provider: str
    external_id: str


# The actual ContextVar. Default ``None`` means "no caller in context"
# (e.g. a Hermes scheduled cron run, or a CLI invocation without a logged-in
# user). Downstream readers must handle ``None`` cleanly — never crash
# because no caller is set; that's a valid state.
_caller: contextvars.ContextVar[Optional[CallerIdentity]] = contextvars.ContextVar(
    "_workfully_caller", default=None,
)


def set_caller(provider: str, external_id: str) -> contextvars.Token:
    """Set the current caller and return a token for ``reset_caller``.

    Pairs with ``reset_caller`` in a try/finally — this is the same pattern
    ``_slash_user_id`` uses in ``gateway/platforms/slack.py``::

        token = set_caller("slack", user_id)
        try:
            await self.handle_message(event)
        finally:
            reset_caller(token)

    Empty / falsy ``provider`` or ``external_id`` are tolerated by setting
    the ContextVar to ``None`` rather than partial state — downstream
    treats missing caller and partial caller the same way (anonymous).
    """
    if not provider or not external_id:
        return _caller.set(None)
    return _caller.set(CallerIdentity(provider=provider, external_id=external_id))


def get_caller() -> Optional[CallerIdentity]:
    """Return the caller for the current async context, or ``None`` if
    none is set.

    Safe to call from anywhere: gateway code, tools, the MCP layer.
    """
    return _caller.get()


def reset_caller(token: contextvars.Token) -> None:
    """Reset the caller ContextVar to its previous value.

    Pair with ``set_caller``. Tolerates the token coming from a different
    context (e.g. when the agent task is cancelled mid-run) — silently
    ignores the LookupError that ``ContextVar.reset`` raises in that case
    so cleanup never masks the original error.
    """
    try:
        _caller.reset(token)
    except (LookupError, ValueError):
        # Already reset, or reset() called from a different context.
        # Don't propagate — we're almost always in a `finally` block and
        # masking the original exception is worse than a no-op reset.
        pass


__all__ = [
    "CallerIdentity",
    "set_caller",
    "get_caller",
    "reset_caller",
]
