"""Immutable, gateway-owned identity context for platform tool dispatch."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class AuthenticatedPlatformContext:
    """Trusted platform identity captured by the gateway, never by model arguments."""

    platform: str
    account_id: str
    user_id: str
    chat_id: str
    thread_id: str | None = None


_authenticated_platform_context: ContextVar[AuthenticatedPlatformContext | None] = ContextVar(
    "authenticated_platform_context", default=None
)


def get_authenticated_platform_context() -> AuthenticatedPlatformContext | None:
    """Return the gateway-bound identity for the current turn/task, if any."""
    return _authenticated_platform_context.get()


def resolve_authenticated_platform_context(
    explicit: AuthenticatedPlatformContext | None = None,
) -> AuthenticatedPlatformContext | None:
    """Resolve the gateway-owned context without allowing explicit callers to forge it.

    The ambient ContextVar is the authority.  An explicit value is accepted only as
    a compatibility assertion from a gateway-owned caller and must exactly match an
    already-bound ambient value; it can never create or replace authority.
    """
    ambient = get_authenticated_platform_context()
    if explicit is not None:
        if ambient is None:
            raise ValueError("explicit authenticated platform context requires ambient gateway context")
        if explicit != ambient:
            raise ValueError("explicit authenticated platform context cannot replace ambient gateway context")
    return ambient


def set_authenticated_platform_context(context: AuthenticatedPlatformContext | None) -> Token:
    """Bind *context* and return a token for the gateway scope's ``finally`` block.

    A nested/plugin caller may not replace or clear an already-bound gateway
    identity.  Normal cleanup uses the token returned here through ``reset``.
    """
    current = get_authenticated_platform_context()
    if current is not None and context != current:
        raise ValueError("authenticated platform context cannot replace ambient gateway context")
    return _authenticated_platform_context.set(context)


def reset_authenticated_platform_context(token: Token) -> None:
    """Restore the previous context, including when the turn raised."""
    _authenticated_platform_context.reset(token)


@contextmanager
def authenticated_platform_context_scope(
    context: AuthenticatedPlatformContext | None,
) -> Iterator[None]:
    """Scope trusted identity to one inbound turn and always restore the prior value."""
    token = set_authenticated_platform_context(context)
    try:
        yield
    finally:
        reset_authenticated_platform_context(token)
