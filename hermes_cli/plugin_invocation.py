"""Trusted, invocation-local context for plugin command handlers."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import Context, ContextVar, copy_context
from dataclasses import dataclass, field
from typing import Iterator, Literal

PluginExecutionKind = Literal["root", "background", "subagent"]


@dataclass(frozen=True, slots=True, repr=False)
class PluginInvocationContext:
    """Host-derived identity and routing facts for one plugin invocation."""

    _profile: str | None
    _session_id: str | None
    _platform: str | None
    _authenticated_actor: str | None
    _target: str | None
    _chat_id: str | None
    _thread_id: str | None
    _origin: str | None
    _execution_kind: PluginExecutionKind
    _lease: list[bool] = field(default_factory=lambda: [True], repr=False, compare=False)

    def _read(self, value):
        if not self._lease[0]:
            raise PluginInvocationContextUnavailable("plugin invocation context has expired")
        return value

    @property
    def profile(self) -> str | None:
        return self._read(self._profile)

    @property
    def session_id(self) -> str | None:
        return self._read(self._session_id)

    @property
    def platform(self) -> str | None:
        return self._read(self._platform)

    @property
    def authenticated_actor(self) -> str | None:
        return self._read(self._authenticated_actor)

    @property
    def target(self) -> str | None:
        return self._read(self._target)

    @property
    def chat_id(self) -> str | None:
        return self._read(self._chat_id)

    @property
    def thread_id(self) -> str | None:
        return self._read(self._thread_id)

    @property
    def origin(self) -> str | None:
        return self._read(self._origin)

    @property
    def execution_kind(self) -> PluginExecutionKind:
        return self._read(self._execution_kind)


class PluginInvocationContextUnavailable(RuntimeError):
    """No trusted plugin invocation is bound to the current execution context."""


_CURRENT_PLUGIN_INVOCATION: ContextVar[PluginInvocationContext | None] = ContextVar(
    "hermes_current_plugin_invocation", default=None
)


def current_plugin_invocation() -> PluginInvocationContext:
    """Return the current host-bound invocation or fail closed when none exists."""
    invocation = _CURRENT_PLUGIN_INVOCATION.get()
    if invocation is None:
        raise PluginInvocationContextUnavailable("no trusted plugin invocation is active")
    invocation._read(None)
    return invocation


def _new_plugin_invocation(
    *,
    profile: str | None,
    session_id: str | None,
    platform: str | None,
    authenticated_actor: str | None,
    target: str | None,
    chat_id: str | None,
    thread_id: str | None,
    origin: str | None,
    execution_kind: PluginExecutionKind,
) -> PluginInvocationContext:
    """Create a host-owned, revocable invocation value."""
    return PluginInvocationContext(
        profile,
        session_id,
        platform,
        authenticated_actor,
        target,
        chat_id,
        thread_id,
        origin,
        execution_kind,
    )


def _revoke_plugin_invocation(invocation: PluginInvocationContext) -> None:
    invocation._lease[0] = False


def _context_without_plugin_invocation() -> Context:
    context = copy_context()
    context.run(_CURRENT_PLUGIN_INVOCATION.set, None)
    return context


@contextmanager
def _bind_plugin_invocation(invocation: PluginInvocationContext) -> Iterator[None]:
    """Bind one invocation for host dispatch and restore the prior value afterwards."""
    if not isinstance(invocation, PluginInvocationContext):
        raise TypeError("invocation must be a PluginInvocationContext")
    token = _CURRENT_PLUGIN_INVOCATION.set(invocation)
    try:
        yield
    finally:
        _CURRENT_PLUGIN_INVOCATION.reset(token)
        _revoke_plugin_invocation(invocation)
