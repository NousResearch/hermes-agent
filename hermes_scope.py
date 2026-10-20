"""The runtime scope: which tenant, agent and actor the current activity serves.

Import-safe, stdlib-only, like ``hermes_constants``. Binding and deriving profile state from a
scope lives in ``agent/runtime_scope.py``; this module holds only the values and the context-local
slot, so ``hermes_constants``, ``agent.secret_scope`` and ``tools.terminal_scope`` can ask "is a
scope bound, and for which agent?" without importing anything heavier.

Every activity a gateway runs for a profile binds a :class:`Scope`: a turn, cache eviction, a
cron tick, a notifier, a deferred callback. A turn binds a :class:`TurnScope`, which adds the
session and the turn. The actor is always present: the person a turn serves, or a
:class:`System` reason for work no person started. On a local install the tenant is
:data:`LOCAL_TENANT`, the agent is the profile home path, and the actor is :data:`LOCAL_OWNER`.

While a scope is bound, the profile state readers use (``get_hermes_home()``, ``get_secret()``,
the terminal policy) is derived from it. The legacy setters (``set_hermes_home_override``,
``set_secret_scope``, ``install_profile_terminal_scope``) cannot split it: inside a local scope, a
call that names a different profile rebinds the whole scope for that profile, so home, secrets
and terminal policy always move together. Inside a non-local (hosted) scope they refuse, because
a hosted activity's agent is fixed by the signed inbound frame and nothing switches profiles
mid-activity.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Callable

LOCAL_TENANT = "local"


@dataclass(frozen=True)
class Principal:
    """A person: the member of a tenant an activity acts for."""

    id: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Principal.id must be a non-empty string")


@dataclass(frozen=True)
class System:
    """Work no person started (boot probes, config reloads): names why it runs."""

    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("System.reason must be a non-empty string")


Actor = Principal | System

# The single user of a local install. Local installs have one person, so every local activity
# acts for them; a shared deployment resolves the actor from the signed inbound frame instead.
LOCAL_OWNER = Principal("owner")


@dataclass(frozen=True)
class Scope:
    """Who an activity serves: ``(tenant, agent, actor)``. Always all three."""

    tenant: str
    agent: str
    actor: Actor

    def __post_init__(self) -> None:
        if not isinstance(self.tenant, str) or not self.tenant:
            raise ValueError("Scope.tenant must be a non-empty string")
        if not isinstance(self.agent, str) or not self.agent:
            raise ValueError("Scope.agent must be a non-empty string")
        if not isinstance(self.actor, (Principal, System)):
            raise TypeError("Scope.actor must be a Principal or a System")

    @property
    def is_local(self) -> bool:
        return self.tenant == LOCAL_TENANT

    def home(self) -> str | None:
        """The profile home this scope's state lives in: the agent path on a local install.

        ``None`` for a hosted scope, whose state lives behind the storage ports rather than in a
        directory."""
        return self.agent if self.is_local else None


@dataclass(frozen=True)
class TurnScope(Scope):
    """A :class:`Scope` for one turn: adds the session the turn belongs to and the turn id."""

    session: str
    turn: str

    def __post_init__(self) -> None:
        Scope.__post_init__(self)
        if not isinstance(self.session, str) or not self.session:
            raise ValueError("TurnScope.session must be a non-empty string")
        if not isinstance(self.turn, str) or not self.turn:
            raise ValueError("TurnScope.turn must be a non-empty string")


class ScopeSwitchRefused(RuntimeError):
    """A legacy profile setter was called inside a hosted scope.

    A hosted activity's agent comes from the signed inbound frame; switching to another profile
    mid-activity would serve one agent's work with another's state. Bind a new scope at the
    activity's entry point instead (``agent.runtime_scope.bind_scope``)."""


_SCOPE: ContextVar[Scope | None] = ContextVar("hermes_runtime_scope", default=None)


def current_scope() -> Scope | None:
    """The bound scope, or ``None`` outside any scoped activity."""
    return _SCOPE.get()


def set_scope(scope: Scope | None) -> Token:
    """Install *scope* in this context. Prefer ``agent.runtime_scope.bind_scope``, which also
    derives the profile state; this is the bare slot for code that binds that state itself."""
    if scope is not None and not isinstance(scope, Scope):
        raise TypeError("set_scope takes a Scope or None")
    return _SCOPE.set(scope)


def reset_scope(token: Token) -> None:
    _SCOPE.reset(token)


@dataclass
class ScopeTokens:
    """Reset tokens for one scope bind: the scope slot plus the home, secret and terminal state
    derived from it. Named fields let a caller release the pieces at different times (the TUI
    keeps the home bound for post-turn housekeeping); :meth:`reset` releases them all.

    Also what a legacy setter returns when it rebinds a whole scope, so its matching
    ``reset_*`` call undoes the whole rebind."""

    scope: Token | None = None
    home: Token | None = None
    secret: Token | None = None
    terminal: Token | None = None
    _resets: list[tuple[Callable[[Any], None], Token]] = field(default_factory=list)
    _released: bool = False

    def __enter__(self) -> ScopeTokens:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.reset()

    def push(self, reset: Callable[[Any], None], token: Token) -> Token:
        self._resets.append((reset, token))
        return token

    def reset(self) -> None:
        """Release in reverse bind order. Each reset is independent: one failing must not leave
        a later piece bound for the next body in this context. The first failure is re-raised
        after every piece has been released."""
        if self._released:
            return
        self._released = True
        first_error: BaseException | None = None
        for reset, token in reversed(self._resets):
            try:
                reset(token)
            except Exception as exc:  # health: allow BLE001 -- release boundary: every piece must unwind; the first failure is re-raised below
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
