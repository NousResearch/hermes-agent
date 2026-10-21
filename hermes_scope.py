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

The scope and the profile state readers use (the ``HERMES_HOME`` override behind
``get_hermes_home()``, the secret mapping behind ``get_secret()``, the terminal policy) are ONE
immutable :class:`ScopeState` in one context variable, so they cannot drift apart. A bind is one
change to that value; a reset removes its change and replays the later ones, dropping any made
inside a scope the result no longer has (:func:`pop_state`). An out-of-order, repeated or stale
reset therefore lands on a state whole binds produce, never one profile's home next to another's
secrets. Whether a change is allowed while a scope is bound (another profile moves the whole
scope; a hosted scope refuses) is decided in ``agent/runtime_scope.py``.
"""

from __future__ import annotations

import weakref
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, NamedTuple

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
    """A change that would split the bound scope, or move a hosted one.

    A hosted activity's agent comes from the signed inbound frame; switching to another profile
    mid-activity would serve one agent's work with another's state. Inside any bound scope,
    clearing one piece (secrets, terminal policy) would leave the others describing a profile it
    no longer matches. Bind a new scope at the activity's entry point instead
    (``agent.runtime_scope.bind_profile_scope``)."""


class BoundSecrets(NamedTuple):
    """An installed secret mapping plus the home it was built for, when the binder declared one:
    the provenance ``serves_routed_profile`` needs when a binding skips the home override."""

    mapping: Mapping[str, str]
    profile_home: str | None


@dataclass(frozen=True)
class ScopeState:
    """Everything the current activity's profile state is read from, as one value.

    ``home`` is the ``HERMES_HOME`` override (``None``: the process home). ``secrets`` is the
    secret scope (``None``: unscoped, ``os.environ`` semantics). ``terminal`` is the terminal
    policy mapping or refusal (``None``: process env)."""

    scope: Scope | None = None
    home: str | None = None
    secrets: BoundSecrets | None = None
    terminal: Any = None


_EMPTY = ScopeState()


class _Change:
    """One :func:`push_state`: the fields it set, the scope it was made in (its owner), and a weak
    reference to its token. A change whose token is gone can never be reset, so it is permanent."""

    __slots__ = ("fields", "owner", "token")

    def __init__(self, fields: dict[str, Any], owner: Scope | None) -> None:
        self.fields, self.owner = fields, owner
        self.token: Callable[[], StateToken | None] = lambda: None

    def resettable(self) -> bool:
        return self.token() is not None


class _Node(NamedTuple):
    state: ScopeState
    change: _Change
    parent: _Node | None


_STATE: ContextVar[_Node | None] = ContextVar("hermes_scope_state", default=None)


def current_state() -> ScopeState:
    node = _STATE.get()
    return node.state if node is not None else _EMPTY


def current_scope() -> Scope | None:
    """The bound scope, or ``None`` outside any scoped activity."""
    return current_state().scope


class StateToken:
    """Undoes one :func:`push_state`; usable as a context manager. Resetting twice, or after the
    change was already removed (its owning scope was reset), does nothing."""

    __slots__ = ("__weakref__", "_change")

    def __init__(self, change: _Change) -> None:
        self._change = change
        change.token = weakref.ref(self)

    def reset(self) -> None:
        pop_state(self)

    def __enter__(self) -> StateToken:
        return self

    def __exit__(self, *_exc: object) -> None:
        pop_state(self)


def _fold(node: _Node | None) -> _Node | None:
    """Merge every run of adjacent changes that can no longer be reset and share an owner into one
    change. Such changes are permanent and replay together, so a caller that discards its tokens
    (a set with no reset, or one token variable reassigned in a loop) keeps the chain as long as
    the number of live tokens, not the number of calls. Returns *node* itself when nothing folds."""
    chain: list[_Node] = []
    while node is not None:
        chain.append(node)
        node = node.parent
    rebuilt: _Node | None = None
    folded = False
    for current in reversed(chain):
        below = rebuilt
        if (below is not None and not current.change.resettable() and not below.change.resettable()
                and current.change.owner is below.change.owner):
            merged = _Change({**below.change.fields, **current.change.fields}, current.change.owner)
            rebuilt, folded = _Node(current.state, merged, below.parent), True
        else:
            rebuilt = current if not folded else _Node(current.state, current.change, below)
    return rebuilt


def push_state(**fields: Any) -> StateToken:
    """Install the current state with *fields* (of :class:`ScopeState`) changed."""
    node = _fold(_STATE.get())
    state = node.state if node is not None else _EMPTY
    change = _Change(fields, state.scope)
    token = StateToken(change)
    _STATE.set(_Node(replace(state, **fields), change, node))
    return token


def _agent(scope: Scope | None) -> tuple[str, str] | None:
    return None if scope is None else (scope.tenant, scope.agent)


def pop_state(token: StateToken) -> None:
    """Remove *token*'s change from this context's state.

    The changes made after it are replayed in order, except those made inside a scope the
    replayed state no longer has: state pushed inside a scope goes with that scope. So the result
    is always a state some sequence of whole binds could have produced, never one piece of a
    profile next to another profile's. Leaving a hosted agent is refused unless *token* is the
    bind that entered it."""
    node, later = _STATE.get(), []
    while node is not None and node.change is not token._change:
        later.append(node.change)
        node = node.parent
    if node is None:
        return
    before = current_state()
    rebuilt = node.parent
    for change in reversed(later):
        state = rebuilt.state if rebuilt is not None else _EMPTY
        if state.scope is change.owner:
            rebuilt = _Node(replace(state, **change.fields), change, rebuilt)
    after = rebuilt.state if rebuilt is not None else _EMPTY
    if before.scope is not None and not before.scope.is_local and _agent(after.scope) != _agent(before.scope):
        base = node.parent.state if node.parent is not None else _EMPTY
        entered = _agent(token._change.fields.get("scope")) == _agent(before.scope) != _agent(base.scope)
        if not entered:
            raise ScopeSwitchRefused(
                f"a reset would leave hosted agent {before.scope.agent!r} (tenant "
                f"{before.scope.tenant!r}) from inside it; only the bind that entered it can")
    _STATE.set(rebuilt)
