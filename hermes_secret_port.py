"""The secret store port: the credentials one scope resolves, as a plugin kind.

:class:`SecretStore` is the one surface for credential reads and writes, at the level today's
call sites use, never a file path: ``get`` / ``set`` / ``delete`` / ``names`` for environment-style
secrets, ``get_auth`` / ``set_auth`` for provider records. Every entry carries the ownership key
(``hermes_scope`` docs): tenant, subject and agent. There is no per-participant secret — the
participant-on-agent level is for conversations, conversation files and ``USER.md`` — so a
backend holds entries at three levels:

- **agent** (``(tenant, NULL, agent)``): everything that is per agent today — a profile's ``.env``
  keys and its ``auth.json`` records;
- **owner-wide** (``(tenant, owner, NULL)``): a secret a principal or a group sets once for all
  the agents it owns;
- **tenant-wide** (``(tenant, NULL, NULL)``): an org-wide secret.

A read resolves the bound scope's view: the agent level first, then the owner-wide entries of the
agent's owner (never the actor's own: a member's owner-wide secrets never reach a group's agent,
whose owner is the group), then tenant-wide. The level a write lands at is decided by the store
from the bound scope plus the explicit ``level`` argument, never from a caller-supplied id.

Writes are administration. ``set``, ``delete`` and ``set_auth`` are refused with
:class:`SecretAdminRefused` unless the bound scope may administer the agent
(``Scope.can_administer``, the signed claim's administration right; a local scope defaults to the
local person administering). A write naming a level the backend does not hold is refused with
:class:`SecretLevelRefused`; a level no secret store holds is a programming error. No refusal says
whether the target exists, and no refusal, log line or error the store raises ever contains a
secret's value — nor is one still bound in any frame such an error's traceback passes through,
not the raising implementation's own frames, not the frames of any helper the failed write
handed it to, and not the frames of any exception chained to the error as context or cause:
an implementation clears the value (or record) from all of them before the error propagates —
on every path, a refusal, a validation error or a mid-write failure — because some error
reporters render traceback frame locals. :func:`clear_plaintext_from_frames` is that clearing,
for a backend's ``except`` block.

Every operation reads its pin (tenant, agent, owner, actor) from ``hermes_scope.current_scope()``
when it runs, never from a path, ``HERMES_HOME`` or an argument; a port exposes no way to pin
anything else. A backend refuses a pin it cannot serve with :class:`SecretPinRefused`. A hosted
backend with no scope bound refuses; the local backend's pin is the ambient profile
(``get_hermes_home()``) when no scope is bound, as every local caller has it today.

This is the interface and its local backend only: today's readers still call
``agent.secret_scope.get_secret``, ``hermes_cli.config.save_env_value`` and the ``hermes_cli.auth``
store directly; moving them behind this port, call site by call site, is a follow-up. The
turn-time secret scope keeps its current shape and semantics, and ``get_secret()`` keeps failing
closed with no scope bound. :func:`resolve_secret_store` takes the backend's name explicitly:
``"local"`` (or empty) is ``hermes_secret_port_local`` (today's profile ``.env`` and ``auth.json``);
any other name is a provider from ``plugins/secret_store/<name>/`` or
``$HERMES_HOME/plugins/<name>/``. A named provider that is missing or unavailable is an error,
never a fallback to the local files.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

# The ownership levels a secret store can hold (``Scope`` docs). Secrets have no
# participant-on-agent level; a backend declares which of these it stores and refuses
# writes at any other.
LEVELS = frozenset({"agent", "owner", "tenant"})


class SecretPinRefused(PermissionError):
    """The bound scope is not one this store serves: another tenant, another agent, or no scope
    at all for a backend that serves none by default. Says nothing about whether the target
    exists."""


class SecretAdminRefused(PermissionError):
    """A credential write from a scope that may not administer the agent. Says nothing about
    whether the target exists, and never contains a secret's value."""


class SecretLevelRefused(PermissionError):
    """A write at a level this backend does not hold. Says nothing about whether the target
    exists, and never contains a secret's value."""


def _plaintext_texts(value: Any) -> tuple[str, ...]:
    """Every string *value* — itself a secret, or a record (dict, list, tuple) holding some —
    is made of: the texts a failed write's frame locals must not still render."""
    texts: list[str] = []

    def collect(node: Any, seen: set[int]) -> None:
        if id(node) in seen:  # a record that contains itself collects nothing twice
            return
        seen.add(id(node))
        if isinstance(node, str):
            if node:
                texts.append(node)
        elif isinstance(node, dict):
            for held in node.values():
                collect(held, seen)
        elif isinstance(node, (list, tuple, set, frozenset)):
            for held in node:
                collect(held, seen)

    collect(value, set())
    return tuple(texts)


def clear_plaintext_from_frames(exc: BaseException, value: Any) -> None:
    """Clear *value* — a secret, or a record holding some — from the locals of every frame
    *exc* passes through: every frame of its own traceback, and of every exception chained to
    it, as context or cause, recursively (a chain that loops clears each frame once), and of
    every ``ExceptionGroup`` member's — the raising backend's frames and the frames of every
    helper the failed write handed the plaintext to, wherever that code lives. A frame local
    that renders the plaintext is replaced with ``None`` (error reporters render traceback
    frame locals, and a re-raised error keeps its tracebacks). Call this from the write path's
    own ``except`` block, before re-raising, on every failure of a credential write — a
    refusal, a validation error, a mid-write I/O failure.
    """
    texts = _plaintext_texts(value)
    if not texts:
        return
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        linked = pending.pop()
        if id(linked) in seen:
            continue
        seen.add(id(linked))
        tb = linked.__traceback__
        while tb is not None:
            locals_ = tb.tb_frame.f_locals
            for key in list(locals_):
                try:
                    if any(text in repr(locals_[key]) for text in texts):
                        locals_[key] = None  # frame locals accept no deletion; None renders nothing
                except Exception:  # health: allow BLE001 -- a binding whose repr or set raises is one that renders nothing
                    continue
            tb = tb.tb_next
        for chained in (linked.__context__, linked.__cause__):
            if chained is not None:
                pending.append(chained)
        if isinstance(linked, BaseExceptionGroup):
            pending.extend(linked.exceptions)


class SecretStore(ABC):
    """The credentials the bound scope resolves, and the writes that change them.

    Implementations read the pin (and the owner and administration right) from
    ``hermes_scope.current_scope()`` on every call. ``get`` and ``get_auth`` resolve through the
    bound scope's view — agent level, then the agent's owner's owner-wide entries, then
    tenant-wide — so a value defined at several levels is the agent level's. ``names`` is that
    whole view's name set. A failed credential write clears its value (or record) from every
    frame its error passes through before the error propagates (:meth:`set`)."""

    @abstractmethod
    def get(self, name: str) -> Optional[str]:
        """The secret *name* resolves to in the bound scope's view, or ``None`` (a name defined
        only by another agent, owner or tenant looks the same as one defined nowhere)."""

    @abstractmethod
    def names(self) -> list[str]:
        """Every name the bound scope's view defines, across all levels, sorted."""

    @abstractmethod
    def set(self, name: str, value: str, *, level: str) -> None:
        """Write *name* at *level* (an explicit :data:`LEVELS` member; the subject and agent of
        the entry come from the bound scope, never from the caller). Administration: refused
        with :class:`SecretAdminRefused` unless the bound scope may administer the agent, and
        with :class:`SecretLevelRefused` at a level this backend does not hold. Any error the
        write raises — a refusal, a level that is not one at all, a locked key, a mid-write I/O
        failure — propagates with *value* bound in no frame the implementation's code owns, in
        no frame of any helper it handed the value to, and in no frame of any exception chained
        to the error as context or cause: clear them (:func:`clear_plaintext_from_frames`)
        before re-raising — error reporters render traceback frame locals."""

    @abstractmethod
    def delete(self, name: str, *, level: str) -> bool:
        """Delete *name* at *level*; ``True`` when the view had it. Refused exactly like
        :meth:`set`; a refusal says nothing about whether *name* exists."""

    @abstractmethod
    def get_auth(self, provider: str) -> Optional[dict[str, Any]]:
        """The provider's auth record (an ``auth.json``-shaped dict) the bound scope's view
        resolves, or ``None``. A record only another agent, owner or tenant holds looks the
        same as a missing one."""

    @abstractmethod
    def set_auth(self, provider: str, record: dict[str, Any]) -> None:
        """Write *provider*'s auth *record* at the agent level (where ``auth.json`` keeps it
        today). Administration, refused like :meth:`set` — and like it, any error the write
        raises propagates with *record* cleared from every frame it passed through, the
        implementation's own and any helper's, chained exceptions included
        (:func:`clear_plaintext_from_frames`)."""


class SecretStoreProvider(ABC):
    """The plugin kind: supplies :class:`SecretStore`. ``name``, ``open`` and
    ``assert_isolating`` are required; every other method has a safe default, and growth stays
    additive."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``"local"``."""

    #: The ownership levels this backend stores (see :data:`LEVELS`). A local install holds the
    #: agent level only, and says so here: its writes at the other levels are refused.
    levels: frozenset[str] = frozenset({"agent"})

    def is_available(self) -> bool:
        """Whether this backend can run here. No network calls."""
        return True

    @abstractmethod
    def open(self, home: Any = None) -> SecretStore:
        """The store for the bound scope. *home* is the local backend's profile home (the
        caller's ``get_hermes_home()`` when omitted); a backend with no local directory ignores
        it and serves the agents of the bound scope."""

    @abstractmethod
    def assert_isolating(self) -> None:
        """Raise unless this backend isolates tenants and agents from each other in storage
        itself (an encrypted hosted backend whose role is a superuser or bypasses row security
        fails, and one that stores values in the clear fails). Required, with no default: a
        backend that has not checked must not pass. Nothing calls it yet: the first caller that
        serves a scope from a resolved provider must call it, and refuse the provider when it
        raises, before :meth:`open`."""


def resolve_secret_store(name: str) -> SecretStoreProvider:
    """The provider named *name* (``"local"`` or empty: the local backend). Raises ``LookupError``
    with the reason for a named provider that is missing, refused or unavailable: secrets never
    fall back to another store."""
    name = name.strip()
    if name in ("", "local"):
        from hermes_secret_port_local import LocalSecretStoreProvider
        return LocalSecretStoreProvider()
    from plugins.secret_store import SecretStoreRefused, load_secret_store
    try:
        provider = load_secret_store(name)
    except SecretStoreRefused as refused:
        raise LookupError(f"secret store {name!r}: {refused}") from refused
    if provider is None:
        raise LookupError(f"secret store {name!r}: no such secret store provider")
    if not provider.is_available():
        raise LookupError(f"secret store {name!r} is not available here")
    return provider
