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
some error reporters render traceback frame locals. Caller-supplied arguments other than
the key's own name — the name a ``set`` targets, or the provider id an auth record is filed
under; the names entries are keyed on, never the secret material — are caller text that can
alias the value, so a port-authored message interpolates none of them: the port's errors
name the valid choices (the secret levels) instead, and the failed write clears its own
binding of such an alias before any error leaves its frame. A key name is exempt from the
no-alias guarantee: it may remain bound in the store's own frames, and that is not a leak —
but it never excuses the value (or record) itself, which no frame an error's traceback
passes through may hold. The design is structural, not a cleanup
pass — an implementation never mutates any frame it does not own. Every load and validation
that can fail — the pin, the level, the write lock, the store a record's write merges into —
runs BEFORE the value (or record) is bound in any frame below the port's own write method,
and the write itself — the only code that binds the value — runs through the backend's one
writer boundary, the private helper that calls every writer: its ``try`` sits inside any
lock the write holds, so a writer's failure is caught before a context manager's ``__exit__``
is handed it (an ``__exit__`` assigns the exception's ``__traceback__``, and a hostile
writer's attribute hook would otherwise run while the record is still bound in the frames
being unwound), and the boundary calls nothing that logs with the live exception
(``exc_info``): another component's log handler may keep the record and render its
traceback frames later. The port's own refusals keep the messages the port wrote for them,
and any other failure of the write — whatever the backend or its writers raised,
refusal-typed or not — becomes one fixed :class:`SecretWriteFailed` the port itself builds
(:func:`write_failure`), naming only the operation and the key the write targeted. The
write path never inspects the writer's exception — no attribute, no type check, no message,
no errno, every one of them text the failing code chose and a channel a hostile writer can
carry the value through: it records only that the write failed, and after its except block
has exited, dropping the failure and its traceback with it, the boundary drops its own
bindings of the writer and what it carried and raises the fixed error with ``from None`` —
no cause, no context — while the write method's own value (or record) binding and any other
local holding it leave its frame as the error unwinds it. The fresh error's traceback holds
only the backend's own frames — the boundary's, the write method's — and the caller's frames
above it, every frame below that held the value is gone with the original failure, and the
caller's own frames — a caller retrying the write from inside an ``except`` handler included
— are never touched. The failure's kind is not reported: a caller who needs
it can watch the writer's own logs; the port's error says the write failed, and nothing more.

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


class SecretWriteFailed(OSError):
    """A credential write that failed part-way through the backend's own write path — a plain
    I/O failure, a validation, even an error the backend typed as one of the store's own
    refusals — replaced by the one fixed error :func:`write_failure` builds: a message the
    port wrote alone, naming only the operation and the key it targeted. The failure's kind
    is not reported at all, because the write path never reads the writer's exception — no
    message, ``strerror``, filename, errno or class name, all of them the failing code's own
    text, and channels a hostile writer can carry the value through (an errno property that
    raises with the value it saw, an errno too large for Python to format) — so no value or
    record the caller asked to store, and no frame that held one, survives on the error the
    caller reads."""


def write_failure(operation: str, key: str) -> SecretWriteFailed:
    """The fixed error a failed credential write raises in place of whatever its writer
    raised: a message naming only the *operation* and the *key* the write targeted, both
    the port's own words and the entry's own name — never anything the failing code wrote,
    because the write path never inspects the writer's exception at all: it records that the
    write failed, and once its except block has exited — dropping the failure and its
    traceback — it deletes the write's own value (or record) binding and raises what this
    returns with ``from None``. The fresh error then has no cause and no context, and its
    traceback holds only the raising ``set``/``set_auth`` frame and the caller's frames
    above it: every lower frame that held the value is gone with the original failure.
    (Python itself still chains the CALLER's own active exception to any raise made inside
    its handler, which the cleared cause suppresses from display; that chain is the
    caller's, and the port never touches it.) KeyboardInterrupt and SystemExit are not the
    writer's failure and pass unchanged, before any fresh error is raised."""
    return SecretWriteFailed(f"the credential write failed: {operation} {key!r}")


class SecretStore(ABC):
    """The credentials the bound scope resolves, and the writes that change them.

    Implementations read the pin (and the owner and administration right) from
    ``hermes_scope.current_scope()`` on every call. ``get`` and ``get_auth`` resolve through the
    bound scope's view — agent level, then the agent's owner's owner-wide entries, then
    tenant-wide — so a value defined at several levels is the agent level's. ``names`` is that
    whole view's name set. A failed credential write holds no value on any error path
    (:meth:`set`)."""

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
        failure — propagates with *value* bound in no frame the error's traceback passes
        through, and none of any exception chained to it, because some error reporters render
        traceback frame locals. The implementation is structural about it, never a cleanup
        pass over frames it does not own: every load and validation that can fail runs before
        *value* is bound in any frame below this method, and the write itself — the only code
        that binds *value* — runs through the backend's one writer boundary, the private helper
        that calls every writer, its ``try`` inside any lock the write holds, so a writer's
        failure is caught before a context manager's ``__exit__`` is handed it (an ``__exit__``
        assigns the exception's ``__traceback__``, and a hostile writer's attribute hook would
        otherwise run while the value is still bound in the frames being unwound) and nothing
        on the write path logs with the live exception (a log handler may keep the record and
        render its frames later). On any failure the write path records only that the write
        failed — never reading the writer's exception, its message, type or errno, all of them
        channels the failing code controls — then, once its except block has dropped the
        failure and its traceback, the boundary drops its own bindings of the writer and the
        value it carried and raises OUTSIDE the except block: the port's own refusal as it
        raised it, and everything a backend or writer raised — refusal-typed or not — as the
        fixed :class:`SecretWriteFailed` :func:`write_failure` built, with ``from None``. The
        fresh error's traceback then holds only the backend's own frames — the boundary's, this
        method's — and the caller's above, no frame that held the value survives, and the
        caller's own frames (a caller retrying from inside an ``except`` handler included) are
        never touched."""

    @abstractmethod
    def delete(self, name: str, *, level: str) -> bool:
        """Delete *name* at *level*; ``True`` when the view had it. Refused exactly like
        :meth:`set`; a refusal says nothing about whether *name* exists. A delete is a
        credential write with a read-modify-write tail of its own, and its failures are
        :meth:`set`'s exactly: whatever the backend's writer raised, the caller reads the
        fixed :class:`SecretWriteFailed` :func:`write_failure` built, and no frame the
        error's traceback passes through — the writer's own frames included, which held the
        stored secrets it was rewriting — binds any of them."""

    @abstractmethod
    def get_auth(self, provider: str) -> Optional[dict[str, Any]]:
        """The provider's auth record (an ``auth.json``-shaped dict) the bound scope's view
        resolves, or ``None``. A record only another agent, owner or tenant holds looks the
        same as a missing one."""

    @abstractmethod
    def set_auth(self, provider: str, record: dict[str, Any]) -> None:
        """Write *provider*'s auth *record* at the agent level (where ``auth.json`` keeps it
        today). *provider* is the record's key name — the id the record is filed under —
        exempt from the alias guarantee exactly like :meth:`set`'s *name*: a provider id that
        happens to equal a value may remain bound in the store's own frames, and that is not
        a leak; *record* is the secret material, and it is held to the full no-plaintext rule.
        Administration, refused like :meth:`set` — and like it, any error the write
        raises propagates with *record* bound in no frame the error's traceback passes
        through, none of any exception chained to it: the same structural rule (the store
        the write merges into loads and parses before *record* is bound in any frame below
        this method, and the record's copy, the merge into the loaded store and the atomic
        replace are the only code that binds *record*, each running through the backend's
        one writer boundary — the private helper that calls every writer, its ``try``
        inside the store lock the write holds, so a writer's failure is caught before the
        lock's context managers unwind it and nothing on the write path logs with the live
        exception — and on any failure the write path records only that the write failed
        — never reading the writer's exception — then, once its except block has dropped
        the failure and its traceback, the boundary drops its own bindings of the writer
        and what it carried and the write method's own *record* binding and any other
        local holding it leave its frame as the error unwinds it, the port's own refusal
        raised as it raised it and everything else as :func:`write_failure`'s fixed result
        raised outside the except block with ``from None``), and never a frame the store
        does not own touched."""


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
