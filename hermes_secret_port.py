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
runs BEFORE the value (or record) is bound in any frame below the port's own write method, and
the write itself, the only code that binds the value, calls nothing that logs with the live
exception (``exc_info``): another component's log handler may keep the record and render its
traceback frames later. The port's own refusals keep the messages the port wrote for them, and
any other failure of the write — whatever the backend or its writers raised, refusal-typed or
not — is re-raised as a fresh :class:`SecretWriteFailed` the port itself builds, outside the
except block, after deleting its own value binding and with both chained-exception slots
cleared (:func:`fresh_write_error`): the fresh error's traceback holds only the raising
method's frame and the caller's frames above it, every frame below that held the value is
gone with the original failure, and the caller's own frames — a caller retrying the write
from inside an ``except`` handler included — are never touched.

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

import errno
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
    refusals — re-made fresh (:func:`fresh_write_error`) with only the details the port itself
    authored: a fixed failure category and, for an OSError, its errno number — never the
    failing code's own text (its message, strerror, filename, even its class name — the
    failing code writes all of those, and one named after the value would publish it), never
    the value or record the caller asked to store, and no frame that held it survives on this
    error's chain."""


def fresh_write_error(exc: BaseException) -> BaseException:
    """The fresh, value-free error a failed credential write raises in place of *exc*.

    Only what this port itself wrote rides along: a message of the port's plus a fixed
    failure category — ``"os-error"`` for an OSError, ``"encode-error"`` for the TypeError
    or ValueError of a failed serialization, ``"write-error"`` for anything else — and, for
    an OSError, its errno number with the ``errno`` module's name for it. Never *exc*'s own
    text: not its message, ``strerror`` or ``filename``, and not its class name either — the
    code that failed — a backend, a writer, a plugin — chose that, and nothing vouches it
    free of the write's value: a writer that raises an exception class named after the value
    it saw puts the value in the error the caller reads as surely as one that names it in a
    message. The fixed categories are the port's own words, so no text the failing code
    controlled reaches the caller; an exception typed as one of the store's own refusals is
    no different, since the port did not raise it. The errno slot is the failing code's too
    — Python allows any object there, so only an errno that is exactly a built-in ``int``
    names the failure (a number nothing the failing code controlled), and every other errno
    is omitted entirely, leaving the category alone. Every failure of the write becomes a
    :class:`SecretWriteFailed`.

    Raise what this returns OUTSIDE the except block, after ``del``-ing the value (or record)
    binding of the write's own frame, with both chained-exception slots cleared first: the
    fresh error's traceback then holds only the raising ``set``/``set_auth`` frame and the
    caller's frames above it — every lower frame that held the value is gone with the original
    failure, and no chain of that failure survives. (Python itself still chains the CALLER's
    own active exception to any raise made inside its handler, which the cleared cause
    suppresses from display; that chain is the caller's, and the port never touches it.)
    """
    # The category is a fixed label of the port's, never a name the failing code chose.
    if isinstance(exc, OSError):
        detail = "os-error"
        # The errno slot is writer-controlled: Python leaves it whatever the failing code
        # passed, string included, so only an errno that is exactly a built-in int names the
        # failure with nothing the failing code wrote.
        if type(exc.errno) is int:
            detail += f", errno {exc.errno}"
            if exc.errno in errno.errorcode:
                detail += f" ({errno.errorcode[exc.errno]})"
    elif isinstance(exc, (TypeError, ValueError)):
        detail = "encode-error"
    else:
        detail = "write-error"
    return SecretWriteFailed(f"the credential write failed: {detail}")


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
        *value* is bound in any frame below this method, the write itself — the only code that
        binds *value* — calls nothing that logs with the live exception (a log handler may keep
        the record and render its frames later), and on ANY failure the method ``del``-etes its
        own *value* binding and raises OUTSIDE the except block: the port's own refusal as it
        raised it, and everything a backend or writer raised — refusal-typed or not — as the
        fresh :class:`SecretWriteFailed` :func:`fresh_write_error` built, with both
        chained-exception slots cleared. The fresh error's traceback then holds only the
        ``set`` frame upward — no lower frame that held the value survives, and the caller's
        own frames (a caller retrying from inside an ``except`` handler included) are never
        touched."""

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
        today). *provider* is the record's key name — the id the record is filed under —
        exempt from the alias guarantee exactly like :meth:`set`'s *name*: a provider id that
        happens to equal a value may remain bound in the store's own frames, and that is not
        a leak; *record* is the secret material, and it is held to the full no-plaintext rule.
        Administration, refused like :meth:`set` — and like it, any error the write
        raises propagates with *record* bound in no frame the error's traceback passes
        through, none of any exception chained to it: the same structural rule (the store
        the write merges into loads and parses before *record* is bound in any frame below
        this method, the merge and the atomic replace are the only code that binds *record*
        and call nothing that logs with the live exception, the method ``del``-etes its own
        *record* binding and raises the port's own refusal as it raised it and everything
        else as :func:`fresh_write_error`'s fresh result outside the except block with the
        chain slots cleared), and never a frame the store does not own touched."""


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
