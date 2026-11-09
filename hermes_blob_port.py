"""The blob storage port: Hermes-owned files behind one plugin kind.

:class:`BlobStore` is ``get`` / ``put`` / ``delete`` / ``list`` / ``exists`` on an opaque
*key*: a string that names a blob inside the agent's namespace (today a ``/``-separated
path relative to a file family, e.g. ``"media/image/<file>"`` or ``"pets/<slug>/<file>"``)
and means nothing else. Tenant, principal and agent are never taken from the key: every
blob carries the ownership key (``hermes_scope``) of the scope bound when the operation
runs, and a port exposes no way to pin anything else. A backend refuses a pin it cannot
serve with :class:`BlobPinRefused`, a write at a level it does not store (or its actor
may not write) with :class:`BlobLevelRefused`, and a key that cannot name a blob of this
agent's namespace at all — one that escapes it (``..``, an absolute path, a component
that is a separator or a drive) — with :class:`BlobKeyRefused`, never by silently
resolving it elsewhere or treating it as missing.

Levels (``hermes_scope``'s ownership key): files that belong to a conversation —
attachments, inbound media, media generated in that conversation — are
``"participant"``-level records of the principal whose conversation they belong to;
pets, skill bodies and the agent's own files are ``"agent"``-level records. A write
names its level explicitly; a read resolves the same key through the one documented
policy, participant-level first (the bound principal's own records), then agent level.
A backend declares which levels it stores (:attr:`BlobStoreProvider.levels`) and
refuses writes at any other with a clear error; today's local backend stores the agent
level only, which is the documented local exception (a local install has one agent
owner and, until profiles become agents, one participant).

This is the interface and its local backend only: no production caller goes through it
yet, and nothing in config.yaml selects a backend. Today's callers still read and write
their files directly; moving them behind this port, family by family, is a follow-up,
and until it lands a provider plugin stores nothing that Hermes reads.
:func:`resolve_blob_store` takes the backend's name explicitly: ``"local"`` (or empty)
is ``hermes_blob_port_local`` (today's files under the profile home, at today's paths);
any other name is a provider from ``plugins/blob_storage/<name>/`` or
``$HERMES_HOME/plugins/<name>/``. A named provider that is missing or unavailable is an
error, never a fallback to the local store. Providers load in-process only: under
``plugins.isolation: host`` a user provider is refused with the reason
(:class:`plugins.blob_storage.BlobStorageRefused`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

# The ownership levels a blob store can hold (``hermes_scope``'s ownership key, restricted
# to the two levels files have): participant-on-agent for conversation files, agent for the
# rest. A backend declares which it stores and refuses writes at any other.
LEVELS = frozenset({"participant", "agent"})


class BlobPinRefused(PermissionError):
    """The bound scope is not one this store serves: another tenant, another agent, or (for a
    backend that requires one) no scope at all. Says nothing about whether the target exists."""


class BlobLevelRefused(PermissionError):
    """A write at a level this backend does not store, or one its bound actor may not write
    (a participant-level write by an actor that is no principal). The refusal names the level
    and says nothing about whether the target exists."""


class BlobKeyRefused(ValueError):
    """A key that cannot name a blob of this agent's namespace: it escapes it (``..`` components,
    an absolute path, a separator or drive inside a component), or its family is not one this
    store serves. Malformed, not missing: never resolved elsewhere, never reported as absent."""


class BlobNotFound(KeyError):
    """The key names no blob this pin can see. A foreign blob and a missing one look the same."""


def parse_key(key: str) -> tuple[str, ...]:
    """The key's path components, checked against the escape rules every backend shares:
    no absolute key, no ``.``/``..`` component, and no component carrying a separator
    (``/``, ``\\``), a NUL or a drive colon — a path-like key must never name anything
    outside the agent's namespace, whatever the backend joins it onto. Backends whose keys
    are opaque non-path tokens simply never see a ``/`` and pass unchanged.

    The key names a blob, never an owner: components are validated for containment only,
    so a backend can treat them as path segments or as opaque tokens.
    """
    if not isinstance(key, str):
        raise BlobKeyRefused(f"a blob key is a string, not {type(key).__name__}")
    if not key:
        raise BlobKeyRefused("a blob key is never empty")
    if key[0] in ("/", "\\"):
        raise BlobKeyRefused(f"the blob key {key!r} is absolute; a key names a blob of the agent's namespace")
    components = key.split("/")
    for component in components:
        if component in ("", ".", ".."):
            raise BlobKeyRefused(
                f"the blob key {key!r} has an empty or parent component; a key cannot leave the agent's namespace")
        if any(bad in component for bad in ("/", "\\", "\x00", ":")):
            raise BlobKeyRefused(
                f"the blob key {key!r} has a component with a separator, drive colon or NUL in it")
    return tuple(components)


class BlobStore(ABC):
    """Files of the pinned agent, addressed by an opaque key."""

    @abstractmethod
    def get(self, key: str) -> bytes:
        """The blob's bytes. Raises :class:`BlobNotFound` when the key names nothing this pin
        can see — a missing blob and a foreign one look the same."""

    @abstractmethod
    def put(self, key: str, data: bytes, *, level: str = "agent") -> None:
        """Write *data* at *key*, at the level *level* (:data:`LEVELS`), which the write names
        explicitly. The blob carries the ownership key of the bound scope; another principal's
        or agent's blob at the same key is a different blob, never one this overwrites."""

    @abstractmethod
    def delete(self, key: str, *, level: str = "agent") -> bool:
        """Delete the blob *key* names at *level*; ``False`` when there was nothing to delete
        (a foreign blob and a missing one look the same)."""

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """Every key this pin can see under *prefix* (``""`` for all of them), sorted. The
        prefix obeys the same escape rules as a key."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether *key* names a blob this pin can see, resolving a read's levels."""


class BlobStoreProvider(ABC):
    """The plugin kind: supplies the port. ``name``, ``open`` and ``assert_isolating`` are
    required; every other method has a safe default, and growth stays additive."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``"local"``."""

    #: The ownership levels this backend stores (see :data:`LEVELS`).
    levels: frozenset[str] = frozenset({"agent"})

    def is_available(self) -> bool:
        """Whether this backend can run here. No network calls."""
        return True

    @abstractmethod
    def open(self, home: Any = None) -> BlobStore:
        """The store for the bound scope. *home* is the local backend's profile home (the
        caller's ``HERMES_HOME``); a backend with no local file ignores it."""

    @abstractmethod
    def assert_isolating(self) -> None:
        """Raise unless this backend isolates tenants and agents from each other in storage
        itself (a bucket or index whose role bypasses tenant isolation fails). Required, with
        no default: a backend that has not checked must not pass. Nothing calls it yet: the
        first caller that serves a scope from a resolved provider must call it, and refuse the
        provider when it raises, before :meth:`open`."""


def resolve_blob_store(name: str) -> BlobStoreProvider:
    """The provider named *name* (``"local"`` or empty: the local backend). Raises ``LookupError``
    with the reason for a named provider that is missing, refused or unavailable: blob storage
    never falls back to another store."""
    name = name.strip()
    if name in ("", "local"):
        from hermes_blob_port_local import LocalBlobStoreProvider
        return LocalBlobStoreProvider()
    from plugins.blob_storage import BlobStorageRefused, load_blob_storage
    try:
        provider = load_blob_storage(name)
    except BlobStorageRefused as refused:
        raise LookupError(f"blob storage {name!r}: {refused}") from refused
    if provider is None:
        raise LookupError(f"blob storage {name!r}: no such blob storage provider")
    if not provider.is_available():
        raise LookupError(f"blob storage {name!r} is not available here")
    return provider
