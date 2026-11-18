"""A two-level reference backend for the blob conformance suite.

Today's local backend stores the agent level only (the documented local exception), so the
suite's participant-level invariants need a backend that stores both: this one, a
hosted-shaped store over a directory tree whose keys are prefixed by the ownership key —
``<tenant>/participant/<principal>/<agent>/<key>`` for conversation files and
``<tenant>/agent/<agent>/<key>`` for the agent's own — exactly the shape a hosted bucket
backend takes. It is a test fixture, not a shipped backend: it exists so the suite's
participant invariants are proven against a real two-level store, and so a plugin author can
see the shape the suite assumes.

The pin is the bound scope, with no fallback: a hosted backend never serves an ambient
profile. A participant record is the bound principal's — the ownership key's subject comes
from the scope, never from the key.

A conversation file is session-owned data, so its writes are fenced the way every hosted
repository write is (``persistence.core.fences``): inside the write the store reads
:func:`~persistence.core.fences.current_turn_fence` for its own store key (the hosted pin key,
``persistence.StorePin(tenant, agent).key``, which ``agent.runtime_scope.bind_turn`` stamps on
the turn a hosted scope binds) and checks it with :func:`~persistence.core.fences.fence_holds`
against the session's lease on its :class:`SessionLeases`, before anything of it lands. Work
designed to outlive the turn (``agent.runtime_scope.outlive_turn``) carries an ``owner_only``
fence, held to the owner epoch alone. The leases model the turn-admission lease the hosted blob
backend would read from its own index.
"""

from __future__ import annotations

import os
from pathlib import Path

from hermes_blob_port import (
    LEVELS, BlobFenceRequired, BlobFenceSuperseded, BlobKeyRefused, BlobLevelRefused,
    BlobNotFound, BlobPinRefused, BlobStore, parse_key,
)
from hermes_scope import Fence, Principal, current_scope
from persistence.core.context import StorePin
from persistence.core.fences import current_turn_fence, fence_holds


class SessionLeases:
    """The turn-admission lease, as this fixture's stand-in for the hosted lease table: the
    owner's epoch (a takeover moves it) and each claimed session's runtime generation (the
    session's next claim moves it). One per agent; the store checks a turn's fence against it."""

    def __init__(self) -> None:
        self.epoch = 1
        self.generations: dict[str, int] = {}

    def claim(self, session: str) -> Fence:
        """Claim *session*'s next input, stamping and returning the lease's fence."""
        generation = self.generations.get(session, 0) + 1
        self.generations[session] = generation
        return Fence(self.epoch, generation)

    def take_over(self) -> None:
        """Another owner takes the agent over: every fence's owner-epoch half is gone."""
        self.epoch += 1

    def current(self, session: str) -> tuple[int, int | None]:
        """The lease as the store reads it inside a write: the owner epoch and *session*'s
        runtime generation (``None``: never claimed here)."""
        return self.epoch, self.generations.get(session)


class ReferenceBlobStore(BlobStore):
    """One tenant-agent pair's files under *root*, both levels, pinned to the bound scope."""

    levels = frozenset({"participant", "agent"})

    def __init__(self, root: Path, tenant: str, agent: str, leases: SessionLeases | None = None) -> None:
        #: The turn-admission leases the store checks conversation-file fences against; one
        #: per agent, shared by every store opened over it in a test.
        self.root, self.tenant, self.agent = root.resolve(), tenant, agent
        self.leases = leases if leases is not None else SessionLeases()
        #: The key turn fences on this store are issued under: the hosted pin's.
        self.store_key = StorePin(tenant, agent).key

    # --- the pin: the bound scope, nothing else

    def _pin(self):
        scope = current_scope()
        if scope is None:
            raise BlobPinRefused("a hosted-style blob store serves a bound scope only, and none is bound")
        if scope.tenant != self.tenant or str(scope.agent) != self.agent:
            raise BlobPinRefused("the bound scope is not this blob store's tenant and agent")

    def _subject(self, level: str) -> str:
        """The ownership key's subject for a write at *level*: the bound principal for a
        participant record, nothing (applies to all of the agent) for an agent record."""
        if level not in LEVELS:
            raise BlobLevelRefused(f"{level!r} is not a blob ownership level")
        scope = current_scope()
        if scope is None:  # the pin already refused this; the subject still comes from the scope
            raise BlobPinRefused("a participant record's subject is the bound scope's principal")
        if level == "agent":
            return ""
        if not isinstance(scope.actor, Principal):
            raise BlobLevelRefused(
                f"a participant-level blob belongs to a principal, and the bound actor is {scope.actor!r}")
        return scope.actor.id

    def _level_dir(self, level: str, subject: str) -> Path:
        if level == "agent":
            return self.root / self.tenant / "agent" / self.agent
        return self.root / self.tenant / "participant" / subject / self.agent

    def _write_fence(self, level: str) -> None:
        """The turn fence on a conversation-file write (``participant`` level): it is
        session-owned data, so it runs under the bound claimed turn, whose fence
        (``current_turn_fence`` for this store's key) this store checks against the session's
        lease on ``self.leases`` inside the write, before anything of it lands — the owner-epoch
        half alone for work designed to outlive the turn (``outlive_turn``: an ``owner_only``
        fence). No fence for this store (no turn bound, the owner's own operation, or a turn
        claimed on another store) refuses the write. Agent-level writes are never turn-fenced."""
        if level != "participant":
            return
        fence = current_turn_fence(self.store_key)
        if fence is None:
            raise BlobFenceRequired(
                "a conversation-file write is session-owned data and must run under its turn's "
                "fence; no turn claimed on this store is bound")
        epoch, generation = self.leases.current(fence.session_id)
        if not fence_holds(fence, epoch, generation):
            raise BlobFenceSuperseded(
                f"turn {fence.turn_id!r} of session {fence.session_id!r} no longer holds its "
                f"session's lease (fence epoch={fence.epoch}, generation={fence.generation}, "
                f"owner_only={fence.owner_only}); refusing its conversation-file write")

    def _locate(self, key: str, level_dir: Path) -> Path:
        """The file *key* names under *level_dir*, refused (``BlobKeyRefused``) when the key
        cannot name a file of that directory: escaping components, or a resolution (a symlink)
        pointing outside it."""
        components = parse_key(key)
        root = level_dir.resolve()
        resolved = root.joinpath(*components).resolve()
        if resolved != root and root not in resolved.parents:
            raise BlobKeyRefused(f"the blob key {key!r} leaves the agent's namespace: it names {resolved}")
        return resolved

    # --- the port

    def get(self, key: str) -> bytes:
        self._pin()
        # Reads resolve through the documented policy: the participant's own records first,
        # then the agent's.
        for level_dir in self._read_dirs():
            try:
                return self._locate(key, level_dir).read_bytes()
            except (IsADirectoryError, FileNotFoundError, NotADirectoryError):
                continue
        raise BlobNotFound(key)

    def put(self, key: str, data: bytes, *, level: str = "agent") -> None:
        self._pin()
        subject = self._subject(level)
        self._write_fence(level)
        path = self._locate(key, self._level_dir(level, subject))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def delete(self, key: str, *, level: str = "agent") -> bool:
        self._pin()
        subject = self._subject(level)
        self._write_fence(level)
        try:
            self._locate(key, self._level_dir(level, subject)).unlink()
            return True
        except (FileNotFoundError, IsADirectoryError):
            return False

    def exists(self, key: str) -> bool:
        self._pin()
        return any(self._locate(key, level_dir).is_file() for level_dir in self._read_dirs())

    def list(self, prefix: str = "") -> list[str]:
        self._pin()
        if prefix.startswith(("/", "\\")):
            raise BlobKeyRefused(f"the blob prefix {prefix!r} is absolute; a key names a blob of the agent's namespace")
        trimmed = prefix.removesuffix("/")
        if trimmed:
            parse_key(trimmed)  # a prefix obeys the same containment rules as a key
        keys = []
        for level_dir in self._read_dirs():
            if not level_dir.is_dir():
                continue
            for dirpath, _dirnames, filenames in os.walk(level_dir):
                for filename in filenames:
                    path = Path(dirpath) / filename
                    resolved = path.resolve()
                    if resolved == level_dir or level_dir in resolved.parents:
                        keys.append(str(path.relative_to(level_dir)).replace(os.sep, "/"))
        return sorted(k for k in keys if k.startswith(prefix))

    def _read_dirs(self) -> list[Path]:
        """The directories a read resolves through, in the documented order: the bound
        principal's participant records, then the agent's."""
        scope = current_scope()
        dirs = []
        if scope is not None and isinstance(scope.actor, Principal):
            dirs.append(self._level_dir("participant", scope.actor.id))
        dirs.append(self._level_dir("agent", ""))
        return dirs


def key_path(root: Path, tenant: str, agent: str, key: str) -> Path:
    """Where an agent-level *key* names its file on disk (the suite's planted-symlink test
    plants through this)."""
    store = ReferenceBlobStore(root, tenant, agent)
    return store._locate(key, store._level_dir("agent", ""))
