"""The local blob storage backend: today's files under the profile home, behind the blob port.

:class:`LocalBlobStore` serves each file family at today's path under the profile home,
moved nowhere: the inbound and generated media caches (``cache/images``, ``cache/audio``,
``cache/videos``, ``cache/documents``, ``cache/screenshots``, honouring a populated legacy
directory as ``get_hermes_dir`` does), room attachments (``hosted-room-attachments``, the
directory beside ``state.db`` that ``gateway.hosted_room_attachments`` roots at), pets
(``pets/``) and skill bodies (``skills/``). Keys map onto those families: a key's first
components name the family, the rest name a file inside it.

Each operation is the same file access the code makes today, after one check of the pin:
the bound scope (``hermes_scope.current_scope()``) must be a local one whose agent is the
store's own profile home; with no scope bound the pin is the ambient profile
(``get_hermes_home()``), as every local caller has it today. Any other pin (another tenant,
another profile) is refused with ``BlobPinRefused`` before anything is read or written.

The store's identity is frozen when it opens: the profile home's canonical path, taken from
the path the store was opened with, never recomputed against a later working directory (a
relative path is refused). A local store holds one agent and only agent-level records, so a
write naming another level is refused with ``BlobLevelRefused`` (the documented local
exception); a local install's one participant is the local owner, whose conversation files
these paths serve at the agent level until profiles become agents.

A key that would leave its family's directory — ``..`` components, an absolute path, a
component with a separator in it, or a symlink planted inside the tree pointing outside it —
is refused with ``BlobKeyRefused``: the file it names through the kernel is resolved, and a
resolution outside the family root names no blob of this store. A link swapped into that place
between the check and the file's own open, or a hardlink to a file outside the tree, is the
machine's own owner rearranging their filesystem under a running process, which
``store_identity``'s threat model puts out of scope here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hermes_blob_port import (
    LEVELS, BlobKeyRefused, BlobLevelRefused, BlobNotFound, BlobPinRefused, BlobStore,
    BlobStoreProvider, parse_key,
)

#: Media cache families: key component → (``get_hermes_dir`` new subpath, legacy name), the
#: same arguments ``gateway.platforms.base`` resolves its cache directories with, so a
#: populated legacy directory is honoured exactly as today's callers honour it.
_MEDIA_FAMILIES = {
    "image": ("cache/images", "image_cache"),
    "audio": ("cache/audio", "audio_cache"),
    "video": ("cache/videos", "video_cache"),
    "document": ("cache/documents", "document_cache"),
    "screenshot": ("cache/screenshots", "browser_screenshots"),
}


def store_identity(home: Any) -> Path:
    """Threat model: the pin guards against Hermes code binding the wrong scope to a store, not
    against the machine's owner rearranging their own files under a running process. The local
    backend serves only the local tenant, whose person owns every profile on this machine, so
    moving a directory alias or renaming a profile directory after the store opened is out of
    scope: the identity is fixed once, here, at open time, from a path that must be absolute.
    A hosted backend never derives a tenant's identity from a filesystem path.

    The profile home's canonical path; ``BlobPinRefused`` for a home that is relative (which
    directory it named cannot be established now)."""
    named = os.fspath(home) if home is not None else None
    if not named or not os.path.isabs(named):
        raise BlobPinRefused(f"blob store home {home!r} does not name an absolute profile directory")
    return Path(named).resolve()


class LocalBlobStore(BlobStore):
    """Today's per-profile files behind :class:`hermes_blob_port.BlobStore`."""

    def __init__(self, home: Any) -> None:
        from hermes_constants import hermes_home_key
        self.home = store_identity(home)
        self._home_key = hermes_home_key(self.home)

    # --- the pin, the same rule the session store's local backend applies

    def _pin(self) -> None:
        from hermes_constants import get_hermes_home, hermes_home_key
        from hermes_scope import LOCAL_TENANT, current_scope
        scope = current_scope()
        if scope is not None and scope.tenant != LOCAL_TENANT:
            raise BlobPinRefused(f"a local blob store serves tenant {LOCAL_TENANT!r} only")
        agent = scope.agent if scope is not None else get_hermes_home()
        # Both sides canonical and absolute: a relative agent is the directory it names now.
        if hermes_home_key(agent) != self._home_key:
            raise BlobPinRefused("the bound agent is not this blob store's profile")

    def _level(self, level: str) -> None:
        """A write names its level; this store holds the agent level only."""
        if level not in LEVELS:
            raise BlobLevelRefused(f"{level!r} is not a blob ownership level")
        if level != "agent":
            raise BlobLevelRefused(
                f"this local blob store holds the agent level only, not {level!r}")

    # --- keys → today's paths

    def _family_root(self, family: str) -> Path:
        """The directory a key's family lives in, resolved under the frozen home: today's path
        for that family (``pets``, ``skills``, the attachment store beside ``state.db``, or a
        media cache, honouring a populated legacy directory as ``get_hermes_dir`` does)."""
        if family in ("pets", "skills"):
            return self.home / family
        if family == "attachments":
            return self.home / "hosted-room-attachments"
        subpath, legacy = _MEDIA_FAMILIES[family]
        from hermes_constants import get_hermes_dir
        return get_hermes_dir(subpath, legacy, home=self.home)

    def _locate(self, key: str) -> tuple[Path, Path]:
        """``(family root, the file the key names under it)``, both canonical. The family must
        be one this store serves (``BlobKeyRefused`` naming it), and the file the kernel
        resolves the key to must stay inside the family root: a symlink anywhere on the way
        pointing outside it makes the key name nothing of this store, never a file elsewhere."""
        components = parse_key(key)
        family = components[0]
        if components[0] == "media":
            kind = components[1] if len(components) > 1 else ""
            if kind not in _MEDIA_FAMILIES:
                raise BlobKeyRefused(f"a media key names its cache (one of {sorted(_MEDIA_FAMILIES)}), not {kind!r}")
            family, components = kind, components[2:]
        elif components[0] in ("pets", "skills", "attachments"):
            family, components = components[0], components[1:]
        else:
            raise BlobKeyRefused(
                f"a blob key of this store starts with a family it serves (media, attachments, pets, skills), not {components[0]!r}")
        root = self._family_root(family).resolve()
        candidate = root.joinpath(*components) if components else root
        resolved = candidate.resolve()
        if resolved != root and root not in resolved.parents:
            raise BlobKeyRefused(
                f"the blob key {key!r} leaves the {family} family: it names {resolved}, outside {root}")
        return root, resolved

    # --- the port

    def get(self, key: str) -> bytes:
        self._pin()
        _root, path = self._locate(key)
        try:
            return path.read_bytes()
        except (IsADirectoryError, FileNotFoundError, NotADirectoryError):
            raise BlobNotFound(key) from None

    def put(self, key: str, data: bytes, *, level: str = "agent") -> None:
        self._pin()
        self._level(level)
        _root, path = self._locate(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def delete(self, key: str, *, level: str = "agent") -> bool:
        self._pin()
        self._level(level)
        _root, path = self._locate(key)
        try:
            path.unlink()
            return True
        except (FileNotFoundError, IsADirectoryError):
            return False

    def exists(self, key: str) -> bool:
        self._pin()
        try:
            _root, path = self._locate(key)
        except BlobKeyRefused:
            raise
        return path.is_file()

    def list(self, prefix: str = "") -> list[str]:
        self._pin()
        if prefix.startswith(("/", "\\")):
            raise BlobKeyRefused(f"the blob prefix {prefix!r} is absolute; a key names a blob of the agent's namespace")
        trimmed = prefix.removesuffix("/")  # a prefix may end in "/"
        components = () if trimmed == "" else parse_key(trimmed)
        if components and components[0] not in ("media", "attachments", "pets", "skills"):
            raise BlobKeyRefused(
                f"a blob key of this store starts with a family it serves (media, attachments, pets, skills), "
                f"not {components[0]!r}")
        keys = self._list_all()
        return sorted(k for k in keys if k.startswith(prefix))

    def _list_all(self) -> list[str]:
        """Every key this store can name: each family's files under its root, skipping entries
        whose resolution leaves the family root (a symlink pointing outside names no blob of
        this store, so no key is listed that a read of it would then have to refuse)."""
        keys = []
        for kind in sorted(_MEDIA_FAMILIES):
            keys += self._walk(self._family_root(kind), f"media/{kind}/")
        for family in ("attachments", "pets", "skills"):
            keys += self._walk(self._family_root(family), f"{family}/")
        return keys

    def _walk(self, root: Path, stem: str) -> list[str]:
        root = root.resolve()
        if not root.is_dir():
            return []
        keys = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                path = Path(dirpath) / filename
                resolved = path.resolve()
                if resolved != root and root not in resolved.parents:
                    continue
                keys.append(stem + str(path.relative_to(root)).replace(os.sep, "/"))
        return keys


class LocalBlobStoreProvider(BlobStoreProvider):
    """Today's files: one profile home, holding that agent's records only."""

    levels = frozenset({"agent"})

    @property
    def name(self) -> str:
        return "local"

    def open(self, home: Any = None) -> BlobStore:
        """The store over *home*, or over the bound profile (``HERMES_HOME``). The home is
        frozen here (``store_identity``); the pin is checked by every operation."""
        if home is None:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
        return LocalBlobStore(home)

    def assert_isolating(self) -> None:
        """A local store is one profile home and refuses every pin but its own (``_pin``);
        there is no shared role that could bypass that."""
        return
