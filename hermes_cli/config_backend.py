"""Where the user config layer lives: the ``ConfigBackend`` seam (config-config design §4.1, D10/D11).

Every read and write of a profile's user ``config.yaml`` goes through the backend returned by
:func:`get_config_backend` (``scripts/check_config_yaml_readers.py`` and
``scripts/check_config_yaml_writers.py`` reject direct file access elsewhere). The key per call is
the hermes home (profile): one multiplexed process serves N profiles, and every config cache
already keys on the config path, so a backend keeps per-home state.

:class:`FileBackend` is the only backend in this build and is today's behaviour byte for byte:
the version is the ``file_signature`` stat tuple, reads are ``open`` + ``fast_safe_load`` and raise
exactly what those raise (``FileNotFoundError`` for an absent file, ``OSError``, YAML errors), and
writes are the ruamel round-trip writers in ``utils``. Callers keep their own error handling, so a
backend whose layer always exists simply never takes the "no file" branches.

The managed scope (``/etc/hermes``) is NOT a backend concern: it stays an overlay applied on top of
whatever user layer the backend returns.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Tuple, Union, runtime_checkable

CONFIG_FILENAME = "config.yaml"
BACKEND_ENV = "HERMES_CONFIG_BACKEND"

PathLike = Union[str, "os.PathLike[str]"]


@dataclass(frozen=True)
class UserLayer:
    """One read of a profile's user layer."""
    doc: Any                    # the parsed root as written (None for an empty document)
    version: Tuple[Any, ...]    # the cache signature this doc was read at (etag for a remote backend)
    locks: Mapping[str, str] = field(default_factory=dict)   # dotted prefix -> level that locks it
    provenance: str = ""        # where the doc came from, for diagnostics


@dataclass(frozen=True)
class Changes:
    """A write to one profile's user layer.

    ``document`` replaces the whole user-layer document (``atomic_config_write``: the file backend
    merges it onto the on-disk document, comment-preserving); ``set`` / ``unset`` are single dotted
    keys (``atomic_roundtrip_yaml_update`` semantics). A backend applies ``document`` first.
    """
    document: Optional[dict] = None
    set: Mapping[str, Any] = field(default_factory=dict)
    unset: Tuple[str, ...] = ()
    extra_content_on_create: Optional[str] = None


@runtime_checkable
class ConfigBackend(Protocol):
    name: str

    def read_user_layer(self, home: Path) -> UserLayer: ...

    def read_user_doc_readonly(self, home: Path) -> Any:
        """``read_user_layer(home).doc`` re-read only when ``version`` changes; never mutate it."""
        ...

    def version(self, home: Path) -> Tuple[Any, ...]:
        """Cheap cache signature; changes whenever the user layer does."""
        ...

    def exists(self, home: Path) -> bool: ...

    def write_changes(self, home: Path, changes: Changes) -> None: ...

    def locked(self, home: Path, dotted: str) -> Optional[str]:
        """The level that locks ``dotted`` (prefix-aware), or None."""
        ...

    def supports_file_tooling(self) -> bool:
        """Whether tools that copy/edit/back up the config FILE (profile clone, backup/restore,
        ``config edit``, last-known-good backups, ``migrate_config`` write-back) may run."""
        ...


class ConfigBackendUnavailable(SystemExit):
    """The selected backend cannot serve config. A ``SystemExit`` on purpose: many config readers
    fail open with ``except Exception`` → defaults, and serving defaults instead of the selected
    backend's config is exactly the silent failure this must not become."""


class FileBackend:
    """The user layer is ``<home>/config.yaml`` on local disk."""

    name = "file"

    # Path-level primitives: the file backend's whole behaviour, also used for explicit files
    # that are not a home's config.yaml (see ``_route``).
    @staticmethod
    def version_path(path: Path) -> Tuple[int, int, int, int]:
        from utils import file_signature
        return file_signature(path.stat())

    @staticmethod
    def read_path(path: Path) -> Any:
        from utils import fast_safe_load
        with open(path, encoding="utf-8-sig") as f:
            return fast_safe_load(f)

    @staticmethod
    def read_path_readonly(path: Path) -> Any:
        from utils import load_yaml_file_readonly
        return load_yaml_file_readonly(path)

    @staticmethod
    def write_path(path: Path, changes: Changes) -> None:
        from utils import atomic_roundtrip_yaml_save, atomic_roundtrip_yaml_update
        if changes.document is not None:
            atomic_roundtrip_yaml_save(path, changes.document, extra_content_on_create=changes.extra_content_on_create)
        for key, value in changes.set.items():
            atomic_roundtrip_yaml_update(path, key, value)
        for key in changes.unset:
            atomic_roundtrip_yaml_update(path, key, None)

    def config_path(self, home: Path) -> Path:
        return Path(home) / CONFIG_FILENAME

    def read_user_layer(self, home: Path) -> UserLayer:
        path = self.config_path(home)
        version = self.version_path(path)
        return UserLayer(doc=self.read_path(path), version=version, provenance=str(path))

    def read_user_doc_readonly(self, home: Path) -> Any:
        return self.read_path_readonly(self.config_path(home))

    def version(self, home: Path) -> Tuple[int, int, int, int]:
        return self.version_path(self.config_path(home))

    def exists(self, home: Path) -> bool:
        return self.config_path(home).exists()

    def write_changes(self, home: Path, changes: Changes) -> None:
        self.write_path(self.config_path(home), changes)

    def locked(self, home: Path, dotted: str) -> Optional[str]:
        return None  # a local file has no levels; the managed scope is a separate overlay

    def supports_file_tooling(self) -> bool:
        return True


_FILE_BACKEND = FileBackend()


def get_config_backend() -> ConfigBackend:
    """The backend selected by ``HERMES_CONFIG_BACKEND`` (default ``file``).

    Read from the environment on every call, before any config read, so no config value can
    select it (D11). Raises :class:`ConfigBackendUnavailable` for a backend this build lacks.
    """
    kind = os.environ.get(BACKEND_ENV, "").strip().lower() or "file"
    if kind == "file":
        return _FILE_BACKEND
    if kind == "remote":
        raise ConfigBackendUnavailable(
            f"{BACKEND_ENV}=remote: the remote config backend is not available in this build of "
            "Hermes. Unset it to use the local config.yaml.")
    raise ConfigBackendUnavailable(f"{BACKEND_ENV}={kind!r} is not a config backend (expected 'file').")


def _route(config_path: PathLike) -> Tuple[Any, Path, bool]:
    """``(backend, home_or_path, is_user_layer)`` for a config path.

    ``<home>/config.yaml`` is that home's user layer and goes to the selected backend. Any other
    file name is an explicit file (a test fixture, an import source), not a user layer: it is read
    and written with the file backend's primitives regardless of the selected backend.
    """
    path = Path(config_path)
    if path.name == CONFIG_FILENAME:
        return get_config_backend(), path.parent, True
    return _FILE_BACKEND, path, False


def read_config_doc(config_path: PathLike) -> Any:
    """Parsed root of a config file (None when empty); raises like ``open`` + ``fast_safe_load``."""
    backend, target, is_layer = _route(config_path)
    return backend.read_user_layer(target).doc if is_layer else backend.read_path(target)


def read_config_doc_readonly(config_path: PathLike) -> Any:
    """Signature-cached :func:`read_config_doc`; the result is shared — never mutate it."""
    backend, target, is_layer = _route(config_path)
    return backend.read_user_doc_readonly(target) if is_layer else backend.read_path_readonly(target)


def config_version(config_path: PathLike) -> Tuple[Any, ...]:
    """Cache signature of a config file; raises ``FileNotFoundError`` / ``OSError`` like ``stat``."""
    backend, target, is_layer = _route(config_path)
    return backend.version(target) if is_layer else backend.version_path(target)


def config_exists(config_path: PathLike) -> bool:
    backend, target, is_layer = _route(config_path)
    return backend.exists(target) if is_layer else target.exists()


def write_config_document(config_path: PathLike, document: dict, *, extra_content_on_create: Optional[str] = None) -> None:
    """Replace a config file's document (comment-preserving merge for the file backend)."""
    backend, target, is_layer = _route(config_path)
    changes = Changes(document=document, extra_content_on_create=extra_content_on_create)
    backend.write_changes(target, changes) if is_layer else backend.write_path(target, changes)


def write_config_key(config_path: PathLike, key_path: str, value: Any) -> None:
    """Set one dotted key (``None`` removes it) — ``atomic_roundtrip_yaml_update`` semantics."""
    backend, target, is_layer = _route(config_path)
    changes = Changes(unset=(key_path,)) if value is None else Changes(set={key_path: value})
    backend.write_changes(target, changes) if is_layer else backend.write_path(target, changes)


def supports_file_tooling() -> bool:
    """Gate for tools that copy, edit or back up the config FILE itself (§4.3, §4.7)."""
    return get_config_backend().supports_file_tooling()
