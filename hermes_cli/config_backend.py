"""Where the user config layer lives: the ``ConfigBackend`` seam (config-config design §4.1, D10/D11).

Every read and write of a profile's user ``config.yaml`` goes through the backend returned by
:func:`get_config_backend` (``scripts/check_config_yaml_readers.py`` and
``scripts/check_config_yaml_writers.py`` reject direct file access elsewhere). The key per call is
the hermes home (profile): one multiplexed process serves N profiles, and every config cache
already keys on the config path, so a backend keeps per-home state.

:class:`FileBackend` (the default) is today's behaviour byte for byte:
the version is the ``file_signature`` stat tuple, reads are ``open`` + ``fast_safe_load`` and raise
exactly what those raise (``FileNotFoundError`` for an absent file, ``OSError``, YAML errors), and
writes are the ruamel round-trip writers in ``utils``. Callers keep their own error handling, so a
backend whose layer always exists simply never takes the "no file" branches. Every other kind
(``HERMES_CONFIG_BACKEND=<kind>``, e.g. ``remote``) is the package ``plugins/config_backends/<kind>/``,
whose ``get_backend()`` returns the instance; this module names no backend beyond that.

The managed scope (``/etc/hermes``) is NOT a backend concern: it stays an overlay applied on top of
whatever user layer the backend returns.
"""
from __future__ import annotations

import contextlib
import contextvars
import importlib
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Union, runtime_checkable

CONFIG_FILENAME = "config.yaml"
BACKEND_ENV = "HERMES_CONFIG_BACKEND"

PathLike = Union[str, "os.PathLike[str]"]


@dataclass(frozen=True)
class UserLayer:
    """One read of a profile's user layer."""
    doc: Any                    # the parsed root as written (None for an empty document)
    version: tuple[Any, ...]    # the cache signature this doc was read at (etag for a remote backend)
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
    unset: tuple[str, ...] = ()
    extra_content_on_create: Optional[str] = None


@runtime_checkable
class ConfigBackend(Protocol):
    name: str

    def read_user_doc(self, home: Path) -> Any:
        """The parsed user layer as written (None for an empty document); version checks are separate."""
        ...

    def read_user_doc_readonly(self, home: Path) -> Any:
        """``read_user_doc(home)`` re-read only when ``version`` changes; never mutate it."""
        ...

    def version(self, home: Path) -> tuple[Any, ...]:
        """Cheap change signature; changes whenever the user layer does."""
        ...

    def cache_version(self, home: Path, version: tuple[Any, ...]) -> tuple[Any, ...]:
        """``version`` (just taken) plus whatever a cache needs to tell apart writes it cannot see."""
        ...

    def exists(self, home: Path) -> bool: ...

    def probe_readable(self, home: Path) -> None:
        """Raise ``OSError`` while the user layer cannot be read; no parse."""
        ...

    def write_changes(self, home: Path, changes: Changes) -> None: ...

    def locked(self, home: Path, dotted: str) -> Optional[str]:
        """The level that locks ``dotted`` (prefix-aware), or None."""
        ...

    def supports_file_tooling(self) -> bool:
        """Whether tools that copy/edit/back up the config FILE (profile clone, backup/restore,
        ``config edit``, last-known-good backups, ``migrate_config`` write-back) may run."""
        ...

    def honors_managed_config(self) -> bool:
        """Whether the managed ``/etc/hermes/config.yaml`` overlay applies (D18: not in remote mode)."""
        ...

    def boot(self, home: Path) -> None:
        """Called once per ``load_hermes_dotenv`` after the ``.env`` loads and before the first
        config read (design §4.4): a backend that fetches config does it here, failing closed."""
        ...

    def protected_env_names(self) -> frozenset:
        """Env names no external secret source may supply (the plane credential, D32)."""
        ...

    def deployment_env_names(self) -> frozenset:
        """Non-secret env names that select and address this backend (see :func:`deployment_env_names`)."""
        ...

    def bootstrap_env_names(self) -> frozenset:
        """Env names published from the launch home's and the managed ``.env`` before the first
        config read (design §4.4 steps 1-2): what this backend needs to select, address and
        authenticate itself before any config exists."""
        ...

    def preserve_read_base(self) -> contextlib.AbstractContextManager:
        """A context in which a read does not become the base a later document save is diffed
        against (see :func:`keeping_read_base`)."""
        ...

    def doctor_warnings(self, home: Path) -> tuple:
        """``(warning, hint, issue)`` triples ``hermes doctor`` reports after a successful read."""
        ...

    def apply_in_memory(self, home: Path, doc: dict) -> None:
        """Replace the user layer in memory only (in-memory migration, D12). Only called for a
        backend without file tooling; nothing is sent or written."""
        ...

    def describe(self, home: Path) -> str:
        """One line naming where the user layer comes from and its state (status line, doctor)."""
        ...

    def destination(self, home: Path) -> str:
        """Where a write lands, for a user-facing "saved in ..." line (no state, no network)."""
        ...


class ConfigBackendUnavailable(SystemExit):
    """The selected backend cannot serve config. A ``SystemExit`` on purpose: many config readers
    fail open with ``except Exception`` → defaults, and serving defaults instead of the selected
    backend's config is exactly the silent failure this must not become."""


class ConfigWriteError(RuntimeError):
    """A config write the backend refused or could not complete; nothing was changed. ``code`` is
    a machine code (the plane's ``error`` for a remote refusal)."""

    def __init__(self, message: str, *, code: str = "config_write_failed") -> None:
        super().__init__(message)
        self.code = code


class ConfigLockedError(ConfigWriteError):
    """The write touches a path locked by an upper level (D7, D10)."""

    def __init__(self, path: str, locked_by: str, message: Optional[str] = None) -> None:
        super().__init__(message or f"{path} is locked by the {locked_by} level", code="config_key_locked")
        self.path = path
        self.locked_by = locked_by


class ConfigValueError(ConfigWriteError):
    """A value the backend cannot store (a YAML-only type, a secret literal); nothing was sent."""

    def __init__(self, message: str, *, code: str = "config_value_invalid") -> None:
        super().__init__(message, code=code)


class FileBackend:
    """The user layer is ``<home>/config.yaml`` on local disk."""

    name = "file"

    # Path-level primitives: the file backend's whole behaviour, also used for explicit files
    # that are not a home's config.yaml (see ``_route``).
    @staticmethod
    def version_path(path: Path) -> tuple[int, int, int, int]:
        st = path.stat()  # before the import: a missing file must not need the utils stack (PM runtime)
        from utils import file_signature
        return file_signature(st)

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

    def read_user_doc(self, home: Path) -> Any:
        return self.read_path(self.config_path(home))

    def read_user_doc_readonly(self, home: Path) -> Any:
        return self.read_path_readonly(self.config_path(home))

    def version(self, home: Path) -> tuple[int, int, int, int]:
        return self.version_path(self.config_path(home))

    def cache_version(self, home: Path, version: tuple[Any, ...]) -> tuple[Any, ...]:
        # utils.file_cache_version's rule. ``version`` is config_version's opaque token, so the
        # racy test takes the file's own signature.
        from utils import racy_content_digest
        return (*version, racy_content_digest(self.config_path(home), self.version(home)))

    def exists(self, home: Path) -> bool:
        return self.config_path(home).exists()

    def probe_readable(self, home: Path) -> None:
        path = self.config_path(home)
        with open(path, "rb") as f:
            f.read(1)

    def write_changes(self, home: Path, changes: Changes) -> None:
        self.write_path(self.config_path(home), changes)

    def locked(self, home: Path, dotted: str) -> Optional[str]:
        return None  # a local file has no levels; the managed scope is a separate overlay

    def supports_file_tooling(self) -> bool:
        return True

    def honors_managed_config(self) -> bool:
        return True

    def boot(self, home: Path) -> None:
        return None

    def protected_env_names(self) -> frozenset:
        return frozenset()

    def deployment_env_names(self) -> frozenset:
        return frozenset()

    def bootstrap_env_names(self) -> frozenset:
        return frozenset()

    def preserve_read_base(self) -> contextlib.AbstractContextManager:
        return contextlib.nullcontext()

    def doctor_warnings(self, home: Path) -> tuple:
        return ()

    def apply_in_memory(self, home: Path, doc: dict) -> None:
        raise NotImplementedError("the file backend persists migrations to config.yaml")

    def describe(self, home: Path) -> str:
        return str(self.config_path(home))

    def destination(self, home: Path) -> str:
        return str(self.config_path(home))


_FILE_BACKEND = FileBackend()
# Reentrant: publishing the deployment can import modules whose import-time config read selects
# the backend again on the SAME thread (``_BOOTSTRAP_OWNER``); every other thread waits for it.
_BOOTSTRAP_LOCK = threading.RLock()
_BOOTSTRAPPED = False
_BOOTSTRAP_OWNER: Optional[int] = None


def _bootstrap_names(kind: str) -> frozenset:
    """The env names backend *kind* (as the ``.env`` layers select it) needs before the first read;
    none for a kind this build lacks (selection refuses it afterwards)."""
    try:
        return _backend_for(kind).bootstrap_env_names()
    except ConfigBackendUnavailable:
        return frozenset()


def _ensure_bootstrap_env() -> None:
    """Once per process, before the first selection: the deployment that selects, addresses and
    authenticates the backend comes from the process env, the launch home's ``.env`` and the
    managed ``.env`` (design §4.4 steps 1-2, D26/D32) — also when the first config read comes before
    ``load_hermes_dotenv`` (``hermes_cli.config`` reads config at import time)."""
    if not _BOOTSTRAPPED:
        _bootstrap(None, None, once=True)


def bootstrap_deployment(home: Optional[PathLike] = None, project_env: Optional[PathLike] = None) -> None:
    """Publish the deployment from an explicit dotenv load's layers (``load_hermes_dotenv``: its
    home and project ``.env``) before that load's first config read. Runs even when the
    once-per-process bootstrap already ran: an earlier one did not know this load's project file."""
    _bootstrap(Path(home) if home is not None else None,
               Path(project_env) if project_env is not None else None, once=False)


def _bootstrap(home: Optional[Path], project_env: Optional[Path], *, once: bool) -> None:
    global _BOOTSTRAPPED, _BOOTSTRAP_OWNER
    with _BOOTSTRAP_LOCK:  # another thread's bootstrap in progress: wait for it to publish
        if (once and _BOOTSTRAPPED) or _BOOTSTRAP_OWNER == threading.get_ident():
            return  # done, or this thread's own bootstrap re-entering (import-time config read)
        _BOOTSTRAP_OWNER = threading.get_ident()
        try:
            from hermes_cli.env_loader import apply_config_bootstrap_env
            # names resolved only for the kind the .env layers select
            apply_config_bootstrap_env(_bootstrap_names, home, project_env)
        except ImportError:  # no usable python-dotenv (a bare installer interpreter): no .env loads either
            pass
        finally:
            # Complete only once the deployment is published: a concurrent first reader that saw
            # the flag earlier would select (and read) the file backend of a remote deployment.
            _BOOTSTRAP_OWNER = None
            _BOOTSTRAPPED = True


def get_config_backend() -> ConfigBackend:
    """The backend selected by ``HERMES_CONFIG_BACKEND`` (default ``file``).

    Read from the environment on every call, before any config read, so no config value can
    select it (D11). Raises :class:`ConfigBackendUnavailable` for a backend this build lacks.
    """
    _ensure_bootstrap_env()
    return _backend_for(os.environ.get(BACKEND_ENV, "").strip().lower() or "file")


# A kind is one package name under plugins/config_backends/: never a dotted or relative path.
_KIND_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_BACKENDS_PACKAGE = "plugins.config_backends"


def _backend_for(kind: str) -> ConfigBackend:
    """``file``, or the instance ``plugins.config_backends.<kind>.get_backend()`` returns. The
    package is imported directly, never through the general plugin loader (which reads config, D11)."""
    if kind == "file":
        return _FILE_BACKEND
    unknown = f"{BACKEND_ENV}={kind!r} is not a config backend this build provides (expected 'file' or 'remote')."
    if not _KIND_RE.fullmatch(kind):
        raise ConfigBackendUnavailable(unknown)
    name = f"{_BACKENDS_PACKAGE}.{kind}"
    module = sys.modules.get(name)
    if module is None:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            if exc.name not in (name, _BACKENDS_PACKAGE, "plugins"):
                raise  # the backend exists but is broken: show that, not "unknown"
            raise ConfigBackendUnavailable(unknown) from exc
    get_backend = getattr(module, "get_backend", None)
    if not callable(get_backend):
        raise ConfigBackendUnavailable(unknown)
    return get_backend()


class _ExplicitFile(FileBackend):
    """A config file that is not a home's user layer (a test fixture, an import source): the file
    backend's behaviour on that exact path, whichever backend is selected."""

    def __init__(self, path: Path):
        self._path = path

    def config_path(self, home: Path) -> Path:
        return self._path


def _route(config_path: PathLike) -> tuple[ConfigBackend, Path]:
    """``(backend, home)`` serving a config path through the one ``ConfigBackend`` surface.

    ``<home>/config.yaml`` is that home's user layer and goes to the selected backend. Any other
    file name is an explicit file, served by the file backend on that path.
    """
    path = Path(config_path)
    if path.name == CONFIG_FILENAME:
        return get_config_backend(), path.parent
    return _ExplicitFile(path), path.parent


def local_file_layer(config_path: PathLike) -> tuple[ConfigBackend, Path]:
    """``(backend, home)`` for the LOCAL file at *config_path*, whichever backend is selected.

    Only for PM's dependency runtime: it publishes plugin selections to the local file and
    compare-and-swaps those bytes, and it carries neither the remote backend's client nor the
    plane credential. Everything else goes through :func:`_route`."""
    path = Path(config_path)
    return _ExplicitFile(path), path.parent


def read_config_doc(config_path: PathLike) -> Any:
    """Parsed root of a config file (None when empty); raises like ``open`` + ``fast_safe_load``."""
    backend, home = _route(config_path)
    return backend.read_user_doc(home)


def read_config_doc_readonly(config_path: PathLike) -> Any:
    """Signature-cached :func:`read_config_doc`; the result is shared — never mutate it."""
    backend, home = _route(config_path)
    return backend.read_user_doc_readonly(home)


def config_version(config_path: PathLike) -> tuple[Any, ...]:
    """Cache signature of a config file; raises ``FileNotFoundError`` / ``OSError`` like ``stat``."""
    backend, home = _route(config_path)
    return backend.version(home)


def config_cache_version(config_path: PathLike) -> tuple[Any, ...]:
    """Signature for a cache of a config file's parse: :func:`config_version` can stay equal across a
    same-tick same-size rewrite, which would pin the older parse for the process lifetime."""
    version = config_version(config_path)  # the one stat; its errors (and test seams) stay config_version's
    backend, home = _route(config_path)
    return backend.cache_version(home, version)


def config_cache_version_or_none(config_path: PathLike) -> Optional[tuple[Any, ...]]:
    """:func:`config_cache_version`, or None when the config is absent or cannot be read."""
    try:
        return config_cache_version(config_path)
    except OSError:
        return None


def config_version_or_none(config_path: PathLike) -> Optional[tuple[Any, ...]]:
    """:func:`config_version`, or None when the config is absent or cannot be stat'ed."""
    try:
        return config_version(config_path)
    except OSError:
        return None


def version_token(version: Optional[tuple[Any, ...]]) -> float:
    """A number that changes with a backend ``version`` (0 = no config), for clients that only
    compare successive values: the file backend's version leads with ``st_mtime_ns``, kept as seconds.
    Any other int-led version is scaled the same way, which is fine: only changes are compared."""
    if not version:
        return 0
    if isinstance(version[0], int):
        return version[0] / 1_000_000_000
    import zlib
    return float(zlib.crc32(repr(version).encode("utf-8")) or 1)


def probe_config_readable(config_path: PathLike) -> None:
    """Raise ``OSError`` while a config file cannot be read; no parse."""
    backend, home = _route(config_path)
    backend.probe_readable(home)


def config_exists(config_path: PathLike) -> bool:
    backend, home = _route(config_path)
    return backend.exists(home)


def write_config_document(config_path: PathLike, document: dict, *, extra_content_on_create: Optional[str] = None) -> None:
    """Replace a config file's document (comment-preserving merge for the file backend)."""
    backend, home = _route(config_path)
    backend.write_changes(home, Changes(document=document, extra_content_on_create=extra_content_on_create))


def config_destination(config_path: PathLike) -> str:
    """Where a write to *config_path* lands, for "✓ Set ... in <here>" lines: the file path for
    the file backend and explicit files, the backend's own name for the user layer otherwise."""
    backend, home = _route(config_path)
    return backend.destination(home)


def write_config_key(config_path: PathLike, key_path: str, value: Any) -> None:
    """Set one dotted key (``None`` removes it) — ``atomic_roundtrip_yaml_update`` semantics."""
    write_config_keys(config_path, {key_path: value})


def write_config_keys(config_path: PathLike, updates: Mapping[str, Any]) -> None:
    """Set several related dotted keys (``None`` removes one) as ONE write.

    A remote backend sends them as one CAS PATCH, so a refusal of any of them (a lock, a value
    check) changes none of them: a grouped edit such as a model switch (default + provider + ...)
    never lands half-applied. The file backend applies them key by key, as before."""
    backend, home = _route(config_path)
    backend.write_changes(home, Changes(set={k: v for k, v in updates.items() if v is not None},
                                        unset=tuple(k for k, v in updates.items() if v is None)))


def keeping_read_base(read):
    """Run ``read()`` (a pre-write guard read) without making it the read a later document save
    is diffed against: the caller's document came from ITS earlier read (Remote Config). A
    context that never read keeps the guard read. No-op for the file backend."""
    with get_config_backend().preserve_read_base():
        return read()


# Dotted prefixes a document save in this context must write whole: a lock on a changed key under
# one refuses the save (ConfigLockedError) instead of being silently omitted from it.
_REFUSE_LOCKED_OMISSIONS: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "config_refuse_locked_omissions", default=())


@contextlib.contextmanager
def refusing_locked_omissions(*prefixes: str):
    """Within the block, a document save that changes a key under one of *prefixes* (dotted, e.g.
    ``"model"``) and finds it locked raises :class:`ConfigLockedError` and sends nothing, instead
    of dropping the locked key and landing the rest (Remote Config bulk-save rule). Checked against
    the locks the save itself sees, so a lock published after an earlier preflight still refuses;
    a CAS retry re-checks every key it sends. No-op for the file backend (no locks)."""
    token = _REFUSE_LOCKED_OMISSIONS.set(_REFUSE_LOCKED_OMISSIONS.get() + tuple(prefixes))
    try:
        yield
    finally:
        _REFUSE_LOCKED_OMISSIONS.reset(token)


def refused_locked_omission_prefixes() -> tuple[str, ...]:
    return _REFUSE_LOCKED_OMISSIONS.get()


def write_changes(config_path: PathLike, changes: Changes) -> None:
    """One :class:`Changes` write. Unlike :func:`write_config_keys`, a ``None`` in ``set`` stores
    an explicit null (Remote Config: "no value", masking an inherited scalar); the file backend
    writes it as YAML null."""
    backend, home = _route(config_path)
    backend.write_changes(home, changes)


def deployment_env_names() -> frozenset:
    """The selected backend's deployment env names (which backend, which plane, which instance).

    The deployment is a property of the instance, not of a profile, but it usually arrives in the
    launch profile's ``.env``, which a child acting for ANOTHER profile drops as launch residue
    (``strip_launch_profile_env``). Such a child re-reads only its own profile's ``.env`` and would
    silently select the file backend, so these names are never residue. Credentials are not part
    of it: the child resolves its own, and fails closed without them."""
    return get_config_backend().deployment_env_names()


def supports_file_tooling() -> bool:
    """Gate for tools that copy, edit or back up the config FILE itself (§4.3, §4.7)."""
    return get_config_backend().supports_file_tooling()


def require_file_tooling(what: str) -> None:
    """Refuse *what* (a file-copy tool) with a clear message when the backend has no config file."""
    if not supports_file_tooling():
        raise ConfigBackendUnavailable(
            f"{what} copies the local config.yaml, and the {get_config_backend().name!r} config "
            "backend has none. It is not available with this backend.")
