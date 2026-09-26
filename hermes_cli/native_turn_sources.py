"""Generic, profile-scoped sources of host-admitted native user turns.

Plugins register a source through :class:`hermes_cli.plugins.PluginContext`.
The host supplies an immutable view of a live session and, if a source has work,
receives an opaque lease. External queue, scheduling, and receipt policy remain
inside the plugin.
"""

from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, cast, final, runtime_checkable

from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)

NativeSurface = Literal["cli", "tui", "gateway"]
NativeAbortOutcome = Literal["not_sent", "canceled", "unknown"]
_ALLOWED_SURFACES = frozenset({"cli", "tui", "gateway"})
_ALLOWED_ABORT_OUTCOMES = frozenset({"not_sent", "canceled", "unknown"})


@dataclass(frozen=True, slots=True)
class NativeSessionView:
    """Canonical, immutable facts the host proved for one live session."""

    profile: str
    profile_home: Path
    session_id: str
    surface: NativeSurface
    compression_lineage: tuple[str, ...] = ()
    session_key: str | None = None
    owner_token: str | None = None
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        home = Path(self.profile_home).expanduser().resolve()
        profile = str(self.profile).strip()
        session_id = str(self.session_id).strip()
        surface = str(self.surface).strip().lower()
        if not profile or not session_id:
            raise ValueError("native session profile and session_id must be non-empty")
        if surface not in _ALLOWED_SURFACES:
            raise ValueError(f"unsupported native turn surface: {surface!r}")
        lineage = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in self.compression_lineage
                if str(item).strip()
            )
        )
        metadata = MappingProxyType({str(k): str(v) for k, v in dict(self.metadata).items()})
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "profile_home", home)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "surface", surface)
        object.__setattr__(self, "compression_lineage", lineage)
        object.__setattr__(self, "metadata", metadata)
        if self.session_key is not None:
            object.__setattr__(self, "session_key", str(self.session_key))
        if self.owner_token is not None:
            object.__setattr__(self, "owner_token", str(self.owner_token))


CommitCallback = Callable[[NativeSessionView], bool]
AbortCallback = Callable[[NativeAbortOutcome, str], None]


@dataclass(frozen=True, slots=True)
class NativeTurnLease:
    """Opaque plugin lease with callbacks at the host's pre-model boundary."""

    lease_id: str
    prompt: str
    commit: CommitCallback
    abort: AbortCallback
    display_kind: str = "native"

    def __post_init__(self) -> None:
        if not str(self.lease_id).strip():
            raise ValueError("native turn lease_id must be non-empty")
        if not self.prompt.strip():
            raise ValueError("native turn prompt must be non-empty")
        if not callable(self.commit) or not callable(self.abort):
            raise TypeError("native turn commit and abort must be callable")


@runtime_checkable
class NativeTurnSource(Protocol):
    """External producer polled only after the host establishes idle ownership."""

    name: str

    def poll(self, session: NativeSessionView) -> NativeTurnLease | None: ...

    def on_user_boundary(self, session: NativeSessionView, reason: str) -> None: ...


@dataclass(frozen=True, slots=True)
class RegisteredNativeTurnSource:
    source: NativeTurnSource
    surfaces: frozenset[str]
    plugin_id: str


_PROFILE_SOURCES_LOCK = threading.RLock()
_PROFILE_SOURCES: dict[str, dict[str, RegisteredNativeTurnSource]] = {}
_SESSION_BOUNDARY_LOCKS_LOCK = threading.Lock()
_SESSION_BOUNDARY_LOCKS: weakref.WeakValueDictionary[
    tuple[str, ...], threading.RLock
] = weakref.WeakValueDictionary()


def _profile_scope(profile_home: str | Path) -> str:
    return hermes_home_key(Path(profile_home).expanduser().resolve(strict=False))


def register_profile_source(
    profile_home: str | Path, registration: RegisteredNativeTurnSource
) -> Callable[[], None]:
    """Publish one manager-owned registration in its exact profile scope."""

    scope = _profile_scope(profile_home)
    with _PROFILE_SOURCES_LOCK:
        bucket = _PROFILE_SOURCES.setdefault(scope, {})
        if registration.source.name in bucket:
            raise ValueError(
                f"Native turn source '{registration.source.name}' is already registered"
            )
        bucket[registration.source.name] = registration

    def unregister() -> None:
        with _PROFILE_SOURCES_LOCK:
            current = _PROFILE_SOURCES.get(scope)
            if current is None:
                return
            if current.get(registration.source.name) is registration:
                current.pop(registration.source.name, None)
            if not current:
                _PROFILE_SOURCES.pop(scope, None)

    return unregister


def _profile_source_snapshot(
    profile_home: str | Path,
) -> dict[str, RegisteredNativeTurnSource]:
    scope = _profile_scope(profile_home)
    with _PROFILE_SOURCES_LOCK:
        return dict(_PROFILE_SOURCES.get(scope, {}))


@final
class NativeTurnAdmission:
    """Exactly-once adapter around a lease's commit/abort callbacks.

    An exception from ``commit`` is conservatively ``unknown``. A false return
    is ``not_sent``. Hosts may call ``abort_if_open`` from cancellation/failure
    cleanup without risking a duplicate callback.
    """

    def __init__(
        self,
        lease: NativeTurnLease,
        session: NativeSessionView,
        boundary_lock: threading.RLock | None = None,
    ):
        self.lease = lease
        self.session = session
        self._lock = threading.Lock()
        self._boundary_lock = boundary_lock or threading.RLock()
        self._state: str = "open"

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def commit(self) -> bool:
        with self._boundary_lock, self._lock:
            if self._state != "open":
                return self._state == "committed"
            try:
                accepted = self.lease.commit(self.session)
            except Exception as exc:
                self._state = "unknown"
                try:
                    self.lease.abort("unknown", f"native lease commit raised: {exc}")
                except Exception:
                    logger.warning("Native lease unknown abort failed", exc_info=True)
                return False
            if accepted is True:
                self._state = "committed"
                return True
            self._state = "not_sent"
            try:
                self.lease.abort("not_sent", "native lease commit declined admission")
            except Exception:
                logger.warning("Native lease not_sent abort failed", exc_info=True)
            return False

    def abort_if_open(self, outcome: NativeAbortOutcome, reason: str) -> bool:
        if outcome not in _ALLOWED_ABORT_OUTCOMES:
            raise ValueError(f"invalid native abort outcome: {outcome!r}")
        with self._boundary_lock, self._lock:
            if self._state != "open":
                return False
            self._state = outcome
            try:
                self.lease.abort(outcome, str(reason))
            except Exception:
                logger.warning("Native lease abort failed", exc_info=True)
            return True


def source_name(source: NativeTurnSource) -> str:
    name = str(getattr(source, "name", "")).strip()
    if not name:
        raise ValueError("native turn source must expose a non-empty name")
    if not callable(getattr(source, "poll", None)):
        raise TypeError("native turn source poll must be callable")
    if not callable(getattr(source, "on_user_boundary", None)):
        raise TypeError("native turn source on_user_boundary must be callable")
    return name


def normalize_surfaces(surfaces: tuple[str, ...] | list[str]) -> frozenset[str]:
    result = frozenset(str(value).strip().lower() for value in surfaces)
    if not result or not result <= _ALLOWED_SURFACES:
        unknown = sorted(result - _ALLOWED_SURFACES)
        detail = f": {', '.join(unknown)}" if unknown else ""
        raise ValueError(f"native turn surfaces must be a non-empty supported set{detail}")
    return result


def poll_registered_sources(
    registrations: Mapping[str, RegisteredNativeTurnSource],
    session: NativeSessionView,
    *,
    boundary_lock: threading.RLock | None = None,
) -> NativeTurnAdmission | None:
    """Poll a stable registration snapshot; source failures are fail-inert."""
    for registration in tuple(registrations.values()):
        if session.surface not in registration.surfaces:
            continue
        try:
            lease = cast(object, registration.source.poll(session))
        except Exception:
            logger.warning(
                "Native turn source %s poll failed", registration.plugin_id, exc_info=True
            )
            continue
        if lease is None:
            continue
        if not isinstance(lease, NativeTurnLease):
            logger.warning(
                "Native turn source %s returned an invalid lease", registration.plugin_id
            )
            continue
        return NativeTurnAdmission(lease, session, boundary_lock)
    return None


def fence_registered_sources(
    registrations: Mapping[str, RegisteredNativeTurnSource],
    session: NativeSessionView,
    reason: str,
) -> None:
    """Synchronously notify each applicable source before session identity changes."""
    for registration in tuple(registrations.values()):
        if session.surface not in registration.surfaces:
            continue
        try:
            registration.source.on_user_boundary(session, str(reason))
        except Exception:
            logger.warning(
                "Native turn source %s boundary failed", registration.plugin_id, exc_info=True
            )


def _profile_registrations(
    profile_home: Path,
) -> Mapping[str, RegisteredNativeTurnSource]:
    """Return the already-discovered registrations for one exact profile."""
    return _profile_source_snapshot(profile_home)


def _session_boundary_lock(session: NativeSessionView) -> threading.RLock:
    key = (
        hermes_home_key(session.profile_home),
        session.profile,
        session.surface,
        session.session_key or "",
        session.session_id,
        session.owner_token or "",
        *session.compression_lineage,
    )
    with _SESSION_BOUNDARY_LOCKS_LOCK:
        lock = _SESSION_BOUNDARY_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _SESSION_BOUNDARY_LOCKS[key] = lock
        return lock


def poll_native_turn(session: NativeSessionView) -> NativeTurnAdmission | None:
    """Poll sources installed in ``session``'s profile; no source is an inert default."""
    lock = _session_boundary_lock(session)
    with lock:
        return poll_registered_sources(
            _profile_registrations(session.profile_home),
            session,
            boundary_lock=lock,
        )


def fence_native_turn_sources(session: NativeSessionView, reason: str) -> None:
    """Synchronously fence native work in ``session``'s exact profile."""
    with _session_boundary_lock(session):
        fence_registered_sources(
            _profile_registrations(session.profile_home), session, reason
        )
