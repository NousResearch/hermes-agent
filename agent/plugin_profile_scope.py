"""Profile identity and transactional storage for plugin provider registries.

The gateway can serve several Hermes profiles in one Python process.  Provider
plugins are therefore process-global code but profile-local state.  This module
provides the narrow shared seam used by provider registries and plugin loading:

* :class:`ProfileKey` is a normalized immutable identity;
* :func:`bind_profile_key` selects an identity with ``ContextVar`` isolation;
* :func:`bound_to_profile` freezes that identity for delayed callbacks; and
* :func:`provider_registration_transaction` atomically rolls back registrations
  when one plugin load fails.

Legacy single-profile callers do not need to pass a key.  The current profile is
resolved lazily from ``HERMES_HOME`` through ``get_active_profile_name()``.
"""

from __future__ import annotations

import inspect
import os
import threading
from collections.abc import MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Generic, Iterator, List, Optional, TypeVar, Union

from hermes_constants import (
    get_default_hermes_root,
    reset_hermes_home_override,
    set_hermes_home_override,
)


@dataclass(frozen=True, order=True)
class ProfileKey:
    """Canonical, immutable key for profile-owned process state."""

    value: str

    def __post_init__(self) -> None:
        value = self.value
        if not isinstance(value, str):
            raise TypeError("profile key must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("profile key cannot be empty")
        if stripped[:7].casefold() == "custom:":
            normalized = "custom:" + stripped[7:]
        else:
            normalized = stripped.casefold()
        object.__setattr__(self, "value", normalized)

    def __str__(self) -> str:
        return self.value


ProfileKeyLike = Union[ProfileKey, str, os.PathLike[str]]
_BOUND_PROFILE_KEY: ContextVar[Optional[ProfileKey]] = ContextVar(
    "plugin_provider_profile_key", default=None
)


def _profile_key_from_runtime() -> ProfileKey:
    """Resolve the active runtime profile without freezing import-time state."""
    try:
        from hermes_cli.profiles import get_active_profile_name

        name = get_active_profile_name()
        if name and name != "custom":
            return ProfileKey(name)
    except Exception:
        pass

    # Custom HERMES_HOME paths must not all collapse to the key ``custom``.
    try:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home()).expanduser().resolve(strict=False)
        return ProfileKey(f"custom:{os.path.normcase(str(home))}")
    except Exception:
        return ProfileKey("default")


def normalize_profile_key(profile_key: Optional[ProfileKeyLike] = None) -> ProfileKey:
    """Return a canonical immutable profile key.

    ``None`` means the currently bound profile, or the active Hermes runtime
    profile when no explicit binding exists.  Explicit strings are normalized
    case-insensitively.  Paths are represented by their normalized absolute
    location so distinct custom homes cannot collide.
    """
    if profile_key is None:
        bound = _BOUND_PROFILE_KEY.get()
        return bound if bound is not None else _profile_key_from_runtime()
    if isinstance(profile_key, ProfileKey):
        return profile_key
    if isinstance(profile_key, os.PathLike):
        path = Path(profile_key).expanduser().resolve(strict=False)
        return ProfileKey(f"custom:{os.path.normcase(str(path))}")
    if not isinstance(profile_key, str):
        raise TypeError("profile key must be a ProfileKey, string, path, or None")
    return ProfileKey(profile_key)


def current_profile_key() -> ProfileKey:
    """Return the selected profile key, resolved once for this call."""
    return normalize_profile_key()


def selected_profile_key(profile_key: Optional[ProfileKeyLike] = None) -> ProfileKey:
    """Freeze an explicit or contextual profile identity for one operation."""
    return normalize_profile_key(profile_key)


def freeze_profile_key(profile_key: Optional[ProfileKeyLike] = None) -> ProfileKey:
    """Public alias emphasizing capture for delayed work."""
    return selected_profile_key(profile_key)


def set_profile_key(profile_key: ProfileKeyLike) -> Token:
    """Bind *profile_key* and return a token suitable for reset."""
    return _BOUND_PROFILE_KEY.set(normalize_profile_key(profile_key))


def reset_profile_key(token: Token) -> None:
    """Restore a prior profile-key binding."""
    _BOUND_PROFILE_KEY.reset(token)


def _profile_home_for_key(profile_key: ProfileKey) -> Path:
    """Resolve the config/state home owned by a canonical profile key."""
    if profile_key.value.startswith("custom:"):
        raw_path = profile_key.value[7:]
        if not raw_path:
            raise ValueError("custom profile key must include a path")
        return Path(raw_path)

    root = get_default_hermes_root()
    if profile_key.value == "default":
        return root
    return root / "profiles" / profile_key.value


@contextmanager
def bind_profile_key(profile_key: ProfileKeyLike) -> Iterator[ProfileKey]:
    """Select one profile identity and Hermes home in this context.

    Provider lookup can read config or invoke provider callbacks after the
    process has entered another profile's runtime scope.  Binding only the
    registry identity would then pair one profile's provider bucket with a
    different profile's ``config.yaml``.  Keep both ContextVars aligned while
    leaving the process-wide ``HERMES_HOME`` environment untouched.
    """
    key = normalize_profile_key(profile_key)
    profile_token = _BOUND_PROFILE_KEY.set(key)
    home_token = set_hermes_home_override(_profile_home_for_key(key))
    try:
        yield key
    finally:
        reset_hermes_home_override(home_token)
        _BOUND_PROFILE_KEY.reset(profile_token)


def bound_to_profile(
    callback: Callable, profile_key: Optional[ProfileKeyLike] = None
) -> Callable:
    """Return a callback that always runs with the profile selected now.

    This is the safe bridge for thread targets, futures, and other callbacks
    invoked after their originating profile runtime scope has exited.
    """
    key = selected_profile_key(profile_key)
    if inspect.iscoroutinefunction(callback):
        @wraps(callback)
        async def _async_bound(*args, **kwargs):
            with bind_profile_key(key):
                return await callback(*args, **kwargs)

        return _async_bound

    @wraps(callback)
    def _bound(*args, **kwargs):
        with bind_profile_key(key):
            return callback(*args, **kwargs)

    return _bound


_MISSING = object()


class _RegistrationTransaction:
    def __init__(self, profile_key: ProfileKey):
        self.profile_key = profile_key
        self._undo: List[Callable[[], None]] = []
        self.failed = False

    def record(self, profile_key: ProfileKey, undo: Callable[[], None]) -> None:
        if profile_key != self.profile_key:
            self.failed = True
            raise RuntimeError(
                "plugin registration transaction cannot mutate a different profile "
                f"({profile_key} != {self.profile_key})"
            )
        self._undo.append(undo)

    def rollback(self) -> None:
        for undo in reversed(self._undo):
            undo()
        self._undo.clear()


_ACTIVE_TRANSACTION: ContextVar[Optional[_RegistrationTransaction]] = ContextVar(
    "plugin_registration_transaction", default=None
)


def record_registration_undo(
    profile_key: ProfileKeyLike,
    undo: Callable[[], None],
) -> bool:
    """Attach a generation-safe undo callback to the active plugin transaction.

    Returns ``False`` when no transaction is active so ordinary launch-time
    registration keeps its historical behavior.  Callers own the compare-before-
    restore guard that prevents an undo from clobbering a later writer.
    """
    transaction = _ACTIVE_TRANSACTION.get()
    if transaction is None:
        return False
    transaction.record(selected_profile_key(profile_key), undo)
    return True


@contextmanager
def plugin_registration_transaction(
    profile_key: Optional[ProfileKeyLike] = None,
) -> Iterator[ProfileKey]:
    """Atomically apply all manager/profile-owned plugin publications."""
    key = selected_profile_key(profile_key)
    active = _ACTIVE_TRANSACTION.get()
    if active is not None:
        if active.profile_key != key:
            active.failed = True
            raise RuntimeError(
                "cannot nest a plugin registration transaction for a different profile"
            )
        try:
            yield key
        except BaseException:
            active.failed = True
            raise
        return

    transaction = _RegistrationTransaction(key)
    transaction_token = _ACTIVE_TRANSACTION.set(transaction)
    try:
        with bind_profile_key(key):
            try:
                yield key
            except BaseException:
                transaction.failed = True
                transaction.rollback()
                raise
            if transaction.failed:
                transaction.rollback()
                raise RuntimeError("plugin registration transaction failed closed")
    finally:
        _ACTIVE_TRANSACTION.reset(transaction_token)


# Backward-compatible name retained for provider-registry callers and tests.
provider_registration_transaction = plugin_registration_transaction


T = TypeVar("T")


class CurrentProfileProviderMapping(MutableMapping[str, T], Generic[T]):
    """Compatibility mapping exposing only the caller's current profile."""

    def __init__(self, registry: "ProfileProviderRegistry[T]"):
        self._registry = registry

    def __getitem__(self, name: str) -> T:
        value = self._registry.get(name)
        if value is None:
            raise KeyError(name)
        return value

    def __setitem__(self, name: str, provider: T) -> None:
        self._registry.register(name, provider)

    def __delitem__(self, name: str) -> None:
        if not self._registry.delete(name):
            raise KeyError(name)

    def __iter__(self):
        return iter(self._registry.snapshot())

    def __len__(self) -> int:
        return len(self._registry.snapshot())

    def clear(self) -> None:
        self._registry.clear_profile()


class ProfileProviderRegistry(Generic[T]):
    """Thread-safe profile-keyed provider map with transactional writes."""

    def __init__(self, *, normalize_name: Callable[[str], str] = lambda name: name.strip()):
        self._normalize_name = normalize_name
        self._providers: Dict[ProfileKey, Dict[str, T]] = {}
        self._generations: Dict[ProfileKey, Dict[str, int]] = {}
        self._next_generation = 0
        self._lock = threading.RLock()

    def _new_generation(self) -> int:
        self._next_generation += 1
        return self._next_generation

    @property
    def lock(self) -> threading.RLock:
        """Compatibility lock for legacy private registry readers."""
        return self._lock

    def compatibility_mapping(self) -> CurrentProfileProviderMapping[T]:
        """Return a mapping view scoped dynamically to the current profile."""
        return CurrentProfileProviderMapping(self)

    def register(
        self,
        name: str,
        provider: T,
        *,
        profile_key: Optional[ProfileKeyLike] = None,
    ) -> Optional[T]:
        key = selected_profile_key(profile_key)
        provider_name = self._normalize_name(name)
        transaction = _ACTIVE_TRANSACTION.get()
        with self._lock:
            bucket = self._providers.setdefault(key, {})
            generations = self._generations.setdefault(key, {})
            previous = bucket.get(provider_name, _MISSING)
            previous_generation = generations.get(provider_name, _MISSING)
            written_generation = self._new_generation()

            if transaction is not None:
                def _undo() -> None:
                    with self._lock:
                        current_bucket = self._providers.get(key)
                        current_generations = self._generations.get(key)
                        if (
                            current_bucket is None
                            or current_generations is None
                            or current_generations.get(provider_name, _MISSING)
                            != written_generation
                        ):
                            # A later writer owns the slot; rollback must not
                            # resurrect stale state over that committed value.
                            return
                        if previous is _MISSING:
                            current_bucket.pop(provider_name, None)
                            current_generations.pop(provider_name, None)
                        else:
                            current_bucket[provider_name] = previous  # type: ignore[assignment]
                            current_generations[provider_name] = previous_generation  # type: ignore[assignment]
                        if not current_bucket:
                            self._providers.pop(key, None)
                            self._generations.pop(key, None)

                transaction.record(key, _undo)

            bucket[provider_name] = provider
            generations[provider_name] = written_generation
        return None if previous is _MISSING else previous  # type: ignore[return-value]

    def get(
        self, name: str, *, profile_key: Optional[ProfileKeyLike] = None
    ) -> Optional[T]:
        key = selected_profile_key(profile_key)
        provider_name = self._normalize_name(name)
        with self._lock:
            return self._providers.get(key, {}).get(provider_name)

    def snapshot(
        self, *, profile_key: Optional[ProfileKeyLike] = None
    ) -> Dict[str, T]:
        key = selected_profile_key(profile_key)
        with self._lock:
            return dict(self._providers.get(key, {}))

    def list(self, *, profile_key: Optional[ProfileKeyLike] = None) -> List[T]:
        return list(self.snapshot(profile_key=profile_key).values())

    def delete(
        self, name: str, *, profile_key: Optional[ProfileKeyLike] = None
    ) -> bool:
        key = selected_profile_key(profile_key)
        provider_name = self._normalize_name(name)
        with self._lock:
            bucket = self._providers.get(key)
            if bucket is None or provider_name not in bucket:
                return False
            del bucket[provider_name]
            generations = self._generations.get(key)
            if generations is not None:
                generations.pop(provider_name, None)
            if not bucket:
                self._providers.pop(key, None)
                self._generations.pop(key, None)
            return True

    def clear_profile(
        self, *, profile_key: Optional[ProfileKeyLike] = None
    ) -> None:
        key = selected_profile_key(profile_key)
        with self._lock:
            self._providers.pop(key, None)
            self._generations.pop(key, None)

    def reset_for_tests(self) -> None:
        with self._lock:
            self._providers.clear()
            self._generations.clear()
            self._next_generation = 0
