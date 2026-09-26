"""Copy-on-write generation seam for the provider registry containers.

The module-level containers that describe which providers exist
(``PROVIDER_REGISTRY``, ``CANONICAL_PROVIDERS``, ``_PROVIDER_LABELS``, ...) are
bound AT THEIR DEFINITION SITES to the facade types below. A facade keeps its
``isinstance`` identity (``GuardedDict`` is a ``dict``, ``GuardedList`` a
``list``), so every existing ``from module import NAME`` reference keeps
working, but:

* every Python-level read is served from the committed :class:`Generation`, an
  immutable per-generation copy (mappingproxy / tuple / frozenset). A reader
  iterating while another thread registers a provider can never see a torn
  container or hit ``dictionary changed size during iteration``;
* every write is a copy-and-swap of that ONE container under a single
  ``RLock`` (compare-and-swap: the swap re-derives if another writer swapped
  since the optimistic build, so no update is lost);
* :func:`publish` swaps several containers in ONE generation reference swap,
  so a multi-surface registration (canonical list + label + parser name) is
  observable whole or not at all through a :func:`snapshot`.

Base storage MIRRORS the generation: after every swap, still under the lock,
the facade's own ``dict``/``list`` storage is rewritten with explicit
base-class calls. C-level consumers that read base storage directly
(``json.dumps`` without ``indent``, ``[] + x``, ``PyDict_Next``) therefore see
the same entries instead of an empty container.

Multi-container readers bind one generation with ``g = snapshot()`` and read
``g.PROVIDER_REGISTRY``, ``g.CANONICAL_PROVIDERS``, ... so every surface they
consult belongs to one committed state. Single-container readers need nothing.

The seam is provider-agnostic: it never imports provider code. Refresh
callbacks (:func:`register_refresh`) let a plugin register providers lazily,
right before a lookup that names them, without a restart; with no callback
registered :func:`refresh` is a no-op.

Lock order: seam lock -> anything else. The lock is only ever held for the
in-memory swap and mirror write; never across imports, discovery, network or
callbacks.
"""

from __future__ import annotations

import collections.abc
import logging
import sys
import threading
import types
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "FACADES",
    "Generation",
    "GuardedDict",
    "GuardedList",
    "GuardedSet",
    "SeamCollision",
    "current",
    "publish",
    "refresh",
    "register_refresh",
    "restore",
    "snapshot",
]

REFRESH_REASONS = frozenset({"picker", "typed", "request"})
_COMMITTED = "committed"

_lock = threading.RLock()
FACADES: dict[str, Any] = {}
_OWNERS: dict[str, str] = {}
_KINDS: dict[str, str] = {}

# Test instrumentation only: when set, called with "build" (after a write's
# optimistic build, before the lock) and "swap" (inside the lock, before the
# reference swap). Barrier tests park a writer here; production leaves it None.
_park: Optional[Callable[[str], None]] = None


class SeamCollision(ValueError):
    """A :func:`publish` delta would replace an entry that already holds a different value."""


class Generation:
    """One committed, immutable state of every registered container.

    Containers are reachable as attributes named after their module-level name
    (``g.PROVIDER_REGISTRY``) or by subscription (``g["_REGISTRY"]``).
    ``committed`` maps a lane to the frozenset of names published for it.
    """

    __slots__ = ("_containers", "committed")

    def __init__(self, containers: Mapping[str, Any], committed: Mapping[str, frozenset]):
        object.__setattr__(self, "_containers", types.MappingProxyType(dict(containers)))
        object.__setattr__(self, "committed", types.MappingProxyType(dict(committed)))

    def __getattr__(self, name: str) -> Any:
        try:
            return self._containers[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, name: str) -> Any:
        return self._containers[name]

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"Generation is frozen ({name})")

    def get(self, name: str, default: Any = None) -> Any:
        return self._containers.get(name, default)

    def names(self) -> tuple[str, ...]:
        return tuple(self._containers)

    def __repr__(self) -> str:
        return f"<Generation containers={len(self._containers)} lanes={sorted(self.committed)}>"


_current = Generation({}, {})


def current() -> Generation:
    """Return the committed generation."""
    return _current


_snapshot_hooks: list[Callable[[], None]] = []


def add_snapshot_hook(hook: Callable[[], None]) -> None:
    """Run ``hook`` before every :func:`snapshot`.

    For containers that materialize lazily on first read: a snapshot reads the
    generation directly, so it must latch the same trigger a facade read would.
    A hook must be cheap once latched.
    """
    if hook not in _snapshot_hooks:
        _snapshot_hooks.append(hook)


def snapshot() -> Generation:
    """Pin one generation for a multi-container read (``g = snapshot()``).

    Returns the object :func:`current` returns (after the snapshot hooks ran),
    so readers that bind a snapshot see exactly what readers of the module
    globals see. The one exception keeps that equivalence: if an owning
    module's global no longer names its facade (a test rebinding it with
    ``monkeypatch.setattr``), the snapshot carries the rebound object for that
    container instead of the facade's generation.
    """
    for hook in tuple(_snapshot_hooks):
        hook()
    gen = _current
    rebound = None
    for name, facade in tuple(FACADES.items()):
        module = sys.modules.get(_OWNERS[name])
        bound = getattr(module, name, facade) if module is not None else facade
        if bound is not facade:
            rebound = rebound or {}
            rebound[name] = bound
    return gen if rebound is None else _with(gen, rebound)


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

def _freeze(kind: str, data: Any) -> Any:
    if kind == "dict":
        return types.MappingProxyType(dict(data))
    if kind == "list":
        return tuple(data)
    return frozenset(data)


def _mirror(name: str, data: Any) -> None:
    """Rewrite a facade's base storage to ``data``. Caller holds the lock."""
    facade = FACADES.get(name)
    if facade is None:
        return
    kind = _KINDS[name]
    if kind == "dict":
        dict.clear(facade)
        dict.update(facade, data)
    elif kind == "list":
        list.clear(facade)
        list.extend(facade, data)
    # sets: GuardedSet has no base storage.


def _with(base: Generation, updates: Mapping[str, Any], committed: Optional[Mapping] = None) -> Generation:
    containers = dict(base._containers)
    containers.update(updates)
    return Generation(containers, base.committed if committed is None else committed)


def _transact(build: Callable[[Generation], Optional[tuple[Generation, tuple[str, ...]]]]) -> Generation:
    """Optimistic build, then compare-and-swap under the seam lock.

    ``build(base)`` returns ``(next_generation, changed_names)`` or ``None`` for
    a no-op. It must be pure (it may run twice): if another writer swapped
    since the optimistic build, the plan is re-derived from the fresh base
    inside the lock, so no update is lost.
    """
    global _current
    base = _current
    plan = build(base)
    if _park is not None:
        _park("build")
    with _lock:
        if _current is not base:
            base = _current
            plan = build(base)
        if plan is None:
            return base
        if _park is not None:
            _park("swap")
        gen, changed = plan
        _current = gen
        for name in changed:
            _mirror(name, gen[name])
        return _current


def _write_one(name: str, derive: Callable[[Any], Any]) -> Generation:
    """Copy-and-swap one container. ``derive(frozen) -> new_frozen`` (``None`` = no-op)."""

    def build(base: Generation):
        new = derive(base[name])
        if new is None:
            return None
        return _with(base, {name: new}), (name,)

    return _transact(build)


def _register(facade: Any, owner: str, name: str, kind: str, data: Any) -> None:
    """Bind ``facade`` as the container ``name`` and seed its generation entry."""
    global _current
    frozen = _freeze(kind, data)
    with _lock:
        FACADES[name] = facade
        _OWNERS[name] = owner
        _KINDS[name] = kind
        _current = _with(_current, {name: frozen})


def owner_of(name: str) -> str:
    """Module name that defines the container ``name``."""
    return _OWNERS[name]


def publish(delta: Mapping[str, Any]) -> Generation:
    """Publish additions to several containers in ONE generation swap.

    ``delta`` maps a container name to the entries to add: a mapping for a dict
    container (insert; an existing key with an EQUAL value is a no-op, with a
    different value it is a :class:`SeamCollision`), a sequence for a list
    container (entries not already present are appended), an iterable for a
    set container (union). The reserved key ``"committed"`` maps a lane to the
    names committed for it; if every such name is already committed the call
    is an idempotent no-op returning the current generation.

    Returns the generation that holds the delta. Nothing is swapped if any part
    of the delta is rejected.
    """
    unknown = [n for n in delta if n != _COMMITTED and n not in _KINDS]
    if unknown:
        raise KeyError(f"unknown seam container(s): {sorted(unknown)}")

    def build(base: Generation):
        committed_delta = delta.get(_COMMITTED) or {}
        if committed_delta and all(
            frozenset(names) <= base.committed.get(lane, frozenset())
            for lane, names in committed_delta.items()
        ):
            return None
        updates: dict[str, Any] = {}
        for name, entries in delta.items():
            if name == _COMMITTED:
                continue
            kind = _KINDS[name]
            cur = base[name]
            if kind == "dict":
                added = {}
                for key, value in dict(entries).items():
                    if key in cur:
                        if cur[key] == value:
                            continue
                        raise SeamCollision(f"{name}[{key!r}] already holds a different entry")
                    added[key] = value
                if added:
                    updates[name] = types.MappingProxyType({**cur, **added})
            elif kind == "list":
                added_list: list = []
                for entry in entries:
                    if entry not in cur and entry not in added_list:
                        added_list.append(entry)
                if added_list:
                    updates[name] = cur + tuple(added_list)
            else:
                extra = frozenset(entries) - cur
                if extra:
                    updates[name] = cur | extra
        committed = None
        if committed_delta:
            committed = dict(base.committed)
            for lane, names in committed_delta.items():
                committed[lane] = committed.get(lane, frozenset()) | frozenset(names)
        if not updates and committed is None:
            return None
        return _with(base, updates, committed), tuple(updates)

    return _transact(build)


def restore(gen: Generation) -> None:
    """Swap back to a saved generation (``saved = current()``) and rewrite every mirror.

    For callers that must undo registrations made in between: test isolation,
    and ``hermes plugins doctor``, which imports a plugin copy and then removes
    what it registered.
    """
    global _current
    with _lock:
        _current = gen
        for name in FACADES:
            try:
                data = gen[name]
            except KeyError:
                continue
            _mirror(name, data)


def _reset(*names: str) -> None:
    """Empty the named containers (test-isolation support, like :func:`restore`)."""
    restore(_with(_current, {n: _freeze(_KINDS[n], ()) for n in names}))


# ---------------------------------------------------------------------------
# Facades
# ---------------------------------------------------------------------------

def _copy_protocol(cls):
    """``copy``/``deepcopy``/``pickle`` of a facade yield a plain builtin, never a second facade."""

    def __copy__(self):
        fn, args = self.__reduce_ex__(4)[:2]
        return fn(*args)

    def __deepcopy__(self, memo):
        import copy

        fn, args = self.__reduce_ex__(4)[:2]
        return copy.deepcopy(fn(*args), memo)

    cls.__copy__ = __copy__
    cls.__deepcopy__ = __deepcopy__
    return cls


@_copy_protocol
class GuardedDict(dict):
    """``dict`` facade over one generation container (mirror storage)."""

    __slots__ = ("_seam_name",)

    def __init__(self, owner: str, name: str, data: Any = ()):
        dict.__init__(self)
        seed = dict(data)
        dict.update(self, seed)
        self._seam_name = name
        _register(self, owner, name, "dict", seed)

    def _data(self) -> Mapping:
        return _current[self._seam_name]

    # -- reads ------------------------------------------------------------
    def __getitem__(self, key):
        return self._data()[key]

    def __iter__(self):
        return iter(self._data())

    def __len__(self):
        return len(self._data())

    def __contains__(self, key):
        return key in self._data()

    def get(self, key, default=None):
        return self._data().get(key, default)

    def keys(self):
        return self._data().keys()

    def values(self):
        return self._data().values()

    def items(self):
        return self._data().items()

    def copy(self):
        return dict(self._data())

    def __eq__(self, other):
        return dict(self._data()) == other

    def __ne__(self, other):
        return dict(self._data()) != other

    def __or__(self, other):
        return dict(self._data()) | other

    def __ror__(self, other):
        return dict(other) | dict(self._data())

    def __repr__(self):
        return repr(dict(self._data()))

    def __reduce_ex__(self, protocol):
        return (dict, (dict(self._data()),))

    __hash__ = None  # type: ignore[assignment]

    # -- writes (copy-and-swap of this one container) -----------------------
    def _swap(self, fn):
        def derive(cur):
            merged = dict(cur)
            if fn(merged) is False:
                return None
            return types.MappingProxyType(merged)

        return _write_one(self._seam_name, derive)

    def __setitem__(self, key, value):
        self._swap(lambda d: d.__setitem__(key, value))

    def setdefault(self, key, default=None):
        self._swap(lambda d: False if key in d else d.__setitem__(key, default))
        return self._data()[key]

    def update(self, *args, **kwargs):
        pairs = dict(*args, **kwargs)
        if pairs:
            self._swap(lambda d: d.update(pairs))

    def __ior__(self, other):
        self.update(other)
        return self

    def __delitem__(self, key):
        if key not in self._data():
            raise KeyError(key)
        self._swap(lambda d: (d.pop(key, None), None)[1])

    _MISSING = object()

    def pop(self, key, default=_MISSING):
        data = self._data()
        if key not in data:
            if default is GuardedDict._MISSING:
                raise KeyError(key)
            return default
        value = data[key]
        self._swap(lambda d: (d.pop(key, None), None)[1])
        return value

    def popitem(self):
        data = self._data()
        if not data:
            raise KeyError("popitem(): dictionary is empty")
        key = next(reversed(tuple(data)))
        return key, self.pop(key)

    def clear(self):
        if self._data():
            self._swap(lambda d: d.clear())


@_copy_protocol
class GuardedList(list):
    """``list`` facade over one generation container (mirror storage)."""

    __slots__ = ("_seam_name",)

    def __init__(self, owner: str, name: str, data: Any = ()):
        list.__init__(self)
        seed = list(data)
        list.extend(self, seed)
        self._seam_name = name
        _register(self, owner, name, "list", seed)

    def _data(self) -> tuple:
        return _current[self._seam_name]

    # -- reads ------------------------------------------------------------
    def __getitem__(self, item):
        value = self._data()[item]
        return list(value) if isinstance(item, slice) else value

    def __iter__(self):
        return iter(self._data())

    def __len__(self):
        return len(self._data())

    def __contains__(self, item):
        return item in self._data()

    def __reversed__(self):
        return reversed(self._data())

    def index(self, *args):
        return self._data().index(*args)

    def count(self, item):
        return self._data().count(item)

    def copy(self):
        return list(self._data())

    def __eq__(self, other):
        return list(self._data()) == other

    def __ne__(self, other):
        return list(self._data()) != other

    def __add__(self, other):
        return list(self._data()) + other

    def __mul__(self, n):
        return list(self._data()) * n

    __rmul__ = __mul__

    def __repr__(self):
        return repr(list(self._data()))

    def __reduce_ex__(self, protocol):
        return (list, (list(self._data()),))

    __hash__ = None  # type: ignore[assignment]

    # -- writes (copy-and-swap) ---------------------------------------------
    def _swap(self, fn):
        out = []

        def derive(cur):
            work = list(cur)
            out[:] = [fn(work)]
            return tuple(work)

        _write_one(self._seam_name, derive)
        return out[0] if out else None

    def append(self, entry):
        self._swap(lambda w: w.append(entry))

    def extend(self, entries):
        added = tuple(entries)
        if added:
            self._swap(lambda w: w.extend(added))

    def __iadd__(self, entries):
        self.extend(entries)
        return self

    def __imul__(self, n):
        self._swap(lambda w: w.__imul__(n) and None)
        return self

    def insert(self, index, entry):
        self._swap(lambda w: w.insert(index, entry))

    def __setitem__(self, index, value):
        self._swap(lambda w: w.__setitem__(index, value))

    def __delitem__(self, index):
        self._swap(lambda w: w.__delitem__(index))

    def pop(self, index=-1):
        return self._swap(lambda w: w.pop(index))

    def remove(self, entry):
        self._swap(lambda w: w.remove(entry))

    def clear(self):
        self._swap(lambda w: w.clear())

    def sort(self, *, key=None, reverse=False):
        self._swap(lambda w: w.sort(key=key, reverse=reverse))

    def reverse(self):
        self._swap(lambda w: w.reverse())


class GuardedSet(collections.abc.Set):
    """Read-only-``Set`` facade with ``add``/``discard``.

    Deliberately NOT a ``set`` subclass: ``set(x)`` / ``s | x`` on a real set
    subclass copy its hash table directly and skip ``__iter__``. It has no base
    storage, so it has nothing to mirror.
    """

    __slots__ = ("_seam_name",)

    def __init__(self, owner: str, name: str, data: Any = ()):
        self._seam_name = name
        _register(self, owner, name, "set", data)

    def _data(self) -> frozenset:
        return _current[self._seam_name]

    def __contains__(self, item) -> bool:
        return item in self._data()

    def __iter__(self):
        return iter(self._data())

    def __len__(self) -> int:
        return len(self._data())

    def __repr__(self) -> str:
        return repr(set(self._data()))

    def copy(self) -> set:
        return set(self._data())

    @classmethod
    def _from_iterable(cls, iterable) -> set:
        # Set-ABC operators (``|``, ``&``, ``-``) build their result through
        # this hook; the result is a plain set, never a second facade.
        return set(iterable)

    def __reduce_ex__(self, protocol):
        return (set, (set(self._data()),))

    def __copy__(self):
        return set(self._data())

    def __deepcopy__(self, memo):
        return set(self._data())

    def add(self, item) -> None:
        _write_one(self._seam_name, lambda cur: None if item in cur else cur | {item})

    def update(self, *iterables) -> None:
        extra = frozenset().union(*iterables)
        if extra:
            _write_one(self._seam_name, lambda cur: None if extra <= cur else cur | extra)

    def discard(self, item) -> None:
        _write_one(self._seam_name, lambda cur: cur - {item} if item in cur else None)

    def remove(self, item) -> None:
        if item not in self._data():
            raise KeyError(item)
        self.discard(item)


# ---------------------------------------------------------------------------
# Refresh callbacks
# ---------------------------------------------------------------------------

_refresh_callbacks: list[Callable[[str, Optional[str]], None]] = []
_refresh_state = threading.local()


def register_refresh(cb: Callable[[str, Optional[str]], None]) -> None:
    """Register a refresh callback ``cb(reason, name)``; idempotent per callable."""
    with _lock:
        if cb not in _refresh_callbacks:
            _refresh_callbacks.append(cb)


def refresh(reason: str, name: Optional[str] = None) -> None:
    """Give registered callbacks a chance to publish providers before a lookup.

    ``reason`` is ``picker`` (a full listing is about to be built), ``typed``
    (the user typed ``provider:model``; ``name`` is the provider part) or
    ``request`` (a runtime resolution for ``name``). Same-thread re-entry — a
    callback whose own work reaches a refresh trigger — is a no-op. A callback
    failure is logged and never propagates into the caller's lookup.
    """
    if reason not in REFRESH_REASONS:
        raise ValueError(f"unknown refresh reason: {reason!r}")
    if not _refresh_callbacks or getattr(_refresh_state, "depth", 0):
        return
    _refresh_state.depth = 1
    try:
        for cb in tuple(_refresh_callbacks):
            try:
                cb(reason, name)
            except Exception:
                logger.warning("provider refresh callback failed (reason=%s)", reason, exc_info=True)
    finally:
        _refresh_state.depth = 0
