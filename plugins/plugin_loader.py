"""Shared directory-plugin loader for ``plugins/<kind>/<name>/`` discovery packages
(cron_providers, context_engine, memory): import ``__init__.py`` by path with siblings
pre-registered so relative imports work, then extract the provider via ``register(ctx)``
or an ABC-subclass fallback."""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import logging
import sys
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

_log = logging.getLogger(__name__)

_PLUGINS_ROOT = Path(__file__).parent
_MODULE_LOAD_LOCKS: dict[str, threading.RLock] = {}
_MODULE_LOAD_LOCKS_GUARD = threading.Lock()
# Inside bounded_load_wait(), how long a second caller waits for another thread's in-flight load of
# the same module. Past it the module is marked stalled (a hung import) and later bounded callers
# refuse at once until it finishes. Other callers (agent builds) wait for the load to finish.
_CONCURRENT_LOAD_WAIT_SECS = 10.0
_BOUNDED_WAIT: ContextVar[bool] = ContextVar("plugin_load_bounded_wait", default=False)
_STALLED_LOADS: set[str] = set()
_LOAD_OWNERS: dict[str, int] = {}  # module -> thread id running its load
_LOAD_WAITERS: dict[int, str] = {}  # thread id -> module whose load it waits for


def _waits_for(tid: int) -> set:
    """Threads *tid* is blocked on: a loader lock owner, or an importlib module-lock owner."""
    owners = {_LOAD_OWNERS.get(_LOAD_WAITERS.get(tid, ""))}
    blocked = getattr(importlib._bootstrap, "_blocking_on", {}).get(tid)  # list on 3.12+, lock on 3.11
    owners.update(getattr(lk, "owner", None) for lk in (blocked if isinstance(blocked, list) else [blocked]))
    return owners - {None}


def _load_would_deadlock(me: int) -> bool:
    """Whether the wait graph (loader locks + importlib's module locks) leads back to *me*.
    importlib cannot see the loader lock in its own deadlock check, so walk both kinds here."""
    seen, todo = set(), list(_waits_for(me))
    while todo:
        tid = todo.pop()
        if tid == me:
            return True
        if tid not in seen:
            seen.add(tid)
            todo.extend(_waits_for(tid))
    return False


@contextlib.contextmanager
def bounded_load_wait():
    """For per-turn callers that must not stall on another thread's hung plugin import: their
    loads give up (None) after ``_CONCURRENT_LOAD_WAIT_SECS`` instead of waiting it out."""
    token = _BOUNDED_WAIT.set(True)
    try:
        yield
    finally:
        _BOUNDED_WAIT.reset(token)


def _module_load_lock(module_name: str) -> threading.RLock:
    with _MODULE_LOAD_LOCKS_GUARD:
        return _MODULE_LOAD_LOCKS.setdefault(module_name, threading.RLock())


def register_synthetic_package(name: str, search_locations: list[str]) -> None:
    """Register an empty package shell so ``<name>.<child>`` relative imports resolve."""
    if name in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = search_locations
    sys.modules[name] = importlib.util.module_from_spec(spec)


def user_plugins_dir() -> Optional[Path]:
    """Return ``$HERMES_HOME/plugins/`` or None if unavailable."""
    try:
        from hermes_constants import get_hermes_home
        d = get_hermes_home() / "plugins"
        return d if d.is_dir() else None
    except Exception:
        return None


def iter_plugin_dirs(root: Path) -> list[Path]:
    """Sorted child dirs of *root* that have an ``__init__.py`` (skips ``_``/``.`` names)."""
    if not root.is_dir():
        return []
    dirs: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.name.startswith(("_", ".")):
            continue
        try:
            if child.is_dir() and (child / "__init__.py").exists():
                dirs.append(child)
        except OSError as exc:  # one mode-000 / ACL-denied child must not abort the listing
            _log.warning("Skipping unreadable plugin directory %s: %s", child, exc)
    return dirs


def read_plugin_description(plugin_dir: Path) -> str:
    """Return ``description`` from ``plugin.yaml`` (empty string if absent/unreadable)."""
    try:
        from utils import fast_safe_load

        with open(plugin_dir / "plugin.yaml", encoding="utf-8-sig") as f:
            meta = fast_safe_load(f) or {}
        return meta.get("description", "")
    except Exception:
        return ""


def _new_module(name: str, file: Path, search_locations: Optional[list[str]] = None) -> Optional[Any]:
    """spec -> module -> sys.modules[name] (NOT executed); None if no spec."""
    spec = importlib.util.spec_from_file_location(
        name, str(file), submodule_search_locations=search_locations)
    if not spec:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    return mod


def _exec(mod: Any, logger: Optional[logging.Logger] = None) -> bool:
    """Exec a ``_new_module`` module (None -> False); False + debug-log if it raised. The sys.modules
    entry stays on failure; callers needing a clean retry pop it themselves."""
    if mod is None:
        return False
    try:
        mod.__spec__.loader.exec_module(mod)
        return True
    except Exception as e:
        if logger:
            logger.debug("Failed to exec_module %s: %s", mod.__name__, e)
        return False


def load_plugin_module(module_name: str, plugin_dir: Path, *, parents: tuple[str, ...],
                       logger: logging.Logger, synthetic_namespace: Optional[str] = None) -> Optional[Any]:
    """Import ``plugin_dir/__init__.py`` as *module_name* (reusing sys.modules when loaded).
    Order matters: parents first (relative imports need them), then siblings as ``module_name.<stem>``
    (so ``from ._x import Y`` resolves), then the module. Finally child is bound onto parent and
    siblings onto module — the shape normal imports produce, which monkeypatch relies on."""
    init_file = plugin_dir / "__init__.py"
    if not init_file.exists():
        return None
    if synthetic_namespace:  # user code: never imported in-process under plugins.isolation: host
        from hermes_cli.plugin_isolation import in_process_import_refusal
        refusal = in_process_import_refusal(f"plugin {plugin_dir.name!r} (loaded as {module_name})")
        if refusal:
            logger.warning("%s", refusal)
            return None
    # _new_module publishes a package shell before executing it so sibling relative
    # imports work. Serialize the complete load so another thread cannot observe
    # that half-built shell as a loaded plugin.
    # The wait polls for a cross-thread cycle through importlib's module locks; in a cycle, accept the
    # partial module like importlib does. It is bounded only inside bounded_load_wait() (per-turn callers).
    lock, me = _module_load_lock(module_name), threading.get_ident()
    deadline = float("inf") if not _BOUNDED_WAIT.get() else time.monotonic() + (
        0 if module_name in _STALLED_LOADS else _CONCURRENT_LOAD_WAIT_SECS)
    _LOAD_WAITERS[me] = module_name
    try:
        while not lock.acquire(timeout=0.05):
            if _load_would_deadlock(me):
                logger.debug("Concurrent circular load of %s; using the partial module", module_name)
                return sys.modules.get(module_name)
            if time.monotonic() >= deadline:
                _STALLED_LOADS.add(module_name)
                logger.warning("Skipping plugin %s: another thread's load of it has not finished "
                               "(import still running after %.0fs)", module_name, _CONCURRENT_LOAD_WAIT_SECS)
                return None
    finally:
        _LOAD_WAITERS.pop(me, None)
    outermost = module_name not in _LOAD_OWNERS  # the RLock re-enters on this thread's recursive loads
    _LOAD_OWNERS.setdefault(module_name, me)
    try:
        return _load_plugin_module_locked(module_name, plugin_dir, init_file, parents, logger,
                                          synthetic_namespace)
    finally:
        if outermost:
            _LOAD_OWNERS.pop(module_name, None)
            _STALLED_LOADS.discard(module_name)
        lock.release()


def _load_plugin_module_locked(module_name: str, plugin_dir: Path, init_file: Path,
                               parents: tuple[str, ...], logger: logging.Logger,
                               synthetic_namespace: Optional[str]) -> Optional[Any]:
    # A synthetic package shell has no __file__; only reuse modules loaded from disk.
    cached = sys.modules.get(module_name)
    if cached is not None and getattr(cached, "__file__", None):
        return cached
    for parent in parents:
        parent_path = _PLUGINS_ROOT.joinpath(*parent.split(".")[1:])
        if parent not in sys.modules and (parent_path / "__init__.py").exists():
            _exec(_new_module(parent, parent_path / "__init__.py", [str(parent_path)]))
    if synthetic_namespace:
        register_synthetic_package(synthetic_namespace, [])
    # Reserve the name before siblings exec so their relative imports resolve.
    mod = _new_module(module_name, init_file, [str(plugin_dir)])
    if mod is None:
        return None
    loaded_submodules = []
    for sub_file in plugin_dir.glob("*.py"):
        full_sub_name = f"{module_name}.{sub_file.stem}"
        if sub_file.name == "__init__.py" or full_sub_name in sys.modules:
            continue
        sub_mod = _new_module(full_sub_name, sub_file)
        if _exec(sub_mod, logger):
            loaded_submodules.append((sub_file.stem, sub_mod))
        else:
            sys.modules.pop(full_sub_name, None)
    if not _exec(mod, logger):
        sys.modules.pop(module_name, None)
        return None
    parent_name, child_name = module_name.rsplit(".", 1)
    parent_mod = sys.modules.get(parent_name)
    if parent_mod is not None:
        setattr(parent_mod, child_name, mod)
    for sub_name, sub_mod in loaded_submodules:
        setattr(mod, sub_name, sub_mod)
    return mod


class NoopPluginContext:
    """Base for fake ``register(ctx)`` contexts: no-op registrations except the one a subclass overrides."""

    def _noop(self, *args, **kwargs):
        pass

    register_tool = register_hook = register_cli_command = register_memory_provider = _noop


class _NamedProviderCollector(NoopPluginContext):
    """Fake plugin context that captures one provider-kind ``register_<method>(provider)`` call."""

    def __init__(self, register_method: str) -> None:
        self.provider = None
        setattr(self, register_method, self._capture)

    def _capture(self, provider):
        self.provider = provider


class NamedProviderKind:
    """Discovery for one provider plugin kind, addressed by explicit name: bundled
    ``plugins/<kind_dir>/<name>/`` then user ``$HERMES_HOME/plugins/<name>/`` (bundled wins on
    collision). The named storage kinds share this shape (``plugins/session_storage``,
    ``plugins/memory_storage``); the local backend of each kind is core, never discovered here.

    Providers load in-process only: under ``plugins.isolation: host`` a user provider is refused
    (the *refused* exception class, with the reason) rather than returned as a host proxy, whose
    port objects would be opaque and whose calls would not carry the caller's scope. A user
    provider's module is named after its canonical directory, so same-named providers in two
    profiles (and their sibling modules) never share a module or its import-time state.

    ``base_cls`` and ``refused`` are callables so this module never imports a port module at load
    time (it is imported while config/auth still load)."""

    def __init__(self, *, kind_dir: str, base_cls: Callable[[], type], register_method: str,
                 markers: tuple[str, ...], kind: str, noun: str, refused: Callable[[], type],
                 logger: logging.Logger):
        self.plugins_dir = _PLUGINS_ROOT / kind_dir
        self.user_namespace = f"_hermes_user_{kind_dir}"
        self._base_cls = base_cls
        self._register_method = register_method
        self._markers = tuple(markers)
        self._kind = kind
        self._noun = noun
        self._refused = refused
        self.logger = logger

    def is_provider_dir(self, path: Path) -> bool:
        """Cheap text heuristic: ``__init__.py`` mentions the kind's contract."""
        init_file = path / "__init__.py"
        if not init_file.exists():
            return False
        try:
            source = init_file.read_text(errors="replace", encoding="utf-8-sig")[:8192]
        except OSError:
            return False
        return any(marker in source for marker in self._markers)

    def find_provider_dir(self, name: str) -> Optional[Path]:
        """Resolve a provider name to its directory (bundled first, then user-installed)."""
        bundled = self.plugins_dir / name
        if bundled.is_dir() and (bundled / "__init__.py").exists():
            return bundled
        user_dir = user_plugins_dir()
        user = user_dir / name if user_dir else None
        return user if user and user.is_dir() and self.is_provider_dir(user) else None

    def load(self, name: str) -> Optional[Any]:
        """Load a provider instance by name; ``None`` if not found or it fails to load. Raises
        the kind's refused exception for a user provider under ``plugins.isolation: host``."""
        provider_dir = self.find_provider_dir(name)
        if not provider_dir:
            self.logger.debug("%s '%s' not found in bundled or user plugins", self._kind, name)
            return None
        if not self._is_bundled(provider_dir):
            from hermes_cli.plugin_isolation import in_process_import_refusal
            refusal = in_process_import_refusal(f"{self._kind.lower()} {name!r}")
            if refusal:
                self.logger.error("%s", refusal)
                raise self._refused()(refusal)
        return load_named(name, provider_dir, self._load_from_dir, kind=self._kind, noun=self._noun,
                          logger=self.logger)

    def _is_bundled(self, provider_dir: Path) -> bool:
        return provider_dir.parent == self.plugins_dir

    def _load_from_dir(self, provider_dir: Path) -> Optional[Any]:
        """Import a provider module in-process and extract its provider (``register(ctx)`` or a
        subclass). ``load_plugin_module`` refuses a user module under host isolation as well."""
        name = provider_dir.name
        namespace = None if self._is_bundled(provider_dir) else self._user_namespace(provider_dir)
        module_name = f"{self.plugins_dir.name}.{name}" if namespace is None else f"{namespace}.{name}"
        if namespace is not None:
            register_synthetic_package(self.user_namespace, [])
        mod = load_plugin_module(
            module_name, provider_dir, parents=("plugins", f"plugins.{self.plugins_dir.name}"),
            logger=self.logger, synthetic_namespace=namespace)
        return mod and instance_from_module(
            mod, collector=_NamedProviderCollector(self._register_method), collected_attr="provider",
            base_cls=self._base_cls(), name=name, logger=self.logger)

    def _user_namespace(self, provider_dir: Path) -> str:
        """The synthetic package a user provider loads under: one per canonical provider
        directory, so profile A's ``<name>`` and profile B's ``<name>`` are different modules
        (a directory alias of the same profile resolves to the same one)."""
        import hashlib
        digest = hashlib.sha256(str(provider_dir.resolve()).encode("utf-8", "surrogatepass")).hexdigest()[:16]
        return f"{self.user_namespace}.p{digest}"


def instance_from_module(mod: Any, *, collector: Any, collected_attr: str, base_cls: type, name: str,
                         logger: logging.Logger) -> Optional[Any]:
    """Extract the provider instance: ``register(ctx)`` first, then any ``base_cls`` subclass."""
    if hasattr(mod, "register"):
        try:
            mod.register(collector)
            instance = getattr(collector, collected_attr)
            if instance:
                return instance
        except Exception as e:
            logger.debug("register() failed for %s: %s", name, e)
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name, None)
        if isinstance(attr, type) and issubclass(attr, base_cls) and attr is not base_cls:
            with contextlib.suppress(Exception):
                return attr()
    return None


def load_named(name: str, plugin_dir: Path, load_from_dir: Callable[[Path], Optional[Any]], *, kind: str,
               noun: str, logger: logging.Logger) -> Optional[Any]:
    """Shared body of ``load_<kind>(name)``: load from *plugin_dir*, warn + None on failure."""
    try:
        instance = load_from_dir(plugin_dir)
    except Exception as e:
        logger.warning("Failed to load %s '%s': %s", kind.lower(), name, e)
        return None
    if not instance:
        logger.warning("%s '%s' loaded but no %s instance found", kind, name, noun)
    return instance or None


def probe_availability(load: Callable[[], Optional[Any]]) -> bool:
    """True iff *load()* returns an instance whose ``is_available()`` (if any) is truthy."""
    try:
        instance = load()
        return instance is not None and (instance.is_available() if hasattr(instance, "is_available") else True)
    except Exception:
        return False
