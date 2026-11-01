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

    def collect(self, register: Callable[[Any], Any], *, source: Optional[str] = None) -> None:
        register(self)


class ForwardingPluginContext:
    """Plugin context for a category loader (memory provider, context engine): a subclass captures
    the one ``register_*`` call its activation path owns and every other ``register_*`` call reaches
    a real ``PluginContext``, so these plugins have the full plugin surface. With ``forward=False``
    (listing/inspection) the other calls are no-ops and nothing touches the plugin manager."""

    label = "Plugin"

    def __init__(self, name: str, *, forward: bool = True):
        self.name = name
        self._forward = forward
        self._context = None
        self._hook_source = None

    def collect(self, register: Callable[[Any], Any], *, source: Optional[str] = None) -> None:
        """Run ``register`` with this collector; hooks it registers form the fallback group that
        general discovery of the same source replaces (see ``PluginLedgerMixin``)."""
        if not self._forward:
            register(self)
            return
        from types import SimpleNamespace
        from hermes_cli.plugins_ledger import _hook_source_of

        module = sys.modules.get(getattr(register, "__module__", ""))
        self._hook_source = _hook_source_of(self.name, SimpleNamespace(__file__=source) if source else module)
        manager = self._plugin_context()._manager
        with manager._discovery_lock:
            manager._drop_fallback_hooks(self._hook_source)
            register(self)

    def collect_in_host(self, host: Any, plugin_dir: Path, module_name: str, *, capture: str,
                        base_ref: str) -> Any:
        """``collect`` for a plugin whose code runs in the plugin host: the host captures the
        instance and forwards every other registration here. The discovery lock is NOT held across
        the host call — its registrations arrive on another thread and take the lock themselves."""
        from types import SimpleNamespace
        from hermes_cli.plugins_ledger import _hook_source_of

        self._hook_source = _hook_source_of(self.name, SimpleNamespace(__file__=str(plugin_dir / "__init__.py")))
        manager = self._plugin_context()._manager

        def drop_hooks() -> None:  # also before a reload into a restarted host: no stale hook proxies
            with manager._discovery_lock:
                manager._drop_fallback_hooks(self._hook_source)

        drop_hooks()
        return host.load_instance(plugin_dir, module_name=module_name, capture=capture, base_ref=base_ref,
                                  ctx=self, before_reload=drop_hooks)

    def register_hook(self, hook_name, callback, *, hold_stream=False):
        if not self._forward:
            return None
        context = self._plugin_context()
        with context._manager._discovery_lock:
            return context._manager._register_fallback_hook(
                context, self._hook_source, hook_name, callback, hold_stream=hold_stream)

    def register_cli_command(self, *args, **kwargs):
        pass  # CLI registration happens via each category's own CLI discovery

    def __getattr__(self, attr: str):
        """Delegate any other ``register_*`` call to a real ``PluginContext`` (a
        hand-maintained stub drifted: unknown calls raised and cost the plugin).
        Non-``register_*`` attributes raise normally so a typo still fails loudly."""
        if not attr.startswith("register_"):
            raise AttributeError(attr)

        def _forward(*args, **kwargs):
            if not self._forward:
                return None
            try:
                return self._plugin_context().__getattribute__(attr)(*args, **kwargs)
            except Exception as exc:  # health: allow BLE001 -- moved from plugins/memory; a plugin bug, logged by name
                # A secondary registration must not cost the plugin itself.
                logging.getLogger(type(self).__module__).warning(
                    "%s '%s' failed to %s: %s", self.label, self.name, attr, exc)
                return None

        return _forward

    def _plugin_context(self):
        """A real ``PluginContext``, built once on demand: the common plugin that only calls its
        capture method must not pay for importing the plugin manager."""
        if self._context is None:
            from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager

            manifest = PluginManifest(name=self.name, key=self.name)
            self._context = PluginContext(manifest, get_plugin_manager())
        return self._context


def instance_from_module(mod: Any, *, collector: Any, collected_attr: str, base_cls: type, name: str,
                         logger: logging.Logger, source: Optional[str] = None) -> Optional[Any]:
    """Extract the provider instance: ``register(ctx)`` first, then any ``base_cls`` subclass.
    ``source`` keys the collector's hook group (default: the module's ``__file__``)."""
    if hasattr(mod, "register"):
        try:
            collector.collect(mod.register, source=source or getattr(mod, "__file__", None))
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
