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


class NamedProviderRefused(RuntimeError):
    """A named provider that exists but may not load in this process (the reason is the message)."""


def bundled_dir_is_genuine(plugin_dir) -> bool:
    """Whether a manifest that claims the ``bundled`` source really is a directory inside the
    plugins tree shipped with Hermes: not a symlink, and its realpath stays inside the bundled
    root (the tree resolved from Hermes's own installed code — never from the deployment home,
    config or environment; a hosted process ignores ``HERMES_BUNDLED_PLUGINS``). A symlink planted
    in the bundled dir (or a dir whose realpath leaves the tree) is operator-unreviewed code
    wearing the bundled label, so every trust decision that consults the bundled source must
    consult this, not the label alone."""
    from hermes_cli.plugins import get_bundled_plugins_dir

    if not plugin_dir:
        return False
    path = Path(plugin_dir)
    try:
        if path.is_symlink():
            return False
        resolved = path.resolve()
        root = get_bundled_plugins_dir().resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def _hosted_allowed_roots() -> tuple[Path, ...]:
    """The realpath roots whose code a deployment image ships — the only roots a hosted process
    imports in-process code from: the plugins tree bundled inside the installed Hermes package
    (resolved from Hermes's own code, never from the deployment home, config or environment)
    and this interpreter's own site-packages directories, where the image's installed packages
    live."""
    import sysconfig

    from hermes_cli.plugins import get_bundled_plugins_dir

    roots: list[Path] = [Path(get_bundled_plugins_dir())]
    paths = sysconfig.get_paths()
    roots.extend(Path(paths[key]) for key in ("purelib", "platlib") if paths.get(key))
    resolved: list[Path] = []
    for root in roots:
        try:
            real = root.resolve()
        except OSError:
            continue
        if real not in resolved:
            resolved.append(real)
    return tuple(resolved)


def _realpath_inside(path, roots: tuple[Path, ...]) -> bool:
    """Whether *path*'s realpath stays inside one of *roots* (equal to it, or below it)."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def hosted_inprocess_refusal(code_path, *, module: Optional[str] = None) -> str:
    """THE hosted provenance rule — the one function every loader that could import Python in
    a hosted process consults before importing: ``""`` when the code may be imported
    in-process, else the reason it may not. The check runs before any import, so refused code
    never executes.

    One realpath rule decides every loader kind alike. A directory loader passes the directory
    in *code_path*; an installed distribution's entry point passes ``code_path=None`` with
    *module* naming the module its entry point targets. Either way the code is eligible only
    when the realpath of what would be imported — the directory, or the file
    ``importlib.util.find_spec`` resolves *module* to WITHOUT importing it — stays inside an
    allowed root (:func:`_hosted_allowed_roots`): the plugins tree bundled inside the installed
    Hermes package, or this interpreter's own site-packages, where the image's packages are
    installed.

    Everything else is refused before import, whatever the distribution's metadata says — an
    editable install pointing elsewhere, a ``.pth``-added path, a directory under the
    deployment home (its records can never vouch for hosted trust: the home is
    tenant-writable, and a catalog plugin a hosted deployment wants is baked into its image by
    the operator)."""
    from hermes_cli.hosted_mode import is_hosted

    if not is_hosted():
        return ""
    roots = _hosted_allowed_roots()
    if code_path is None:
        # An entry point's eligibility is where its module's file REALLY is, found without
        # importing it — never the distribution's metadata, which a ``.pth`` file or an
        # editable record can aim anywhere.
        origin: Optional[str] = None
        if module:
            try:
                spec = importlib.util.find_spec(module)
            except (ImportError, ValueError):  # unresolvable or ambiguous: no file to judge
                spec = None
            origin = getattr(spec, "origin", None) if spec is not None else None
        if origin and _realpath_inside(origin, roots):
            return ""
        return ("a hosted process loads in-process code only from what its deployment image "
                "ships (the plugins bundled with Hermes, or a package installed into the "
                "image's environment); the entry point's module "
                + (f"{module!r} " if module else "")
                + "resolves outside that image, so its distribution's metadata is not "
                "provenance and the entry point is refused")
    if _realpath_inside(code_path, roots):
        return ""
    return ("a hosted process loads in-process code only from what its deployment image ships "
            "(the plugins bundled with Hermes, or a package installed into the image's "
            f"environment); {code_path} is in the deployment home's writable directories, and an "
            "install record is never consulted for hosted trust")


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
    """Shared body of ``load_<kind>(name)``: load from *plugin_dir*, warn + None on failure.

    A provider outside the deployment image's bundled tree is refused in a hosted process
    BEFORE its module is imported — :func:`hosted_inprocess_refusal` is the one rule, and
    :exc:`NamedProviderRefused` names its reason."""
    refusal = hosted_inprocess_refusal(plugin_dir)
    if refusal:
        logger.error("%s %s: %s", kind, name, refusal)
        raise NamedProviderRefused(refusal)
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
