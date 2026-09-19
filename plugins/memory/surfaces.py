"""Lazy, core-owned companion surfaces for installed memory providers.

Callers bind the owning home/secret/terminal scope before resolving or importing.
Companions are trusted plugin code, not a sandbox: their own imports may have
side effects. This loader never activates the provider or imports other siblings.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
from importlib import _bootstrap
from pathlib import Path
import re
import sys
from types import ModuleType

from plugins import memory


_COMPANIONS = frozenset({"clone", "settings", "oauth_flow"})


def is_provider_name(name: object) -> bool:
    """Accept safe ASCII directory/entry-point segments, not Python syntax names."""
    return isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", name) is not None


class ProviderCompanionLoadError(ImportError):
    """A selected companion could not be resolved or imported; not absence."""


class _CompanionSourceLoader(importlib.machinery.SourceFileLoader):
    """Compile the fingerprinted bytes, never a timestamp-validated .pyc."""

    def __init__(self, name: str, path: Path, source: bytes):
        super().__init__(name, str(path))
        self.source = source

    def get_code(self, fullname):
        return self.source_to_code(self.source, self.path)


class _CompanionFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root, _, relative = fullname.partition(".")
        if not root.startswith("_hermes_memory_companions_") or not relative:
            return None
        package = sys.modules.get(root)
        sources = getattr(package, "_companion_sources", None)
        if sources is None:
            return None
        stem = relative.replace(".", "/")
        candidates = (stem + "/__init__.py", stem + ".py")
        # Root companions are exact files even when a sibling imports them first.
        # All other helpers retain Python's normal package-before-module order.
        if relative in _COMPANIONS:
            candidates = (stem + ".py", stem + "/__init__.py")
        filename = next((name for name in candidates if name in sources), None)
        if filename is None:
            if any(name.startswith(stem + "/") for name in sources):
                spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
                spec.submodule_search_locations = []
                return spec
            # Never fall through to the mutable on-disk tree for this generation.
            raise ModuleNotFoundError(fullname)
        source_path = package._companion_directory / filename
        loader = _CompanionSourceLoader(fullname, source_path, sources[filename])
        return importlib.util.spec_from_file_location(fullname, source_path, loader=loader)


# Delayed importlib.import_module() calls do not pass through a module's private
# __import__. Keep this finder available for those calls, but claim only our
# synthetic roots carrying a frozen source map; ordinary/runtime imports pass on.
sys.meta_path.insert(0, _CompanionFinder())


def _companion_sources(provider_dir: Path) -> tuple[str, dict[str, bytes]]:
    # Revisions alone miss manual edits and helper-only repairs. Read Python
    # siblings without importing them or descending into environments.
    sources = {}
    for directory, dirs, files in os.walk(provider_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in {
            "__pycache__", "venv", "env", "node_modules",
        })
        for filename in sorted(files):
            if filename.endswith(".py"):
                path = Path(directory) / filename
                sources[path.relative_to(provider_dir).as_posix()] = path.read_bytes()
    digest = hashlib.sha256(str(provider_dir.resolve()).encode())
    for filename, source in sorted(sources.items()):
        digest.update(filename.encode() + b"\0" + hashlib.sha256(source).digest())
    return digest.hexdigest(), sources


def _load_companion_file(provider_dir: Path, companion_file: Path) -> ModuleType:
    # Runtime packages are untouched: conversations own those objects. Clone's
    # caller holds the installation lock through snapshot, preparation and publish.
    # These bytes pin even delayed relative imports to their original generation.
    digest, sources = _companion_sources(provider_dir)
    package_name = f"_hermes_memory_companions_{digest}"
    with _bootstrap._ModuleLockManager(package_name):
        package = sys.modules.get(package_name)
        if package is None:
            spec = importlib.machinery.ModuleSpec(package_name, None, is_package=True)
            spec.submodule_search_locations = []
            package = importlib.util.module_from_spec(spec)
            package._companion_sources = sources
            package._companion_directory = provider_dir
            sys.modules[package_name] = package

    module_name = f"{package_name}.{companion_file.stem}"
    # Preserve the importer's lock/initializing protocol for concurrent relatives.
    with _bootstrap._ModuleLockManager(module_name):
        module = sys.modules.get(module_name)
        if module is None:
            loader = _CompanionSourceLoader(module_name, companion_file, sources[companion_file.name])
            spec = importlib.util.spec_from_file_location(module_name, companion_file, loader=loader)
            module = _bootstrap._load_unlocked(spec)
        expected_origin = str(package._companion_directory / companion_file.name)
        if getattr(module.__spec__, "origin", None) != expected_origin:
            raise ImportError("Cached companion does not match its captured source path")
        setattr(package, companion_file.stem, module)
        return module


def load_provider_companion(name: str, companion: str) -> ModuleType | None:
    """Load an optional exact ``<companion>.py`` without provider ``__init__``.

    Resolution shares runtime's bundled-first precedence and project opt-in,
    not its module objects: companion helpers must not rely on runtime globals.
    Like activation, this does not require ``plugins.enabled``. Only a missing
    provider/file or a directory-less module entry point returns ``None``;
    failures are sanitized because dependency exceptions can contain credentials.

    Identity is the resolved source directory plus every Python source's bytes
    (including helpers), not install metadata or timestamps. Existing handles keep
    their Python generation, including delayed relative imports; non-Python assets
    and absolute imports are not snapshotted. Clone hosts must hold
    ``plugin_installation_lock(source_home)`` across code copy, lookup, invocation
    and publication. Non-cooperating manual writes are outside that transaction.
    """
    if not is_provider_name(name):
        raise ValueError("Memory provider name must contain only ASCII letters, digits, underscores, or hyphens.")
    if not isinstance(companion, str) or companion not in _COMPANIONS:
        raise ValueError("Unsupported memory provider companion; use clone, settings, or oauth_flow.")

    try:
        provider_dir = memory.find_provider_dir(name)
        if provider_dir is None:
            entry_point = memory.find_provider_entry_point(name)
            if entry_point is not None:
                from hermes_cli.plugins import resolve_module_origin

                origin = resolve_module_origin(entry_point.value.split(":", 1)[0].strip())
                if not origin or not Path(origin).is_file() or Path(origin).name == "__init__.py":
                    raise ImportError("Provider entry point cannot be resolved")
            return None
        companion_file = provider_dir / f"{companion}.py"
        try:
            companion_file.stat()
        except FileNotFoundError:
            return None
        return _load_companion_file(provider_dir, companion_file)
    except Exception:
        raise ProviderCompanionLoadError(
            f"Could not load memory provider '{name}' companion '{companion}'. "
            "Check the provider installation and its dependencies, then retry."
        ) from None
