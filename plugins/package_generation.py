"""Frozen Python packages for external memory providers and their OAuth companions.

A generation is the captured source bytes of one package directory; its relative imports,
including delayed ones, resolve against those bytes and never against the mutable install.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
from importlib import _bootstrap
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class PackageSnapshot:
    directory: Path
    digest: str
    sources: Mapping[str, bytes]


def capture_package(directory: Path) -> PackageSnapshot:
    directory = directory.resolve()
    sources = {}
    for parent, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in {
            '__pycache__', 'venv', 'env', 'node_modules',
        })
        for filename in sorted(files):
            if filename.endswith('.py'):
                path = Path(parent) / filename
                sources[path.relative_to(directory).as_posix()] = path.read_bytes()
    digest = hashlib.sha256(str(directory).encode())
    for filename, source in sorted(sources.items()):
        digest.update(filename.encode() + b'\0' + hashlib.sha256(source).digest())
    return PackageSnapshot(directory, digest.hexdigest(), MappingProxyType(sources))


class _FrozenSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, source, *, snapshot=None, exact_files=frozenset()):
        super().__init__(name, str(path))
        self.source = source
        self.snapshot = snapshot
        self.exact_files = exact_files

    def get_code(self, fullname):
        return self.source_to_code(self.source, self.path)

    def exec_module(self, module):
        if self.snapshot is not None:
            module._hermes_package_snapshot = self.snapshot
            module._hermes_exact_files = self.exact_files
        super().exec_module(module)


class _FrozenPackageFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(("_hermes_user_memory.", "_hermes_memory_companions_")):
            return None
        ancestor = fullname
        while '.' in ancestor:
            ancestor = ancestor.rpartition('.')[0]
            package = sys.modules.get(ancestor)
            snapshot = getattr(package, '_hermes_package_snapshot', None)
            if isinstance(snapshot, PackageSnapshot):
                break
        else:
            return None
        relative = fullname[len(ancestor) + 1:]
        stem = relative.replace('.', '/')
        candidates = (stem + '/__init__.py', stem + '.py')
        if relative in package._hermes_exact_files:
            candidates = candidates[::-1]
        filename = next((f for f in candidates if f in snapshot.sources), None)
        if filename is None:
            if any(f.startswith(stem + '/') for f in snapshot.sources):
                spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
                spec.submodule_search_locations = []
                return spec
            # Never fall through to the mutable install for a frozen package.
            raise ModuleNotFoundError(fullname)
        source_path = snapshot.directory / filename
        loader = _FrozenSourceLoader(fullname, source_path, snapshot.sources[filename])
        return importlib.util.spec_from_file_location(fullname, source_path, loader=loader)


# Also covers delayed importlib.import_module, unlike a private __import__ hook.
sys.meta_path.insert(0, _FrozenPackageFinder())


def load_package_generation(name: str, snapshot: PackageSnapshot, *,
                            execute_init: bool, exact_files=frozenset()):
    with _bootstrap._ModuleLockManager(name):
        package = sys.modules.get(name)
        if package is not None:
            return package
        if execute_init:
            path = snapshot.directory / '__init__.py'
            loader = _FrozenSourceLoader(name, path, snapshot.sources['__init__.py'],
                                         snapshot=snapshot, exact_files=exact_files)
            spec = importlib.util.spec_from_file_location(name, path, loader=loader,
                                                         submodule_search_locations=[])
            package = _bootstrap._load_unlocked(spec)
        else:
            spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
            spec.submodule_search_locations = []
            package = importlib.util.module_from_spec(spec)
            package._hermes_package_snapshot = snapshot
            package._hermes_exact_files = exact_files
            sys.modules[name] = package
        parent, _, child = name.rpartition('.')
        if parent in sys.modules:
            setattr(sys.modules[parent], child, package)
        return package
