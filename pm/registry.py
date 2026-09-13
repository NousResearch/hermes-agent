"""Package registry and dependency walk."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

from pm.package import InstallError, Package

_packages: dict[str, Package] = {}


def register(cls):
    instance = cls()
    if not instance.name:
        raise ValueError("package has no name")
    _packages[instance.name] = instance
    return cls


def get_package(name: str) -> Package:
    if name not in _packages:
        raise KeyError(f"unknown package: {name}")
    return _packages[name]


def all_packages() -> list[str]:
    return sorted(_packages)


def package_definitions(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Declarations for a fresh worker; built-ins already load with pm.

    Only import identities and file locations cross the wire, never instances,
    source bodies, or the application's import path. ``names`` selects a closure.
    """
    definitions = []
    for package in walk(names) if names is not None else list(_packages.values()):
        cls = type(package)
        if cls.__module__ == "pm.packages":
            continue
        definition = {"name": package.name, "module": cls.__module__, "qualname": cls.__qualname__}
        module = sys.modules.get(cls.__module__)
        resolved = module
        for part in cls.__qualname__.split("."):
            resolved = getattr(resolved, part, None)
        if cls.__module__ == "__main__" or "<locals>" in cls.__qualname__ or resolved is not cls:
            raise InstallError(
                package.name, f"package definition {cls.__module__}.{cls.__qualname__} is not importable",
                "define and register a module-level Package subclass in an importable file",
            )
        source = getattr(module, "__file__", None)
        if source:
            definition["path"] = str(Path(source).resolve())
        namespaces = {}
        parent = cls.__module__.rpartition(".")[0]
        while parent:
            loaded = sys.modules.get(parent)
            if loaded is not None and not getattr(loaded, "__file__", None) and hasattr(loaded, "__path__"):
                namespaces[parent] = [str(Path(path).resolve()) for path in loaded.__path__]
            parent = parent.rpartition(".")[0]
        if namespaces:
            definition["namespaces"] = namespaces
        definitions.append(definition)
    return definitions


def load_package_definitions(definitions: list[dict[str, Any]]) -> None:
    """Restore explicit registrations after built-ins, without plugin discovery."""
    restored = {}
    for definition in definitions:
        module_name = definition["module"]
        try:
            # Directory plugins use synthetic package parents. Their explicit
            # search paths preserve relative imports without exposing app deps.
            for name, locations in definition.get("namespaces", {}).items():
                if name not in sys.modules:
                    namespace = ModuleType(name)
                    namespace.__path__ = locations
                    sys.modules[name] = namespace
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                # Missing dependencies inside the definition are not evidence
                # that its module is absent; never execute such a module twice.
                if not exc.name or not (module_name == exc.name or module_name.startswith(exc.name + ".")):
                    raise
                spec = importlib.util.spec_from_file_location(module_name, definition["path"])
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot load {module_name}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                except BaseException:
                    sys.modules.pop(module_name, None)
                    raise
            source = definition.get("path")
            if source and Path(getattr(module, "__file__", "") or "").resolve() != Path(source).resolve():
                raise ImportError(f"{module_name} resolved to a different source")
            cls = module
            for part in definition["qualname"].split("."):
                cls = getattr(cls, part)
            if not isinstance(cls, type) or not issubclass(cls, Package):
                raise TypeError("definition is not a Package subclass")
            instance = cls()
            if instance.name != definition["name"]:
                raise ValueError(f"definition now registers {instance.name!r}")
            restored[instance.name] = instance
        except Exception as exc:
            raise InstallError(
                definition["name"], f"cannot load package definition {module_name}.{definition['qualname']}: {exc}",
                "keep the package definition and its dependencies importable in the isolated PM runtime",
            ) from exc
    # Import-time decorators must not override the caller's final selection.
    _packages.update(restored)


def walk(names: list[str]) -> list[Package]:
    """Deps-first topological order over the requested packages."""
    seen: dict[str, Package] = {}

    def visit(name: str, chain: tuple[str, ...]) -> None:
        if name in chain:
            cycle = " -> ".join(chain + (name,))
            raise ValueError(f"dependency cycle: {cycle}")
        if name in seen:
            return
        package = get_package(name)
        for dep in package.deps:
            visit(dep, chain + (name,))
        seen[name] = package

    for name in names:
        visit(name, ())
    return list(seen.values())
