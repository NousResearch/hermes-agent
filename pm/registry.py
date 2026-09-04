"""Package registry and dependency walk."""

from __future__ import annotations

from pm.package import Package

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
