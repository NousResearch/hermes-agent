"""Build/CI operations with caller-owned inputs, independent of live selection."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from types import MappingProxyType

from pm.ensure import InstalledPackage
from pm.lock import Lockfile
from pm.package import InstallError


def check_project_lock(
    source: Path, *, python: Path | None = None, cache: Path | None = None,
    env: Mapping[str, str] | None = None, offline: bool = False, explicit: bool = False,
) -> None:
    """Reject a missing or stale lock without rewriting source or creating a venv."""
    from pm.environment import managed_environment
    from pm.operations import _require_install_allowed

    if not offline:
        _require_install_allowed(explicit)
    source = Path(source).absolute()
    if not (source / "pyproject.toml").is_file():
        raise InstallError("venv", f"project manifest is missing: {source}")
    environment = managed_environment(
        source / ".venv", python=Path(python) if python is not None else None,
        cache=Path(cache) if cache is not None else None, env=env,
        offline=offline, explicit=explicit, output=sys.stderr,
    )
    environment.check_lock(source)


def export_requirements(
    source: Path, out: Path, *, extras: Sequence[str] = (), python: Path | None = None,
    cache: Path | None = None, env: Mapping[str, str] | None = None, explicit: bool = False,
) -> None:
    """Export locked runtime requirements, preserving markers and direct URL pins."""
    from pm.environment import managed_environment
    from pm.operations import _require_install_allowed

    _require_install_allowed(explicit)
    source, out = Path(source).absolute(), Path(out).absolute()
    if not (source / "pyproject.toml").is_file():
        raise InstallError("venv", f"project manifest is missing: {source}")
    if not (source / "uv.lock").is_file():
        raise InstallError("venv", f"frozen export requires a lock: {source / 'uv.lock'}")
    environment = managed_environment(
        source / ".venv", python=Path(python) if python is not None else None,
        cache=Path(cache) if cache is not None else None, env=env,
        explicit=explicit, output=sys.stderr,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    environment.export_requirements(source, out, extras=extras)


def build_requirements_environment(
    requirements: Sequence[str], *, out: Path, python: Path | None = None,
    cache: Path | None = None, env: Mapping[str, str] | None = None,
    wheelhouse: Path | None = None, offline: bool = False, sealed: bool = False,
    explicit: bool = False,
) -> Path:
    """Create and check a fresh environment. Never mutate an existing destination.

    Wheelhouse builds disable indexes and source builds. Failure removes only
    this invocation's exclusively claimed output, not prior builds.
    """
    from pm.environment import _fresh_build, managed_environment
    from pm.operations import _require_install_allowed, _requirements

    _require_install_allowed(explicit)
    requirements = _requirements(requirements) if requirements or isinstance(requirements, str) else []
    out = Path(out).absolute()
    wheelhouse = Path(wheelhouse).absolute() if wheelhouse is not None else None
    if wheelhouse is not None and not wheelhouse.is_dir():
        raise InstallError("venv", f"wheelhouse directory is missing: {wheelhouse}")
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"environment destination already exists: {out}")
    environment = managed_environment(
        out, python=Path(python) if python is not None else None,
        cache=Path(cache) if cache is not None else None, env=env,
        offline=offline, explicit=explicit, output=sys.stderr,
    )
    with _fresh_build(environment, sealed=sealed):
        environment.install_requirements(requirements, wheelhouse=wheelhouse)
    return environment.executable


def prepare_tools(names: Sequence[str], *, out: Path, target: str,
                  cache: Path | None = None) -> Path:
    """Realize an explicit tool closure in a build-owned store, not live state."""
    from pm.ensure import _install, _lockfile
    from pm.lock import Facts
    from pm.package import StatePackage
    from pm.registry import walk
    from pm.store import Store

    if isinstance(names, str):
        raise TypeError("names must be a sequence, not a string")
    packages = walk(list(names))
    if any(isinstance(package, StatePackage) for package in packages):
        raise ValueError("prepare_tools accepts tools, not application state")
    out = Path(out).resolve()
    store, lock = Store(out), _lockfile()
    with store.install_lock():
        facts = Facts(out / "facts.json", strict=True)
        for package in packages:
            _install(package, lock, facts, store, target, _lock_held=True)
    return out


def _copy_links(package, entry: Path) -> None:
    import os
    from pm.filesystem import is_junction

    for directory, dirs, files in os.walk(entry):
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink() or is_junction(path):
                if (Path(os.readlink(path)).is_absolute() or not path.exists()
                        or not path.resolve().is_relative_to(entry)):
                    raise InstallError(package.name, f"tool copy link escapes entry: {path}")


@dataclass(frozen=True)
class VerifiedTools:
    entries: Mapping[str, InstalledPackage]
    _envs: tuple[dict, ...]

    def environment(self, base: Mapping[str, str]) -> dict[str, str]:
        from pm.package import compose_env

        return compose_env(list(self._envs), base=dict(base))


def verified_tools(names: Sequence[str], *, source_store: Path, target: str,
                   lock: Lockfile | None = None) -> VerifiedTools:
    """Read a pinned tool closure without writes, acquisition or binary execution.

    Publication already ran package verification. Admission binds its recorded
    digest to the canonical entry, lock and declared environment; it does not
    rerun arbitrary version probes just to read a previously published tool.
    """
    from pm.ensure import _identity, _lockfile
    from pm.lock import Facts
    from pm.package import StatePackage
    from pm.registry import walk
    from pm.store import tree_digest

    if isinstance(names, str):
        raise TypeError("names must be a sequence, not a string")
    packages = walk(list(names))
    if any(isinstance(package, StatePackage) for package in packages):
        raise ValueError("verified_tools accepts tool packages, not application state")
    source_store = Path(source_store).resolve()
    if not (source_store / "facts.json").is_file():
        raise InstallError("tools", f"source facts missing: {source_store}")
    facts = Facts(source_store / "facts.json", strict=True)
    lock = lock if lock is not None else _lockfile()
    entries, envs = {}, []
    for package in packages:
        if getattr(package, "pin_only", False):
            continue
        version = lock.version(package.name)
        identity = _identity(lock, package.name, target)
        if not version or identity is None:
            raise InstallError(package.name, "tool source failed verification: missing pin")
        fact = facts.get(package.name)
        expected = package.store_entry(version, target)
        entry = source_store / expected
        if (not fact or fact.get("entry") != expected
                or not entry.is_dir() or not entry.resolve().is_relative_to(source_store)
                or not facts.installed(package.name, version, source_store, identity)
                or tree_digest(entry) != fact.get("digest")):
            raise InstallError(package.name, "tool source failed verification", "run preparation again")
        _copy_links(package, entry)
        binary = package.binary(entry, target)
        env = package.env(entry, target)
        if ((binary is not None and (not binary.is_file() or not binary.resolve().is_relative_to(entry)))
                or facts.env_for(package.name, source_store) != env):
            raise InstallError(package.name, "tool source failed verification: binary or environment")
        entries[package.name] = InstalledPackage(entry, version, binary)
        envs.append(env)
    return VerifiedTools(MappingProxyType(entries), tuple(envs))


def stage_tools(names: Sequence[str], *, source_store: Path, out: Path, target: str) -> Path:
    """Copy a current, verified tool closure without acquisition or live selection.

    Stores must be disjoint. Both publication locks cover admission and copying;
    the ordinary installer owns verification, independent copies and fresh facts.
    """
    from contextlib import ExitStack
    from pm.ensure import _entry_verified, _install, _lockfile
    from pm.lock import Facts
    from pm.package import StatePackage
    from pm.registry import walk
    from pm.store import Store

    if isinstance(names, str):
        raise TypeError("names must be a sequence, not a string")
    source_store, out = Path(source_store).resolve(), Path(out).resolve()
    if source_store.is_relative_to(out) or out.is_relative_to(source_store):
        raise ValueError("tool stores must be disjoint")
    if not (source_store / "facts.json").is_file():
        raise InstallError("tools", f"source facts missing: {source_store}")
    packages = walk(list(names))
    if any(isinstance(package, StatePackage) for package in packages):
        raise ValueError("stage_tools accepts tool packages, not application state")
    source, destination = Store(source_store), Store(out)
    lock = _lockfile()
    with ExitStack() as stack:
        for store in sorted((source, destination), key=lambda store: str(store.root)):
            stack.enter_context(store.install_lock())
        selection = verified_tools(names, source_store=source_store, target=target, lock=lock)
        facts = Facts(source_store / "facts.json", strict=True)
        for package in packages:
            if package.name not in selection.entries:
                continue
            fact = facts.get(package.name)
            assert fact is not None  # Admission requires the selected fact under the same store lock.
            if not _entry_verified(package, fact, source, target):
                raise InstallError(package.name, "tool copy source failed verification", "run preparation again")
        staged = Facts(out / "facts.json", strict=True)
        for package in packages:
            _install(package, lock, staged, destination, target,
                     copy_from=(facts, source), _lock_held=True, _fresh_copy=True)
    return out


def prune_cache(cache: Path, *, ci: bool = False) -> None:
    """Prune unused cache entries; CI mode also discards downloaded wheels."""
    from pm.environment import managed_environment

    cache = Path(cache).absolute()
    environment = managed_environment(cache, cache=cache, realize=False, output=sys.stderr)
    environment.prune_cache(ci=ci)
