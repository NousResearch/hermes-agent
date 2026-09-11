"""Build/CI operations with caller-owned inputs, independent of live selection."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
import sys

from pm.package import InstallError


def check_project_lock(
    source: Path, *, python: Path | None = None, cache: Path | None = None,
    env: Mapping[str, str] | None = None, offline: bool = False, explicit: bool = False,
) -> None:
    """Reject a missing or stale lock without rewriting source or creating a venv."""
    from pm.environment import managed_environment

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
    from pm.environment import managed_environment, prune_site_pth
    from pm.operations import _requirements

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
    out.mkdir(parents=True)
    try:
        environment.create()
        environment.install_requirements(requirements, wheelhouse=wheelhouse)
        environment.check()
        if sealed:
            prune_site_pth(out)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return environment.executable


def prune_cache(cache: Path, *, ci: bool = False) -> None:
    """Prune unused cache entries; CI mode also discards downloaded wheels."""
    from pm.environment import managed_environment

    cache = Path(cache).absolute()
    environment = managed_environment(cache, cache=cache, explicit=True, output=sys.stderr)
    environment.prune_cache(ci=ci)
