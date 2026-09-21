"""Bootstrap the Python environment link for newly-created Git worktrees.

Worktree checkouts do not carry ignored directories from their source checkout.
Hermes therefore links the source repository's canonical ``.venv`` into a new
worktree instead of copying or rebuilding it.  The link is only created when
the source environment is provably part of the same Git repository, including
the common case where a linked source worktree already points at the main
checkout's environment.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path


logger = logging.getLogger(__name__)

_GIT_TIMEOUT = 30


def _git_value(path: Path, args: list[str]) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve(strict=False)


def _same_repository_environment(
    source_root: Path,
    resolved_source: Path,
    *,
    governed_roots: tuple[Path, ...] = (),
) -> bool:
    """Return whether an environment resolves inside this repo's checkout."""
    try:
        if resolved_source.is_relative_to(source_root):
            return True
    except ValueError:
        pass
    if any(
        resolved_source == root or resolved_source.is_relative_to(root)
        for root in governed_roots
    ):
        return True

    # A linked worktree may contain ``.venv -> <main-checkout>/.venv``.  The
    # symlink leaves the source worktree, but remains safe when Git proves that
    # both paths use the same common repository directory.
    common_dir = _git_value(source_root, ["rev-parse", "--path-format=absolute", "--git-common-dir"])
    if common_dir is None:
        return False
    main_root = common_dir.parent
    return (
        _git_value(resolved_source, ["rev-parse", "--show-toplevel"]) == main_root
        and _git_value(main_root, ["rev-parse", "--path-format=absolute", "--git-common-dir"])
        == common_dir
    )


def _venv_python_path(resolved_source: Path, *, _platform: str | None = None) -> Path:
    """The interpreter entrypoint for a virtualenv rooted at *resolved_source*.

    Native Windows virtualenvs use ``Scripts/python.exe`` rather than
    ``bin/python``; POSIX (incl. WSL) always uses the ``bin/`` layout.

    Checks ``sys.platform`` (matching ``venv_bin_dir()``'s own default), not ``os.name``: Python
    3.13's ``Path.__new__`` dispatches its concrete class from ``os.name`` at call time, so a test
    monkeypatching ``os.name`` to simulate Windows would make ``venv_bin_dir()``'s internal
    ``Path(venv_dir)`` reconstruction try to build a ``WindowsPath`` and crash on a real POSIX host.

    *_platform* pins the platform for tests; omit in production (defaults to ``sys.platform``).
    """
    from hermes_constants import venv_bin_dir
    platform = _platform if _platform is not None else sys.platform
    windows = platform == "win32"
    python_name = "python.exe" if windows else "python"
    return venv_bin_dir(resolved_source, windows=windows) / python_name


def bootstrap_worktree_environments(
    repo_root: Path,
    target: Path,
    *,
    environment_names: tuple[str, ...] = (".venv", "venv"),
    require_python: bool = True,
    allow_venv_fallback: bool = True,
) -> tuple[str, ...]:
    """Link usable source environments into a new worktree when absent.

    The destination is never replaced, including a broken symlink.  For the
    required ``.venv`` destination, ``venv`` can be accepted as a source
    fallback so Hermes repositories that use the legacy name still expose the
    stable ``./.venv/bin/python`` entrypoint.  Callers that preserve a legacy
    directory-only behavior can disable both checks explicitly. The returned
    tuple contains the names that were linked.
    """
    try:
        source_root = repo_root.expanduser().resolve(strict=True)
        target_root = target.expanduser().resolve(strict=True)
    except OSError as exc:
        logger.warning("worktree environment bootstrap roots unavailable: %s", exc)
        return ()

    # Existing linked worktrees commonly have no ignored environment of their
    # own. Resolve the main checkout from Git's common directory so reuse of a
    # worktree gets the same repository-scoped environment as a newly-created
    # worktree. The repository check below still rejects environments from a
    # different Git repository.
    source_roots = [source_root]
    governed_roots: list[Path] = []
    common_dir = _git_value(
        source_root, ["rev-parse", "--path-format=absolute", "--git-common-dir"]
    )
    if common_dir is not None and common_dir.name == ".git":
        main_root = common_dir.parent
        if main_root not in source_roots:
            source_roots.append(main_root)
        # Managed Hermes installs keep the repository venv outside the
        # checkout at ``<repo-parent>/venvs``. It is still repository-scoped:
        # this is the same explicit sibling layout used by the runtime and is
        # narrower than trusting arbitrary absolute symlink targets.
        governed_roots.append(main_root.parent / "venvs")

    linked: list[str] = []
    for environment_name in environment_names:
        destination = target_root / environment_name
        if destination.exists() or destination.is_symlink():
            continue

        source_names = (
            (environment_name, "venv")
            if environment_name == ".venv" and allow_venv_fallback
            else (environment_name,)
        )
        for candidate_root in source_roots:
            for source_name in source_names:
                source = candidate_root / source_name
                try:
                    if not source.exists():
                        continue
                    resolved_source = source.resolve(strict=True)
                    python = _venv_python_path(resolved_source)
                    if not resolved_source.is_dir() or not _same_repository_environment(
                        source_root,
                        resolved_source,
                        governed_roots=tuple(governed_roots),
                    ):
                        continue
                    if require_python and (
                        not python.is_file() or not os.access(python, os.X_OK)
                    ):
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(
                        str(resolved_source), str(destination), target_is_directory=True
                    )
                    linked.append(environment_name)
                    break
                except OSError as exc:
                    logger.warning(
                        "worktree environment bootstrap could not link %s",
                        environment_name,
                    )
            if linked and linked[-1] == environment_name:
                break
    return tuple(linked)
