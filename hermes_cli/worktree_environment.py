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


def _same_repository_environment(source_root: Path, resolved_source: Path) -> bool:
    """Return whether an environment resolves inside this repo's checkout."""
    try:
        if resolved_source.is_relative_to(source_root):
            return True
    except ValueError:
        pass

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
        for source_name in source_names:
            source = source_root / source_name
            try:
                if not source.exists():
                    continue
                resolved_source = source.resolve(strict=True)
                python = resolved_source / "bin" / "python"
                if not resolved_source.is_dir() or not _same_repository_environment(
                    source_root, resolved_source
                ):
                    continue
                if require_python and (
                    not python.is_file() or not os.access(python, os.X_OK)
                ):
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(str(resolved_source), str(destination), target_is_directory=True)
                linked.append(environment_name)
                break
            except OSError as exc:
                logger.warning(
                    "worktree environment bootstrap could not link %s: %s",
                    environment_name,
                    exc,
                )
    return tuple(linked)
