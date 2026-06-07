r"""Map a host skill directory to its backend-visible path.

When a non-local terminal backend is active (Docker / Singularity / Modal /
SSH / Daytona), the skills tree the agent sees lives at a *different*
filesystem location than the host ``~/.hermes/skills`` path.  The mount/sync
layout in :mod:`tools.credential_files` already computes those backend-visible
locations (``get_skills_directory_mount`` mounts the local skills dir at
``<container_base>/skills`` and external dirs at
``<container_base>/external_skills/<index>``).

The prompt renderer (``agent/skill_preprocessing.py`` and
``agent/skill_commands.py``) historically emitted the raw HOST path into the
agent prompt for ``${HERMES_SKILL_DIR}``, the ``[Skill directory: ...]`` header,
and supporting-file/script hints.  On a remote backend the agent then saw a
host path (``L:\...`` / ``/Users/...``) it cannot reach, so bundled skill
scripts (e.g. ``${HERMES_SKILL_DIR}/scripts/todo``) were unrunnable inside the
sandbox.

:func:`map_skill_dir_for_backend` consults the *same* mount layout as the
single source of truth and returns the backend-visible path.  It mirrors the
backend-resolution precedence used by
``tools/image_generation_tool._agent_cache_base_for_env`` for cache files:
prefer the active environment instance's ``_remote_home`` (SSH/Daytona), then
the well-known ``/root/.hermes`` container base (Docker/Singularity/Modal),
falling back to the ``TERMINAL_ENV`` name when no environment has been created
yet.  Local backend (and any unresolved/unmapped path) keeps the host path.
"""

from __future__ import annotations

import logging
import os
import posixpath
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Container backends with a deterministic Hermes root inside the sandbox.
_CONTAINER_BACKENDS = {"docker", "singularity", "modal"}
_CONTAINER_ENV_CLASSES = {
    "DockerEnvironment",
    "SingularityEnvironment",
    "ModalEnvironment",
}


def _active_terminal_env(task_id: str | None) -> Any:
    """Return the live terminal environment for *task_id*, or ``None``."""
    try:
        from tools.terminal_tool import get_active_env

        return get_active_env(task_id or "default")
    except Exception as exc:  # noqa: BLE001 - path hinting must not break skills
        logger.debug("Could not inspect active terminal environment: %s", exc)
        return None


def _hermes_base_for_env(env: Any) -> str | None:
    """Resolve the backend-visible ``.hermes`` root for the active backend.

    Mirrors ``image_generation_tool._agent_cache_base_for_env``: the live
    environment instance is the source of truth when present (SSH/Daytona
    expose ``_remote_home``), otherwise the ``TERMINAL_ENV`` name selects a
    deterministic container root.  Returns ``None`` for the local backend or
    when the backend cannot be resolved without side effects.
    """
    if env is not None:
        remote_home = getattr(env, "_remote_home", None)
        if remote_home:
            return f"{str(remote_home).rstrip('/')}/.hermes"

        if env.__class__.__name__ in _CONTAINER_ENV_CLASSES:
            return "/root/.hermes"

    backend = (os.getenv("TERMINAL_ENV") or "local").strip().lower()
    if backend in _CONTAINER_BACKENDS:
        return "/root/.hermes"
    if backend == "daytona":
        # Daytona's remote home is only known from a live environment instance
        # (resolved at sandbox creation); without one we cannot translate.
        return None
    if backend == "ssh":
        # SSH can use a shell-visible tilde path before the first environment is
        # created; the first sync uploads the skills tree before any command runs.
        return "~/.hermes"
    return None


def map_skill_dir_for_backend(
    host_skill_dir: Path | str,
    task_id: str | None = None,
) -> str:
    """Translate *host_skill_dir* to the path the agent sees on the active backend.

    Longest-prefix-matches the host skill dir against the existing mount layout
    (:func:`tools.credential_files.get_skills_directory_mount`) and returns the
    corresponding backend-visible path.  Local backend, an unresolved backend,
    or a directory not under any known skills mount all fall back to the host
    path string (no regression for the common case).
    """
    host_str = str(host_skill_dir)

    env = _active_terminal_env(task_id)
    container_base = _hermes_base_for_env(env)
    if not container_base:
        return host_str

    try:
        from tools.credential_files import get_skills_directory_mount

        mounts = get_skills_directory_mount(container_base=container_base)
    except Exception as exc:  # noqa: BLE001 - never break skill rendering
        logger.debug("Could not load skills mount layout for backend: %s", exc)
        return host_str

    host_path = Path(host_str)
    # Longest-prefix match so a more specific mount (external_skills/<idx>) wins
    # over a parent that also matches.
    best: tuple[int, str] | None = None
    for mount in mounts:
        mount_host = Path(mount["host_path"])
        try:
            rel = host_path.relative_to(mount_host)
        except ValueError:
            continue
        depth = len(mount_host.parts)
        rel_posix = rel.as_posix()
        # ``rel`` is "." when the skill dir IS the mount root — emit the mount
        # container path directly rather than appending a spurious "/.".
        if rel_posix == ".":
            mapped = mount["container_path"]
        else:
            mapped = posixpath.join(mount["container_path"], rel_posix)
        if best is None or depth > best[0]:
            best = (depth, mapped)

    if best is not None:
        return best[1]
    return host_str
