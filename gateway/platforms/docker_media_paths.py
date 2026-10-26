"""Host directories that stand in for a Docker container's ``/workspace`` and ``/root``.

``base.py`` is over the file-size ratchet, so the candidate list lives here.
The workspace digest is passed in: delivery runs after the turn's context is
gone, and scanning every ``*_ws_*`` directory would serve another session's file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_WORKSPACE_KEY_RE = re.compile(r"^[0-9a-f]{16}$")


def _scope():
    from gateway.platforms.base import _TRUTHY, _docker_persistent_active, _tenv

    return _tenv, _TRUTHY, _docker_persistent_active


def _docker_sandbox_dir_candidates(session_key: str = "", workspace_key: str = "") -> list[str]:
    """Candidate host sandbox dir names, best first.

    Mirrors ``_resolve_container_task_id``: profile scope (``default``, else
    ``profile:<name>``), then the legacy ``session:<key>`` layout. A session
    that bound a host workspace also has ``<base>:ws:<digest>`` — a different
    directory from the shared profile sandbox. The digest is explicit because
    an ambient lookup after the turn would collapse onto ``default``.
    """
    tenv, _truthy, _persistent = _scope()
    try:
        from tools.environments.path_utils import sanitize_task_id_for_path
    except Exception:  # health: allow BLE001 -- path translation fails closed; a miss is a 404, not a 500
        return ["default"]
    try:
        from hermes_cli.profiles import get_active_profile_name
        profile = get_active_profile_name() or "default"
    except Exception:  # health: allow BLE001 -- path translation fails closed; a miss is a 404, not a 500
        profile = "default"
    candidates: list[str] = []
    # Explicit trusted-profiles opt-in: one shared container identity. That
    # key is not a per-workspace container, so a :ws: digest does not apply.
    if shared := tenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "").strip():
        candidates.append(sanitize_task_id_for_path(f"shared:{shared}"))
    elif _WORKSPACE_KEY_RE.fullmatch(workspace_key or ""):
        base = "default" if profile == "default" else f"profile:{profile}"
        candidates.append(sanitize_task_id_for_path(f"{base}:ws:{workspace_key}"))
    if profile != "default":
        candidates.append(sanitize_task_id_for_path(f"profile:{profile}"))
    candidates.append("default")
    if session_key:  # bug-window legacy layout: per-session sandboxes
        candidates.append(sanitize_task_id_for_path(f"session:{session_key}"))
    return candidates


def _docker_persistent_sandbox_roots(
    session_key: str, leaf: str, workspace_key: str = "",
) -> list[Path]:
    """Existing ``<sandbox>/docker/<candidate>/<leaf>`` dirs, candidate order.

    Empty unless Docker + persistent. The translator tries each until the file
    resolves.
    """
    _tenv, _truthy, persistent = _scope()
    if not persistent():
        return []
    try:
        from tools.environments.base import get_sandbox_dir
        base = get_sandbox_dir() / "docker"
        return [cand for name in _docker_sandbox_dir_candidates(session_key, workspace_key)
                if (cand := (base / name / leaf).resolve(strict=False)).is_dir()]
    except Exception:  # health: allow BLE001 -- path translation fails closed; a miss is a 404, not a 500
        return []


def _mounted_workspace_cwd(workspace_cwd: str = "") -> list[Path]:
    tenv, _truthy, _persistent = _scope()
    cwd = workspace_cwd or tenv("TERMINAL_CWD") or os.getcwd()
    try:
        host = Path(os.path.expanduser(cwd)).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return []
    return [host] if host.is_dir() else []


def _session_mounted_workspace(workspace_key: str, workspace_cwd: str) -> list[Path]:
    """The host dir a session bound to a workspace mounts at ``/workspace``, or nothing.

    A gateway turn runs with the session's own cwd as ``TERMINAL_CWD``
    (``gateway.session_policy.build_policy``), and that cwd is also what
    ``bind_turn_workspace`` registers and hashes into ``workspace_key``. So the
    mount is that cwd, never the profile's ``terminal.cwd``. A cwd whose digest
    is not ``workspace_key`` names some other container's mount: no root.
    """
    from tools.terminal_tool import workspace_key_for

    cwd = (workspace_cwd or "").strip()
    if not cwd or "\0" in cwd or not os.path.isabs(os.path.expanduser(cwd)):
        return []
    if workspace_key_for(cwd) != workspace_key:
        return []
    return _mounted_workspace_cwd(cwd)


def _default_docker_workspace_host_roots(
    session_key: str = "", workspace_key: str = "", workspace_cwd: str = "",
) -> list[Path]:
    """Host dirs that stand in for ``/workspace``.

    A cwd mount is tried first. With no workspace digest that mount is the
    profile's cwd and the only candidate, so MEDIA delivery (which has no
    session) stays cwd-only. A download that names a digest mounts that
    session's own cwd (``workspace_cwd``) instead, then tries that session's
    sandbox: the container key can be ``default:ws:<digest>`` while
    ``/workspace`` is still the sandbox directory, because the host cwd was
    not bind-mounted when the container was created.
    """
    tenv, truthy, persistent = _scope()
    if not persistent():
        return []
    roots: list[Path] = []
    mounted = tenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false").strip().lower() in truthy
    if mounted:
        if not _WORKSPACE_KEY_RE.fullmatch(workspace_key or ""):
            return _mounted_workspace_cwd()
        roots.extend(_session_mounted_workspace(workspace_key, workspace_cwd))
    for root in _docker_persistent_sandbox_roots(session_key, "workspace", workspace_key):
        if root not in roots:
            roots.append(root)
    return roots
