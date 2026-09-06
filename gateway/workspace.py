"""Gateway workspace normalization shared by routing and slash commands."""

from __future__ import annotations

import os
import posixpath
from pathlib import Path


class WorkspaceUnavailable(ValueError):
    """A configured workspace cannot be used by the selected backend."""


def configured_docker_host_mount_source(cwd: str = "") -> str:
    """Return the explicit configured Docker host mount matching ``cwd``."""
    from agent.runtime_cwd import scope_terminal_cwd
    from tools.terminal_scope import terminal_env
    from tools.terminal_tool_config import _is_host_cwd

    if terminal_env("TERMINAL_ENV", "local") != "docker":
        return ""
    mount_enabled = terminal_env(
        "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false",
    ).strip().lower() in {"1", "true", "yes", "on"}
    raw = scope_terminal_cwd().strip()
    if not mount_enabled or raw in {"", ".", "auto", "cwd"}:
        return ""
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute() or not candidate.is_dir():
        return ""
    candidate_text = str(candidate.resolve())
    if not _is_host_cwd(candidate_text) and candidate_text.startswith(("/workspace", "/root")):
        return ""
    if cwd and str(Path(cwd).expanduser().resolve()) != candidate_text:
        return ""
    return candidate_text


def normalize_gateway_workspace(raw: str, *, backend: str) -> str:
    """Return a usable backend-native cwd without dereferencing remote paths."""
    value = str(raw or "").strip()
    backend = str(backend or "local").strip().lower() or "local"
    if value in {"", ".", "auto", "cwd"}:
        if backend == "local":
            try:
                value = os.getcwd()
            except OSError as exc:
                raise WorkspaceUnavailable("the gateway launch directory is unavailable") from exc
        elif backend == "ssh":
            return "~"
        else:
            return "/root"

    if backend == "local":
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_dir():
            raise WorkspaceUnavailable(f"local workspace is not a directory: {value}")
        return str(path.resolve())

    if backend == "ssh" and (value == "~" or value.startswith("~/")):
        return value
    if not posixpath.isabs(value):
        raise WorkspaceUnavailable(f"{backend} workspace must be an absolute path: {value}")
    normalized = posixpath.normpath(value)
    from tools.terminal_tool_config import _is_container_backend, _is_unusable_container_cwd
    if _is_container_backend(backend) and _is_unusable_container_cwd(normalized):
        raise WorkspaceUnavailable(
            f"{backend} workspace must be a backend path such as /workspace/project: {value}"
        )
    return normalized


def configured_gateway_workspace(config, source) -> str:
    """Resolve channel/thread override, then the active profile terminal policy."""
    from agent.runtime_cwd import scope_terminal_cwd
    from tools.terminal_scope import terminal_env

    override = None
    platform_config = (getattr(config, "platforms", None) or {}).get(source.platform)
    overrides = getattr(platform_config, "channel_overrides", None) or {}
    keys = dict.fromkeys(
        str(value) for value in (source.chat_id, source.thread_id, source.parent_chat_id) if value
    )
    for key in keys:
        if key in overrides:
            override = overrides[key]
            break
    backend = terminal_env("TERMINAL_ENV", "local")
    if override and override.cwd is not None:
        return normalize_gateway_workspace(override.cwd, backend=backend)

    raw = scope_terminal_cwd()
    host_source = configured_docker_host_mount_source()
    if host_source:
        # Keep the host source as session truth. The terminal planner uses it
        # for the bind mount and maps the execution cwd to /workspace.
        return host_source
    return normalize_gateway_workspace(raw, backend=backend)
