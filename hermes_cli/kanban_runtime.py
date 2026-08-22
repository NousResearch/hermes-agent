"""Task-scoped Docker runtime contract for Kanban workers.

The Kanban dispatcher owns workspace resolution.  This module turns that
resolved workspace into a narrow, versioned runtime envelope that can cross the
worker subprocess boundary without abusing the user/profile
``TERMINAL_DOCKER_VOLUMES`` setting.

The envelope carries *local* canonical source paths.  The Docker-backed worker
translates those sources to Docker-host paths with ``docker_host_path_map``
after its profile config is loaded.  This keeps precedence explicit:

    task runtime > worker profile terminal config > global terminal config

Only dispatcher-generated envelopes are consumed.  Callers validate the task
id and workspace against ``HERMES_KANBAN_TASK`` / ``HERMES_KANBAN_WORKSPACE``
before using the mounts.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

KANBAN_TERMINAL_RUNTIME_ENV = "HERMES_KANBAN_TERMINAL_RUNTIME"
KANBAN_TERMINAL_RUNTIME_VERSION = 1
_WORKSPACE_TARGET = "/workspace"
_ALLOWED_WORKSPACE_KINDS = {"scratch", "dir", "worktree"}
_ALLOWED_PURPOSES = {"workspace", "git-common-dir"}


class KanbanRuntimeError(ValueError):
    """Raised when a Kanban runtime envelope is malformed or unsafe."""


def _canonical_existing_dir(value: str | os.PathLike[str], *, label: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise KanbanRuntimeError(f"{label} must be an absolute path: {value!r}")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise KanbanRuntimeError(f"{label} does not exist: {raw}") from exc
    if not resolved.is_dir():
        raise KanbanRuntimeError(f"{label} is not a directory: {resolved}")
    if resolved == Path(resolved.anchor):
        raise KanbanRuntimeError(f"refusing to mount filesystem root as {label}")
    return resolved


def _git_common_dir(workspace: Path) -> Path:
    """Return the canonical common Git directory for a linked worktree.

    A linked worktree's ``.git`` file points at ``<common>/.git/worktrees/...``
    and that administrative directory points back to the common dir.  Mounting
    only the worktree itself therefore breaks commits inside Docker.  We mount
    the common Git directory, not the parent repository working tree.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise KanbanRuntimeError(
            f"unable to resolve Git common dir for worktree {workspace}: {exc}"
        ) from exc
    if result.returncode != 0:
        raise KanbanRuntimeError(
            f"workspace_kind=worktree but git cannot resolve common dir for "
            f"{workspace}: {result.stderr.strip() or result.stdout.strip()}"
        )
    raw = result.stdout.strip()
    if not raw:
        raise KanbanRuntimeError(f"git returned an empty common dir for {workspace}")
    common = Path(raw)
    if not common.is_absolute():
        common = workspace / common
    return _canonical_existing_dir(common, label="git common dir")


def build_kanban_terminal_runtime(
    *,
    task_id: str,
    workspace_kind: str,
    workspace: str | os.PathLike[str],
    authorized_roots: Iterable[str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    """Build the dispatcher-owned runtime envelope for one task workspace.

    ``authorized_roots`` is provenance, not another workspace coordinate: the
    dispatcher derives it from existing board/project authority.  Docker later
    requires every physical bind source to remain inside one of these roots.
    """
    task_id = str(task_id or "").strip()
    if not task_id:
        raise KanbanRuntimeError("task_id is required")
    kind = str(workspace_kind or "scratch").strip().lower()
    if kind not in _ALLOWED_WORKSPACE_KINDS:
        raise KanbanRuntimeError(f"unsupported workspace kind: {workspace_kind!r}")

    ws = _canonical_existing_dir(workspace, label="kanban workspace")
    roots: list[Path] = []
    for value in authorized_roots or []:
        root = _canonical_existing_dir(value, label="authorized workspace root")
        if root not in roots:
            roots.append(root)
    mounts: list[dict[str, Any]] = [
        {
            "source": str(ws),
            "target": _WORKSPACE_TARGET,
            "read_only": False,
            "purpose": "workspace",
        }
    ]

    if kind == "worktree":
        common = _git_common_dir(ws)
        # A normal linked worktree's common dir is outside the worktree itself.
        # If Git reports a common dir inside the workspace, the workspace mount
        # already covers it and a second mount only adds ambiguity.
        try:
            common.relative_to(ws)
            common_inside_workspace = True
        except ValueError:
            common_inside_workspace = False
        if not common_inside_workspace:
            mounts.append(
                {
                    "source": str(common),
                    # Preserve the exact absolute path expected by the worktree's
                    # .git indirection.  This exposes Git administrative state,
                    # never the parent repository working tree.
                    "target": str(common),
                    "read_only": False,
                    "purpose": "git-common-dir",
                }
            )

    return {
        "version": KANBAN_TERMINAL_RUNTIME_VERSION,
        "task_id": task_id,
        "workspace_kind": kind,
        "workspace": str(ws),
        "authorized_roots": [str(root) for root in roots],
        "container_cwd": _WORKSPACE_TARGET,
        "mounts": mounts,
    }


def encode_kanban_terminal_runtime(runtime: Mapping[str, Any]) -> str:
    """Encode a runtime envelope deterministically for subprocess transport."""
    # Validate before transport so malformed data never crosses the boundary.
    normalized = validate_kanban_terminal_runtime(runtime)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def validate_kanban_terminal_runtime(
    runtime: Mapping[str, Any],
    *,
    expected_task_id: str | None = None,
    expected_workspace: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate and normalize an already-decoded runtime envelope."""
    if not isinstance(runtime, Mapping):
        raise KanbanRuntimeError("runtime envelope must be a mapping")
    try:
        version = int(runtime.get("version"))
    except (TypeError, ValueError) as exc:
        raise KanbanRuntimeError("runtime envelope has no valid version") from exc
    if version != KANBAN_TERMINAL_RUNTIME_VERSION:
        raise KanbanRuntimeError(
            f"unsupported Kanban terminal runtime version {version}; "
            f"expected {KANBAN_TERMINAL_RUNTIME_VERSION}"
        )

    task_id = str(runtime.get("task_id") or "").strip()
    if not task_id:
        raise KanbanRuntimeError("runtime task_id is required")
    if expected_task_id is not None and task_id != str(expected_task_id):
        raise KanbanRuntimeError(
            f"runtime task mismatch: envelope={task_id!r} expected={expected_task_id!r}"
        )

    kind = str(runtime.get("workspace_kind") or "").strip().lower()
    if kind not in _ALLOWED_WORKSPACE_KINDS:
        raise KanbanRuntimeError(f"invalid runtime workspace kind: {kind!r}")

    workspace = _canonical_existing_dir(
        str(runtime.get("workspace") or ""), label="runtime workspace"
    )
    if expected_workspace is not None:
        expected = _canonical_existing_dir(expected_workspace, label="expected workspace")
        if workspace != expected:
            raise KanbanRuntimeError(
                f"runtime workspace mismatch: envelope={workspace} expected={expected}"
            )

    raw_authorized_roots = runtime.get("authorized_roots", [])
    if not isinstance(raw_authorized_roots, list):
        raise KanbanRuntimeError("runtime authorized_roots must be a list")
    authorized_roots: list[Path] = []
    for idx, value in enumerate(raw_authorized_roots):
        root = _canonical_existing_dir(
            str(value or ""), label=f"authorized workspace root #{idx}"
        )
        if root not in authorized_roots:
            authorized_roots.append(root)

    if runtime.get("container_cwd") != _WORKSPACE_TARGET:
        raise KanbanRuntimeError(
            f"runtime container_cwd must be {_WORKSPACE_TARGET!r}"
        )

    raw_mounts = runtime.get("mounts")
    if not isinstance(raw_mounts, list) or not raw_mounts:
        raise KanbanRuntimeError("runtime mounts must be a non-empty list")

    normalized_mounts: list[dict[str, Any]] = []
    workspace_mounts = 0
    git_common_mounts = 0
    # Recompute worktree metadata from the validated workspace instead of
    # trusting an envelope-provided path.  This validates mount SHAPE only;
    # dispatcher-derived provenance containment is enforced at Docker
    # consumption before any remote-host path translation.
    expected_git_common: Path | None = None
    expected_git_common_needs_mount = False
    if kind == "worktree":
        expected_git_common = _git_common_dir(workspace)
        try:
            expected_git_common.relative_to(workspace)
        except ValueError:
            expected_git_common_needs_mount = True
    targets: set[str] = set()
    for idx, mount in enumerate(raw_mounts):
        if not isinstance(mount, Mapping):
            raise KanbanRuntimeError(f"runtime mount #{idx} is not a mapping")
        purpose = str(mount.get("purpose") or "").strip()
        if purpose not in _ALLOWED_PURPOSES:
            raise KanbanRuntimeError(f"runtime mount #{idx} has invalid purpose {purpose!r}")
        if purpose == "git-common-dir" and kind != "worktree":
            raise KanbanRuntimeError("git-common-dir mount is only valid for worktree tasks")

        source = _canonical_existing_dir(
            str(mount.get("source") or ""), label=f"runtime mount #{idx} source"
        )
        target = str(mount.get("target") or "").strip()
        if not target or not os.path.isabs(target):
            raise KanbanRuntimeError(f"runtime mount #{idx} target must be absolute")
        if "," in target or "," in str(source):
            # Docker --mount uses a comma-delimited key/value grammar.
            raise KanbanRuntimeError(
                "Kanban runtime mount paths containing commas are not supported"
            )
        if target in targets:
            raise KanbanRuntimeError(f"duplicate runtime mount target: {target}")
        targets.add(target)

        read_only = bool(mount.get("read_only", False))
        if purpose == "workspace":
            workspace_mounts += 1
            if target != _WORKSPACE_TARGET or source != workspace or read_only:
                raise KanbanRuntimeError(
                    "workspace mount must map the exact runtime workspace rw to /workspace"
                )
        elif purpose == "git-common-dir":
            git_common_mounts += 1
            # Git commits update objects/refs; read-only would make a seemingly
            # healthy worktree fail only at commit time.  More importantly,
            # never trust an arbitrary envelope path here: re-derive the common
            # directory from the validated worktree and require an exact match.
            if read_only:
                raise KanbanRuntimeError("git-common-dir mount must be read-write")
            if expected_git_common is None or source != expected_git_common:
                raise KanbanRuntimeError(
                    "git-common-dir mount does not match this worktree's Git metadata"
                )
            if target != str(expected_git_common):
                raise KanbanRuntimeError(
                    "git-common-dir mount target must preserve Git's absolute metadata path"
                )

        normalized_mounts.append(
            {
                "source": str(source),
                "target": target,
                "read_only": read_only,
                "purpose": purpose,
            }
        )

    if workspace_mounts != 1:
        raise KanbanRuntimeError("runtime must contain exactly one workspace mount")
    if kind == "worktree":
        expected_count = 1 if expected_git_common_needs_mount else 0
        if git_common_mounts != expected_count:
            raise KanbanRuntimeError(
                "runtime must contain exactly the Git metadata mount required by the worktree"
            )
    elif git_common_mounts:
        raise KanbanRuntimeError("non-worktree runtime cannot mount Git common metadata")

    return {
        "version": version,
        "task_id": task_id,
        "workspace_kind": kind,
        "workspace": str(workspace),
        "authorized_roots": [str(root) for root in authorized_roots],
        "container_cwd": _WORKSPACE_TARGET,
        "mounts": normalized_mounts,
    }


def decode_kanban_terminal_runtime(
    raw: str,
    *,
    expected_task_id: str | None = None,
    expected_workspace: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Decode and validate a dispatcher runtime envelope."""
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise KanbanRuntimeError("invalid JSON in Kanban terminal runtime") from exc
    return validate_kanban_terminal_runtime(
        payload,
        expected_task_id=expected_task_id,
        expected_workspace=expected_workspace,
    )


def _normalize_path_map_entry(entry: Any) -> tuple[Path, Path]:
    if not isinstance(entry, Mapping):
        raise KanbanRuntimeError("docker_host_path_map entries must be mappings")
    local_raw = str(entry.get("local_root") or "").strip()
    host_raw = str(entry.get("host_root") or "").strip()
    if not local_raw or not host_raw:
        raise KanbanRuntimeError(
            "docker_host_path_map entries require local_root and host_root"
        )
    local = _canonical_existing_dir(local_raw, label="docker_host_path_map local_root")
    host = Path(host_raw).expanduser()
    if not host.is_absolute():
        raise KanbanRuntimeError("docker_host_path_map host_root must be absolute")
    if "," in str(host):
        raise KanbanRuntimeError("docker_host_path_map host_root cannot contain commas")
    return local, host


def is_remote_docker_host(docker_host: str | None) -> bool:
    """Conservatively classify Docker hosts that refer to another machine."""
    raw = str(docker_host or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    if lowered.startswith(("unix://", "npipe://", "fd://")):
        return False
    if "://" not in raw:
        # Docker accepts bare local socket-ish values through wrappers; do not
        # break them by assuming remote.
        return False
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if parsed.scheme in {"tcp", "http", "https", "ssh"}:
        return host not in {"", "localhost", "127.0.0.1", "::1"}
    return False


def translate_host_path(
    source: str | os.PathLike[str],
    *,
    path_map: Iterable[Mapping[str, Any]] | None,
    docker_host: str | None,
) -> str:
    """Translate a local canonical path to the Docker daemon's host path.

    Longest-prefix match wins.  A genuinely remote Docker daemon must have an
    explicit mapping, including an explicit identity mapping when both machines
    intentionally use the same absolute paths.  This prevents Docker from
    silently creating an empty bind source on the remote host.
    """
    src = _canonical_existing_dir(source, label="runtime mount source")
    mappings = [_normalize_path_map_entry(entry) for entry in (path_map or [])]
    mappings.sort(key=lambda pair: len(str(pair[0])), reverse=True)

    for local_root, host_root in mappings:
        try:
            rel = src.relative_to(local_root)
        except ValueError:
            continue
        translated = host_root / rel
        return str(translated)

    if is_remote_docker_host(docker_host):
        raise KanbanRuntimeError(
            f"Docker daemon {docker_host!r} is remote but no docker_host_path_map "
            f"entry covers {src}; refusing an unverified bind mount"
        )
    return str(src)


def _path_within_authority(path: Path, roots: Iterable[Path]) -> bool:
    """Return whether *path* is equal to or below one trusted authority root."""
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def translate_runtime_mounts(
    runtime: Mapping[str, Any],
    *,
    path_map: Iterable[Mapping[str, Any]] | None,
    docker_host: str | None,
) -> list[dict[str, Any]]:
    """Authorize local bind sources, then translate them for the Docker host."""
    normalized = validate_kanban_terminal_runtime(runtime)
    authorized_roots = [Path(value) for value in normalized["authorized_roots"]]
    if not authorized_roots:
        raise KanbanRuntimeError(
            "Kanban Docker runtime has no authorized workspace roots; "
            "refusing host bind mounts"
        )

    workspace = Path(normalized["workspace"])
    if not _path_within_authority(workspace, authorized_roots):
        raise KanbanRuntimeError(
            f"runtime workspace {workspace} is outside authorized workspace roots"
        )

    translated: list[dict[str, Any]] = []
    for mount in normalized["mounts"]:
        # Authorize the canonical LOCAL source before any remote-host path
        # translation.  That includes worktree Git administrative state: a
        # valid task workspace must not be able to smuggle an unrelated common
        # Git directory into the container.
        source = Path(mount["source"])
        if not _path_within_authority(source, authorized_roots):
            raise KanbanRuntimeError(
                f"runtime mount source {source} ({mount['purpose']}) is outside "
                "authorized workspace roots"
            )
        translated.append(
            {
                "source": translate_host_path(
                    source, path_map=path_map, docker_host=docker_host
                ),
                "target": mount["target"],
                "read_only": mount["read_only"],
                "purpose": mount["purpose"],
            }
        )
    return translated
