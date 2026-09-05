"""Strict, fail-closed configuration for the Phase 1 control-plane state."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from hermes_constants import get_hermes_home
from plugins.agentops.control.models import AuthorityMode


CONFIG_SCHEMA_VERSION = 1
STATE_MARKER = ".agentops-state"
DEFAULT_SPOOL_MAX_BYTES = 256 * 1024 * 1024


class StateDirectoryError(RuntimeError):
    """Raised before the daemon can mutate an unsafe state location."""


@dataclass(frozen=True)
class AgentOpsConfig:
    config_path: Path
    state_dir: Path
    sqlite_path: Path
    spool_dir: Path
    socket_path: Path
    backup_dir: Path
    lock_path: Path
    event_spool_max_bytes: int
    default_authority: AuthorityMode
    global_write_enabled: bool
    safe_start_reasons: tuple[str, ...]
    state_dir_safe: bool


def default_config_path() -> Path:
    return Path(get_hermes_home()) / "agentops" / "agentops.yaml"


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _canonical(path: Path) -> Path:
    return _absolute(path).resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _has_symlink_component(path: Path) -> bool:
    """Reject a path when any existing component is a symlink.

    A non-existing tail is accepted so a first explicit daemon start can create
    it, but it is always canonicalized before that happens.
    """
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
    return False


def _inside_git_worktree(path: Path) -> bool:
    current = path
    while current != current.parent:
        marker = current / ".git"
        if marker.exists() or marker.is_symlink():
            return True
        current = current.parent
    return False


def _resolve_config_path(value: Any, *, base: Path, fallback: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        return fallback
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else base / candidate


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _marker_is_valid(state_dir: Path) -> bool:
    marker = state_dir / STATE_MARKER
    try:
        if marker.is_symlink() or not marker.is_file():
            return False
        data = json.loads(marker.read_text(encoding="utf-8"))
        return data == {"schema_version": CONFIG_SCHEMA_VERSION, "kind": "agentops-state"}
    except (OSError, json.JSONDecodeError):
        return False


def _state_dir_reasons(state_dir: Path) -> list[str]:
    reasons: list[str] = []
    hermes_home = _canonical(Path(get_hermes_home()))
    if state_dir == hermes_home:
        reasons.append("hermes_root_state_dir_rejected")
    if _has_symlink_component(state_dir):
        reasons.append("state_dir_symlink_rejected")
    if _inside_git_worktree(state_dir):
        reasons.append("git_worktree_state_dir_rejected")
    try:
        if state_dir.exists():
            metadata = state_dir.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                reasons.append("state_dir_not_directory")
            elif metadata.st_uid != os.getuid():
                reasons.append("state_dir_owner_rejected")
            elif not _marker_is_valid(state_dir):
                reasons.append("unmanaged_state_dir_rejected")
    except OSError:
        reasons.append("state_dir_unreadable")
    return reasons


def _safe_default(path: Path, reasons: tuple[str, ...]) -> AgentOpsConfig:
    state_dir = _canonical(Path(get_hermes_home()) / "agentops")
    return AgentOpsConfig(
        config_path=path,
        state_dir=state_dir,
        sqlite_path=state_dir / "state.db",
        spool_dir=state_dir / "event-spool",
        socket_path=state_dir / "agentops.sock",
        backup_dir=state_dir / "backups",
        lock_path=state_dir / "agentops.lock",
        event_spool_max_bytes=DEFAULT_SPOOL_MAX_BYTES,
        default_authority=AuthorityMode.OBSERVE_ONLY,
        global_write_enabled=False,
        safe_start_reasons=reasons,
        state_dir_safe=False,
    )


def load_agentops_config(path: Path) -> AgentOpsConfig:
    """Read config without creating paths; unsafe layouts are never repaired."""
    path = _absolute(Path(path))
    if not path.exists():
        return _safe_default(path, ("config_missing",))
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return _safe_default(path, ("config_invalid",))
    if not isinstance(parsed, Mapping):
        return _safe_default(path, ("config_invalid",))

    reasons: list[str] = []
    if parsed.get("schema_version", CONFIG_SCHEMA_VERSION) != CONFIG_SCHEMA_VERSION:
        reasons.append("unsupported_config_schema")
    storage = _as_mapping(parsed.get("storage"))
    control_plane = _as_mapping(parsed.get("control_plane"))
    safety = _as_mapping(parsed.get("safety"))

    default_state = _canonical(Path(get_hermes_home()) / "agentops")
    configured_state = _resolve_config_path(storage.get("state_dir"), base=path.parent, fallback=default_state)
    lexical_state = _absolute(configured_state)
    state_dir = _canonical(configured_state)
    if lexical_state != state_dir:
        reasons.append("state_dir_symlink_rejected")
    reasons.extend(_state_dir_reasons(state_dir))

    sqlite_path = _canonical(
        _resolve_config_path(storage.get("sqlite_path"), base=state_dir, fallback=state_dir / "state.db")
    )
    spool_dir = _canonical(
        _resolve_config_path(storage.get("spool_dir"), base=state_dir, fallback=state_dir / "event-spool")
    )
    socket_path = _canonical(
        _resolve_config_path(control_plane.get("socket_path"), base=state_dir, fallback=state_dir / "agentops.sock")
    )
    backup_dir = state_dir / "backups"
    lock_path = state_dir / "agentops.lock"
    for label, candidate in (("sqlite", sqlite_path), ("spool", spool_dir), ("socket", socket_path)):
        if not _is_within(candidate, state_dir) or candidate == state_dir:
            reasons.append(f"{label}_outside_state_dir")
        if _has_symlink_component(candidate):
            reasons.append(f"{label}_symlink_rejected")
    if sqlite_path.name != "state.db" or spool_dir.name != "event-spool" or socket_path.name != "agentops.sock":
        reasons.append("state_layout_rejected")

    raw_spool_mb = control_plane.get("event_spool_max_mb", 256)
    if not isinstance(raw_spool_mb, int) or isinstance(raw_spool_mb, bool) or raw_spool_mb <= 0:
        reasons.append("invalid_spool_budget")
        spool_bytes = DEFAULT_SPOOL_MAX_BYTES
    else:
        spool_bytes = raw_spool_mb * 1024 * 1024

    if safety.get("default_authority", AuthorityMode.OBSERVE_ONLY.value) != AuthorityMode.OBSERVE_ONLY.value:
        reasons.append("unsupported_authority_requested")
    if safety.get("global_write_enabled") is True:
        reasons.append("write_requested_but_disabled")

    deduplicated = tuple(dict.fromkeys(reasons))
    return AgentOpsConfig(
        config_path=path,
        state_dir=state_dir,
        sqlite_path=sqlite_path,
        spool_dir=spool_dir,
        socket_path=socket_path,
        backup_dir=backup_dir,
        lock_path=lock_path,
        event_spool_max_bytes=spool_bytes,
        default_authority=AuthorityMode.OBSERVE_ONLY,
        global_write_enabled=False,
        safe_start_reasons=deduplicated,
        state_dir_safe=not deduplicated,
    )


def initialize_state_dir(config: AgentOpsConfig) -> None:
    """Create a new dedicated AgentOps root or validate its existing marker."""
    if not config.state_dir_safe:
        raise StateDirectoryError("unsafe state directory")
    state_dir = config.state_dir
    if state_dir.exists():
        if not _marker_is_valid(state_dir):
            raise StateDirectoryError("unmanaged state directory")
    else:
        state_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        marker = state_dir / STATE_MARKER
        marker.write_text(
            json.dumps({"schema_version": CONFIG_SCHEMA_VERSION, "kind": "agentops-state"}, sort_keys=True),
            encoding="utf-8",
        )
    try:
        os.chmod(state_dir, 0o700)
        os.chmod(state_dir / STATE_MARKER, 0o600)
        status = state_dir.stat()
    except OSError as exc:
        raise StateDirectoryError("state directory permissions invalid") from exc
    if status.st_uid != os.getuid() or stat.S_IMODE(status.st_mode) != 0o700:
        raise StateDirectoryError("state directory permissions invalid")
    marker_descriptor = os.open(state_dir / STATE_MARKER, os.O_RDONLY)
    state_descriptor = os.open(state_dir, os.O_RDONLY)
    try:
        os.fsync(marker_descriptor)
        os.fsync(state_descriptor)
    except OSError as exc:
        raise StateDirectoryError("state directory durability invalid") from exc
    finally:
        os.close(marker_descriptor)
        os.close(state_descriptor)


def path_is_within_state(config: AgentOpsConfig, path: Path) -> bool:
    canonical = _canonical(Path(path))
    return _is_within(canonical, config.state_dir) and not _has_symlink_component(canonical)
