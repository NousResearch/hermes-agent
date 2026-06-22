"""Helpers for project-local ``.mcp.json`` MCP server config."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_MCP_FILENAME = ".mcp.json"
PROJECT_MCP_ENV_FLAG = "HERMES_USE_PROJECT_MCP_JSON"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ProjectMcpConfig:
    """Project-local MCP servers loaded from a discovered ``.mcp.json`` file."""

    path: Path | None
    servers: dict[str, dict[str, Any]]


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
    return default


def project_mcp_enabled(config: Mapping[str, Any] | None = None) -> bool:
    """Return whether project ``.mcp.json`` loading is enabled for this run."""

    if _boolish(os.environ.get("HERMES_SAFE_MODE"), default=False):
        return False
    if _boolish(os.environ.get(PROJECT_MCP_ENV_FLAG), default=False):
        return True

    mcp_cfg = (config or {}).get("mcp")
    if not isinstance(mcp_cfg, Mapping):
        return False
    return _boolish(mcp_cfg.get("use_project_mcp_json"), default=False)


def _start_dir(cwd: str | os.PathLike[str] | None = None) -> Path:
    raw = cwd or os.environ.get("TERMINAL_CWD") or os.getcwd()
    path = Path(raw).expanduser()
    try:
        path = path.resolve(strict=False)
    except OSError:
        path = path.absolute()
    if path.is_file():
        return path.parent
    return path


def find_project_mcp_json(cwd: str | os.PathLike[str] | None = None) -> Path | None:
    """Find the nearest project-local ``.mcp.json`` at or above ``cwd``.

    The walk intentionally stops before the user's home directory so a broad
    home-level file is not treated as project-local config by accident.
    """

    current = _start_dir(cwd)
    try:
        home = Path.home().resolve(strict=False)
    except OSError:
        home = None

    while True:
        if home is not None and current == home:
            return None
        candidate = current / PROJECT_MCP_FILENAME
        if candidate.is_file():
            return candidate
        if current.parent == current:
            return None
        current = current.parent


def _normalize_url(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.startswith("<") and stripped.endswith(">") and len(stripped) > 2:
        return stripped[1:-1].strip()
    return value


def _extract_project_servers(data: Any, *, source: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(data, Mapping):
        logger.warning("Ignoring %s: root value must be an object", source)
        return {}

    raw_servers = data.get("mcpServers")
    if not isinstance(raw_servers, Mapping):
        raw_servers = data.get("servers")
    if not isinstance(raw_servers, Mapping):
        return {}

    servers: dict[str, dict[str, Any]] = {}
    for raw_name, raw_cfg in raw_servers.items():
        name = str(raw_name).strip()
        if not name:
            continue
        if not isinstance(raw_cfg, Mapping):
            logger.warning(
                "Ignoring MCP server %r in %s: entry must be an object",
                raw_name,
                source,
            )
            continue
        cfg = dict(raw_cfg)
        if "url" in cfg:
            cfg["url"] = _normalize_url(cfg["url"])
        servers[name] = cfg
    return servers


def load_project_mcp_servers(
    *,
    cwd: str | os.PathLike[str] | None = None,
    config: Mapping[str, Any] | None = None,
    force: bool = False,
) -> ProjectMcpConfig:
    """Load MCP servers from the nearest ``.mcp.json`` when enabled.

    ``force`` is used by explicit inspection commands such as
    ``hermes mcp list --include-project`` and ``hermes mcp test --from-project``.
    """

    if not force and not project_mcp_enabled(config):
        return ProjectMcpConfig(path=None, servers={})

    path = find_project_mcp_json(cwd)
    if path is None:
        return ProjectMcpConfig(path=None, servers={})

    try:
        with path.open(encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load project MCP config %s: %s", path, exc)
        return ProjectMcpConfig(path=path, servers={})

    return ProjectMcpConfig(path=path, servers=_extract_project_servers(data, source=path))


def merge_mcp_server_configs(
    config_servers: Mapping[str, Any] | None,
    project_servers: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Merge project and Hermes MCP configs with Hermes config taking priority."""

    merged: dict[str, dict[str, Any]] = {}
    for name, cfg in (project_servers or {}).items():
        if isinstance(cfg, Mapping):
            merged[str(name)] = dict(cfg)
    for name, cfg in (config_servers or {}).items():
        if isinstance(cfg, Mapping):
            merged[str(name)] = dict(cfg)
    return merged


def mcp_server_source_map(
    config_servers: Mapping[str, Any] | None,
    project_config: ProjectMcpConfig,
) -> dict[str, str]:
    """Return display sources for an effective MCP server mapping."""

    sources = {str(name): PROJECT_MCP_FILENAME for name in project_config.servers}
    for name in (config_servers or {}):
        sources[str(name)] = "config.yaml"
    return sources
