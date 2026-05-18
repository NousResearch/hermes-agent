"""Configuration parsing for the memory-integration provider."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_MEMORY_SUBDIR = "wiki/memory-integration"
DEFAULT_INCLUDE_ABSOLUTE_PATHS = False
_TRUE_STRINGS = {"true", "yes", "on", "1"}
_FALSE_STRINGS = {"false", "no", "off", "0"}


@dataclass(frozen=True)
class MemoryIntegrationConfig:
    mode: str | None
    vault_path: Path | None
    memory_subdir: str
    include_absolute_paths: bool


def _section(config: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    value = (config or {}).get(key, {})
    return value if isinstance(value, Mapping) else {}


def _runtime_config() -> Mapping[str, Any]:
    """Load the real Hermes config.yaml using the canonical config loader."""
    try:
        from hermes_cli.config import load_config

        loaded = load_config()
    except Exception:
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


def _parse_bool(value: Any, *, default: bool = DEFAULT_INCLUDE_ABSOLUTE_PATHS) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return default


def load_memory_integration_config(config: Mapping[str, Any] | None = None) -> MemoryIntegrationConfig:
    source = config if config is not None else _runtime_config()
    root = _section(source, "memory_integration")
    vault = _section(root, "vault")
    status = _section(root, "status")

    path_value = vault.get("path")
    vault_path = Path(path_value).expanduser() if isinstance(path_value, str) and path_value else None
    memory_subdir = vault.get("memory_subdir", DEFAULT_MEMORY_SUBDIR)
    if not isinstance(memory_subdir, str) or not memory_subdir:
        memory_subdir = DEFAULT_MEMORY_SUBDIR

    include_paths = status.get("include_absolute_paths", DEFAULT_INCLUDE_ABSOLUTE_PATHS)
    return MemoryIntegrationConfig(
        mode=vault.get("mode") if isinstance(vault.get("mode"), str) else None,
        vault_path=vault_path,
        memory_subdir=memory_subdir,
        include_absolute_paths=_parse_bool(include_paths),
    )
