from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

import yaml

from hermes_cli import managed_scope
from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_constants import get_hermes_home


PRE_UPDATE_SNAPSHOT_KEEP = 1
PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE = 1 << 30
_ENV_REF = re.compile(r"\${([^}]+)}")


def normalize_pre_update_backup_mode(raw: Any) -> str:
    if raw is None:
        raise ValueError("updates.pre_update_backup must not be null")
    if isinstance(raw, bool):
        return "full" if raw else "off"

    value = str(raw).strip().lower()
    if value in {"off", "false", "none", "disabled"}:
        return "off"
    if value in {"full", "zip", "true"}:
        return "full"
    if value == "quick":
        return "quick"
    raise ValueError("unknown updates.pre_update_backup value")


def normalize_pre_update_backup_keep(raw: Any) -> int:
    return max(1, int(raw))


def _read_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as config_file:
            parsed = yaml.safe_load(config_file) or {}
    except FileNotFoundError:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} root must be a mapping")
    return parsed


def _updates_mapping(config: dict[str, Any], label: str) -> dict[str, Any]:
    updates = config.get("updates", {})
    if updates is None:
        return {}
    if not isinstance(updates, dict):
        raise ValueError(f"{label} updates must be a mapping")
    return updates


def _expand_update_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        reference = match.group(1).strip()
        if reference.startswith("env:"):
            reference = reference[4:].strip()
        elif ":" in reference and re.match(r"^[a-z][a-z0-9_-]*:", reference):
            return match.group(0)
        return os.environ.get(reference, match.group(0))

    return _ENV_REF.sub(replace, value)


def resolve_pre_update_backup_policy_strict() -> dict[str, int | str]:
    """Read the canonical update policy without mutating ``HERMES_HOME``."""
    defaults = dict(DEFAULT_CONFIG.get("updates", {}))
    home = get_hermes_home()
    user = _updates_mapping(
        _read_yaml_mapping(home / "config.yaml", "config"), "config"
    )

    managed: dict[str, Any] = {}
    managed_dir = managed_scope.get_managed_dir()
    if managed_dir is not None:
        managed = _updates_mapping(
            _read_yaml_mapping(managed_dir / "config.yaml", "managed config"),
            "managed config",
        )

    merged = {**defaults, **user, **managed}
    raw_mode = _expand_update_scalar(merged.get("pre_update_backup", "quick"))
    raw_keep = _expand_update_scalar(merged.get("backup_keep", 5))

    return {
        "mode": normalize_pre_update_backup_mode(raw_mode),
        "backup_keep": normalize_pre_update_backup_keep(raw_keep),
        "quick_keep": PRE_UPDATE_SNAPSHOT_KEEP,
        "quick_max_file_size": PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
    }
