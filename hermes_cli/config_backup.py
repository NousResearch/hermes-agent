"""Config backup and restore for non-destructive configuration changes.

Backups are plain copies of config.yaml stored in ~/.hermes/config-backups/
with timestamped filenames. Selective restore can target just model_profiles
and model_routing sections without touching other config.
"""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BACKUP_DIR_NAME = "config-backups"
MAX_BACKUPS = 10


def get_backup_dir() -> Path:
    """Return the config backup directory path."""
    from hermes_cli.config import get_hermes_home
    return get_hermes_home() / BACKUP_DIR_NAME


def create_backup(config_path: Path, reason: str = "") -> Optional[Path]:
    """Create a timestamped backup of config.yaml.

    Args:
        config_path: Path to the current config.yaml.
        reason: Short tag for the filename (e.g. "routing", "reset", "pre_restore").

    Returns:
        Path to the backup file, or None if nothing to back up.
    """
    if not config_path.exists():
        return None

    backup_dir = get_backup_dir()
    backup_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = _sanitize_reason(reason)
    name = f"config_{stamp}_{tag}.yaml" if tag else f"config_{stamp}.yaml"
    backup_path = backup_dir / name

    # Avoid overwriting if called twice in the same second
    counter = 1
    while backup_path.exists():
        name = f"config_{stamp}_{tag}_{counter}.yaml" if tag else f"config_{stamp}_{counter}.yaml"
        backup_path = backup_dir / name
        counter += 1

    shutil.copy2(config_path, backup_path)
    _secure_file(backup_path)
    prune_backups()
    return backup_path


def prune_backups(max_count: int = MAX_BACKUPS) -> List[Path]:
    """Remove oldest backups beyond max_count.

    Returns list of deleted paths.
    """
    backup_dir = get_backup_dir()
    if not backup_dir.is_dir():
        return []

    backups = sorted(
        backup_dir.glob("config_*.yaml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    deleted = []
    for old in backups[max_count:]:
        try:
            old.unlink()
            deleted.append(old)
        except OSError as e:
            logger.debug("Could not delete old backup %s: %s", old, e)
    return deleted


def list_backups() -> List[Tuple[Path, datetime]]:
    """Return available backups as (path, mtime) tuples, newest first."""
    backup_dir = get_backup_dir()
    if not backup_dir.is_dir():
        return []

    result = []
    for p in backup_dir.glob("config_*.yaml"):
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        result.append((p, mtime))
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def restore_routing_sections(backup_path: Path) -> Dict[str, Any]:
    """Restore only model_profiles and model_routing from a backup.

    Creates a pre-restore backup of the current config first.
    Other config sections are left untouched.

    Returns the updated config dict.
    """
    import yaml
    from hermes_cli.config import load_config, save_config, get_config_path

    # Load the backup
    with open(backup_path, encoding="utf-8") as f:
        backup_data = yaml.safe_load(f) or {}

    # Pre-restore backup
    create_backup(get_config_path(), reason="pre_restore")

    # Load current config and selectively replace
    config = load_config()
    if "model_profiles" in backup_data:
        config["model_profiles"] = backup_data["model_profiles"]
    if "model_routing" in backup_data:
        config["model_routing"] = backup_data["model_routing"]

    save_config(config)
    return config


def restore_full(backup_path: Path) -> Dict[str, Any]:
    """Restore the entire config from a backup.

    Creates a pre-restore backup of the current config first.

    Returns the restored config dict.
    """
    import yaml
    from hermes_cli.config import save_config, get_config_path

    # Pre-restore backup
    create_backup(get_config_path(), reason="pre_restore")

    with open(backup_path, encoding="utf-8") as f:
        backup_data = yaml.safe_load(f) or {}

    save_config(backup_data)
    return backup_data


def format_backup_summary(backup_path: Path) -> str:
    """Return a one-line summary of what a backup contains."""
    import yaml

    try:
        with open(backup_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return "(unreadable)"

    parts = []

    profiles = data.get("model_profiles", {})
    if isinstance(profiles, dict):
        configured = [k for k, v in profiles.items()
                      if isinstance(v, dict) and v.get("model")]
        if configured:
            parts.append(f"profiles: {', '.join(configured)}")
        else:
            parts.append("profiles: defaults")
    else:
        parts.append("profiles: missing")

    rules = data.get("model_routing", {})
    if isinstance(rules, dict):
        rule_list = rules.get("rules", [])
        if rule_list:
            parts.append(f"routing: {len(rule_list)} rule(s)")

    model = data.get("model")
    if isinstance(model, dict):
        default = model.get("default", "")
        if default:
            parts.append(f"model: {default}")
    elif isinstance(model, str) and model:
        parts.append(f"model: {model}")

    return " | ".join(parts) if parts else "(empty config)"


def _sanitize_reason(reason: str) -> str:
    """Sanitize a reason tag for use in filenames."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", (reason or "").strip())
    return clean[:20].strip("_")


def _secure_file(path: Path) -> None:
    """Set restrictive permissions on a file (Unix only)."""
    try:
        path.chmod(0o600)
    except (OSError, AttributeError):
        pass  # Windows or permission error — acceptable
