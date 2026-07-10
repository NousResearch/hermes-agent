"""Small model-config restore points for risky provider/model writes."""

from __future__ import annotations

import copy
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODEL_CONFIG_BACKUP_KEEP = 5


def model_config_backup_paths(config_path: Path) -> list[Path]:
    return sorted(config_path.parent.glob(f"{config_path.name}.model-backup.*.bak"))


def _prune_model_config_backups(config_path: Path, *, keep: int) -> None:
    backups = model_config_backup_paths(config_path)
    for old in backups[:-keep]:
        try:
            old.unlink()
        except OSError:
            logger.debug("Failed to prune model config backup %s", old, exc_info=True)


def load_raw_config_for_model_write(config_path: Path) -> dict[str, Any]:
    import yaml

    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return loaded if isinstance(loaded, dict) else {}


def create_model_config_backup(
    config_path: Path,
    cfg: dict[str, Any],
    *,
    reason: str,
    keep: int = MODEL_CONFIG_BACKUP_KEEP,
) -> Path:
    """Snapshot the model-critical config slice before a risky write."""
    from utils import atomic_yaml_write

    now = datetime.now(timezone.utc)
    backup_path = config_path.with_name(
        f"{config_path.name}.model-backup.{now.strftime('%Y%m%d-%H%M%S-%f')}.bak"
    )
    payload = {
        "version": 2,
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "reason": reason,
        "had_model": "model" in cfg,
        "model": copy.deepcopy(cfg.get("model")),
        "had_custom_providers": "custom_providers" in cfg,
        "custom_providers": copy.deepcopy(cfg.get("custom_providers")),
        "had_providers": "providers" in cfg,
        "providers": copy.deepcopy(cfg.get("providers")),
    }
    atomic_yaml_write(backup_path, payload, sort_keys=False)
    try:
        os.chmod(backup_path, 0o600)
    except OSError:
        logger.debug("Failed to chmod model config backup %s", backup_path, exc_info=True)
    _prune_model_config_backups(config_path, keep=keep)
    return backup_path


def restore_latest_model_config_backup(
    config_path: Path,
) -> tuple[dict[str, Any] | None, Path | None, str]:
    """Return config restored from the latest model/custom-provider backup."""
    import yaml

    backups = model_config_backup_paths(config_path)
    if not backups:
        return None, None, "No model configuration backup is available yet."

    backup_path = backups[-1]
    with open(backup_path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict) or payload.get("version") not in {1, 2}:
        return None, backup_path, f"Model configuration backup {backup_path.name} is not readable."

    cfg = load_raw_config_for_model_write(config_path)
    if payload.get("had_model"):
        cfg["model"] = copy.deepcopy(payload.get("model"))
    else:
        cfg.pop("model", None)
    if payload.get("had_custom_providers"):
        cfg["custom_providers"] = copy.deepcopy(payload.get("custom_providers"))
    else:
        cfg.pop("custom_providers", None)
    if payload.get("version") >= 2:
        if payload.get("had_providers"):
            cfg["providers"] = copy.deepcopy(payload.get("providers"))
        else:
            cfg.pop("providers", None)

    return cfg, backup_path, f"Restored model configuration from {backup_path.name}."
