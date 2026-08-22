"""
Plugin-owned config for dreaming.

Resolution order (later wins):
  1. Bundled defaults in plugins/dreaming/config.yaml
  2. $HERMES_HOME/dreaming/config.yaml (seeded on first register)
  3. dreaming: section in $HERMES_HOME/config.yaml (setup UX override)
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home

_BUNDLED_DEFAULTS = Path(__file__).with_name("config.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def bundled_defaults() -> dict[str, Any]:
    return _read_yaml(_BUNDLED_DEFAULTS)


def user_config_path(hermes_home: str | Path | None = None) -> Path:
    base = Path(hermes_home) if hermes_home else get_hermes_home()
    return base / "dreaming" / "config.yaml"


def ensure_user_config(hermes_home: str | Path | None = None) -> Path:
    """Seed $HERMES_HOME/dreaming/config.yaml from bundled defaults if missing."""
    path = user_config_path(hermes_home)
    if path.exists():
        return path
    defaults = bundled_defaults()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(defaults, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return path


def load_config(hermes_home: str | Path | None = None) -> dict[str, Any]:
    cfg = bundled_defaults()
    cfg = _deep_merge(cfg, _read_yaml(user_config_path(hermes_home)))

    main_cfg_path = (
        Path(hermes_home) / "config.yaml"
        if hermes_home
        else get_hermes_home() / "config.yaml"
    )
    main_section = _read_yaml(main_cfg_path).get("dreaming", {})
    if isinstance(main_section, dict):
        cfg = _deep_merge(cfg, main_section)
    return cfg


def is_enabled(cfg: dict[str, Any] | None = None) -> bool:
    data = cfg if cfg is not None else load_config()
    return bool(data.get("enabled", False))


def schedule(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("schedule", {})
    if not isinstance(block, dict):
        block = {}
    return {
        "min_hours": float(block.get("min_hours", 24)),
        "min_sessions": int(block.get("min_sessions", 5)),
        "poll_seconds": int(block.get("poll_seconds", 300)),
    }


def rem(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("rem", {})
    if not isinstance(block, dict):
        block = {}
    return {
        "model": str(block.get("model", "mistral:7b")),
        "base_url": str(block.get("base_url", "http://localhost:11434")).rstrip("/"),
    }


def promote_threshold(cfg: dict[str, Any]) -> float:
    block = cfg.get("scoring", {})
    if not isinstance(block, dict):
        return 0.55
    try:
        return float(block.get("promote_threshold", 0.55))
    except (TypeError, ValueError):
        return 0.55
