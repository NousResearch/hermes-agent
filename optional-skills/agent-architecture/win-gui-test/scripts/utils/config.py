"""Configuration loading — YAML/JSON with env-var override."""

import os
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


DEFAULT_CONFIG = {
    "screenshot_dir": os.path.expanduser("~/Desktop/gui_screenshots"),
    "log_dir": os.path.expanduser("~/win-gui-test-logs"),
    "temp_dir": os.path.expanduser("~/Temp"),
    "retry": {"count": 3, "delay": 1.0},
    "timeout": {"window": 10, "control": 5},
    "screenshot": {
        "fallback_to_pil": True,
        "format": "png",
    },
}


def _merge_deep(base: dict, override: dict) -> dict:
    """Deep-merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_deep(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str | None = None) -> dict:
    """Load config from YAML/JSON file, overlaying env-var overrides.

    Environment variable overrides (uppercase, dot-separated):
      WG_SCREENSHOT_DIR=/path        -> config["screenshot_dir"]
      WG_RETRY_COUNT=5               -> config["retry"]["count"]
    """
    cfg = DEFAULT_CONFIG.copy()

    if path:
        path = os.path.expanduser(path)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                if path.endswith((".yaml", ".yml")):
                    if yaml is None:
                        raise ImportError("PyYAML is required for .yaml config files")
                    file_cfg: dict = yaml.safe_load(f) or {}
                else:
                    file_cfg = json.load(f)
            cfg = _merge_deep(cfg, file_cfg)

    # Env-var overrides
    for key, val in os.environ.items():
        if key.startswith("WG_"):
            parts = key[3:].lower().split("_")
            target = cfg
            for p in parts[:-1]:
                if p not in target:
                    break
                target = target[p]
            else:
                last = parts[-1]
                if last in target and isinstance(target[last], (str, int, float)):
                    try:
                        target[last] = int(val)
                    except ValueError:
                        try:
                            target[last] = float(val)
                        except ValueError:
                            target[last] = val
    return cfg
