"""Gateway config-loading helpers — extracted from gateway/run.py.

All static config-loading methods that GatewayRunner used to own inline.
Each function is self-contained and takes no gateway-runner state.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_hermes_home = get_hermes_home()


def load_gateway_config() -> dict:
    """Load and parse ~/.hermes/config.yaml, returning {} on any error."""
    config_path = _hermes_home / 'config.yaml'
    try:
        from hermes_cli.config import get_config_path, read_raw_config
        if config_path == get_config_path():
            return read_raw_config()
    except Exception:
        pass
    try:
        if config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
    except Exception:
        logger.debug("Could not load gateway config from %s", config_path)
    return {}


def resolve_gateway_model(config: dict | None = None) -> str:
    """Read model from config.yaml — single source of truth."""
    cfg = config if config is not None else load_gateway_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, str):
        return model_cfg
    elif isinstance(model_cfg, dict):
        return model_cfg.get("default") or model_cfg.get("model") or ""
    return ""


def load_prefill_messages(config: dict | None = None) -> List[Dict[str, Any]]:
    """Load prefill messages from config.yaml."""
    cfg = config if config is not None else load_gateway_config()
    raw = cfg.get("prefill", [])
    if isinstance(raw, list):
        return raw
    return []


def load_ephemeral_system_prompt(config: dict | None = None) -> str:
    """Load the ephemeral system prompt override from config."""
    cfg = config if config is not None else load_gateway_config()
    return str(cfg.get("ephemeral_system_prompt") or cfg.get("system_prompt") or "")


def load_reasoning_config(config: dict | None = None) -> dict | None:
    """Load reasoning config from config.yaml."""
    cfg = config if config is not None else load_gateway_config()
    raw = cfg.get("reasoning")
    if isinstance(raw, dict):
        return raw
    return None


def parse_reasoning_command_args(raw_args: str) -> tuple[str, bool]:
    """Parse reasoning command args into (level, show_budget)."""
    parts = raw_args.strip().split()
    level = parts[0] if parts else "medium"
    show_budget = "--budget" in parts or "-b" in parts
    return level, show_budget


def load_service_tier(config: dict | None = None) -> str | None:
    """Load service_tier from config.yaml."""
    cfg = config if config is not None else load_gateway_config()
    raw = cfg.get("service_tier")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def load_show_reasoning(config: dict | None = None) -> bool:
    """Load show_reasoning from config.yaml."""
    cfg = config if config is not None else load_gateway_config()
    return bool(cfg.get("show_reasoning", True))


def load_busy_input_mode(config: dict | None = None) -> str:
    """Load busy_input_mode from config.yaml."""
    cfg = config if config is not None else load_gateway_config()
    raw = cfg.get("busy_input_mode", "interrupt")
    return str(raw).strip().lower() if isinstance(raw, str) else "interrupt"


def load_restart_drain_timeout(config: dict | None = None) -> float:
    """Load restart drain timeout from config.yaml."""
    cfg = config if config is not None else load_gateway_config()
    try:
        return float(cfg.get("restart_drain_timeout", 30.0))
    except (TypeError, ValueError):
        return 30.0


def load_background_notifications_mode(config: dict | None = None) -> str:
    """Load background_process_notifications from config.yaml."""
    cfg = config if config is not None else load_gateway_config()
    raw = cfg.get("background_process_notifications", "smart")
    return str(raw).strip().lower() if isinstance(raw, str) else "smart"


def load_provider_routing(config: dict | None = None) -> dict:
    """Load provider_routing from config.yaml."""
    cfg = config if config is not None else load_gateway_config()
    raw = cfg.get("provider_routing", {})
    return raw if isinstance(raw, dict) else {}


def load_fallback_model(config: dict | None = None) -> list | dict | None:
    """Load fallback_model from config.yaml."""
    cfg = config if config is not None else load_gateway_config()
    raw = cfg.get("fallback_model")
    if isinstance(raw, (list, dict)):
        return raw
    return None


def platform_config_key(platform: Any) -> str:
    """Map a Platform enum to its config.yaml key."""
    from gateway.platform_registry import Platform
    return "cli" if platform == Platform.LOCAL else platform.value


def teams_pipeline_plugin_enabled(config: dict | None = None) -> bool:
    """Return True when the standalone Teams pipeline plugin is enabled."""
    cfg = config if config is not None else load_gateway_config()
    enabled = cfg_get_safe(cfg, "plugins", "enabled", default=[])
    if not isinstance(enabled, list):
        return False
    return "teams_pipeline" in enabled or "teams-pipeline" in enabled


def cfg_get_safe(cfg: dict, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested config dict."""
    current = cfg
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current
