"""
NOX V3 Configuration Management
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import os


# Configuration file path
CONFIG_FILE = Path.home() / ".hermes" / "plugins" / "nox_v3" / "config.json"


def load_config() -> Dict[str, Any]:
    """
    Load NOX V3 configuration from file.

    Returns:
        Configuration dictionary
    """
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # Return default config on error
            return get_default_config()

    return get_default_config()


def save_config(config: Dict[str, Any]) -> bool:
    """
    Save NOX V3 configuration to file.

    Args:
        config: Configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except IOError:
        return False


def get_default_config() -> Dict[str, Any]:
    """
    Get default NOX V3 configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "enabled": False,
        "mode": "balanced",  # conservative | balanced | aggressive
        "max_daily_tokens": 10000,
        "latency_budget_ms": 50,
        "fast_path_threshold": 100,  # tokens - below this, skip verification
    }


def validate_config(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate NOX V3 configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check mode
    if "mode" in config:
        if config["mode"] not in ["conservative", "balanced", "aggressive"]:
            return False, f"Invalid mode: {config['mode']}"

    # Check max_daily_tokens
    if "max_daily_tokens" in config:
        if not isinstance(config["max_daily_tokens"], int) or config["max_daily_tokens"] < 0:
            return False, "max_daily_tokens must be a non-negative integer"

    # Check latency_budget_ms
    if "latency_budget_ms" in config:
        if not isinstance(config["latency_budget_ms"], int) or config["latency_budget_ms"] < 0:
            return False, "latency_budget_ms must be a non-negative integer"

    # Check fast_path_threshold
    if "fast_path_threshold" in config:
        if not isinstance(config["fast_path_threshold"], int) or config["fast_path_threshold"] < 0:
            return False, "fast_path_threshold must be a non-negative integer"

    return True, None


def merge_config(user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user configuration with defaults.

    Args:
        user_config: User-provided configuration

    Returns:
        Merged configuration
    """
    default = get_default_config()
    merged = default.copy()
    merged.update(user_config)
    return merged
