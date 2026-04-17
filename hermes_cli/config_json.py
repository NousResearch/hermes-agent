"""
JSON Configuration Loader for Hermes Agent.

This module provides a modern JSON-based configuration system inspired by
openclaw.json, featuring:
- Centralized provider management (one API key, multiple models)
- Environment variable substitution (${VAR} syntax)
- Backward compatibility with config.yaml
- Clean, readable structure

Usage:
    from hermes_cli.config_json import load_config_json
    
    config = load_config_json()  # Loads config.json or falls back to config.yaml
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from hermes_cli.config import get_hermes_home, get_config_path, get_env_path, DEFAULT_CONFIG


# =============================================================================
# Environment Variable Expansion
# =============================================================================

_ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')


def expand_env_vars(obj: Any, env: Optional[Dict[str, str]] = None) -> Any:
    """
    Recursively expand ${VAR} references in config values.
    
    Args:
        obj: The object to process (dict, list, or string)
        env: Environment variables dict (defaults to os.environ)
    
    Returns:
        The object with all ${VAR} references replaced with environment values.
        Unresolved variables are kept verbatim.
    """
    if env is None:
        env = os.environ
    
    if isinstance(obj, str):
        def replacer(match):
            var_name = match.group(1)
            return env.get(var_name, match.group(0))  # Keep original if not found
        
        return _ENV_VAR_PATTERN.sub(replacer, obj)
    
    if isinstance(obj, dict):
        return {k: expand_env_vars(v, env) for k, v in obj.items()}
    
    if isinstance(obj, list):
        return [expand_env_vars(item, env) for item in obj]
    
    return obj


# =============================================================================
# JSON Config Loading
# =============================================================================

def get_config_json_path() -> Path:
    """Get the JSON config file path."""
    return get_hermes_home() / "config.json"


def load_env_file() -> Dict[str, str]:
    """Load environment variables from .env file."""
    env_path = get_env_path()
    env_vars = {}
    
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    env_vars[key.strip()] = value.strip().strip('"\'')
    
    return env_vars


def load_config_json(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from config.json with environment variable expansion.
    
    Args:
        config_path: Optional path to config file (defaults to ~/.hermes/config.json)
    
    Returns:
        Configuration dictionary with expanded environment variables
    
    Raises:
        FileNotFoundError: If config.json doesn't exist
        json.JSONDecodeError: If config.json is invalid JSON
    """
    if config_path is None:
        config_path = get_config_json_path()
    
    # Load .env file and merge with os.environ
    env_file_vars = load_env_file()
    effective_env = {**os.environ, **env_file_vars}
    
    # Load JSON config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Expand environment variables
    config = expand_env_vars(config, effective_env)
    
    return config


def config_exists_json() -> bool:
    """Check if config.json exists."""
    return get_config_json_path().exists()


# =============================================================================
# YAML to JSON Migration
# =============================================================================

def migrate_yaml_to_json(yaml_path: Optional[Path] = None, 
                         json_path: Optional[Path] = None,
                         dry_run: bool = False) -> Dict[str, Any]:
    """
    Migrate config.yaml to config.json format.
    
    This function converts the existing YAML configuration to the new JSON
    format with centralized provider management.
    
    Args:
        yaml_path: Path to source YAML file (defaults to ~/.hermes/config.yaml)
        json_path: Path to destination JSON file (defaults to ~/.hermes/config.json)
        dry_run: If True, don't write files, just return the result
    
    Returns:
        Dictionary containing:
        - success: bool
        - config: dict (the migrated config)
        - warnings: list of warning messages
        - errors: list of error messages
    """
    result = {
        "success": False,
        "config": {},
        "warnings": [],
        "errors": []
    }
    
    if yaml_path is None:
        yaml_path = get_config_path()
    
    if json_path is None:
        json_path = get_config_json_path()
    
    # Check source file exists
    if not yaml_path.exists():
        result["errors"].append(f"Source file not found: {yaml_path}")
        return result
    
    # Load YAML config
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f) or {}
    except Exception as e:
        result["errors"].append(f"Failed to load YAML: {e}")
        return result
    
    # Load .env for API key references
    env_vars = load_env_file()
    
    # Convert to JSON format
    json_config = convert_yaml_to_json_structure(yaml_config, env_vars)
    
    result["config"] = json_config
    
    # Write if not dry run
    if not dry_run:
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_config, f, indent=2, ensure_ascii=False)
            result["success"] = True
        except Exception as e:
            result["errors"].append(f"Failed to write JSON: {e}")
    else:
        # Dry run is always successful if we got here
        result["success"] = True
    
    return result


def convert_yaml_to_json_structure(yaml_config: Dict[str, Any], 
                                    env_vars: Dict[str, str]) -> Dict[str, Any]:
    """
    Convert YAML config structure to JSON format with centralized providers.
    
    This is the core migration logic that restructures the config.
    """
    json_config = {
        "$schema": "https://hermes-agent.dev/schemas/config.v1.json",
        "_version": 1,
        "_comment": "Migrated from config.yaml by Hermes Agent",
    }
    
    # Extract provider configurations
    providers = {}
    
    # Handle custom_providers from YAML
    custom_providers = yaml_config.get("custom_providers", [])
    for provider in custom_providers:
        name = provider.get("name", "unknown")
        providers[name] = {
            "base_url": provider.get("base_url", ""),
            "api_key": mask_api_key(provider.get("api_key", "")),
            "models": []  # Will be populated from model config
        }
    
    # Add providers from auxiliary configs
    auxiliary = yaml_config.get("auxiliary", {})
    for feature_name, feature_config in auxiliary.items():
        provider_name = feature_config.get("provider", "")
        if provider_name and provider_name != "auto":
            if provider_name not in providers:
                providers[provider_name] = {
                    "base_url": feature_config.get("base_url", ""),
                    "api_key": "",
                    "models": []
                }
    
    # Extract model information
    model_config = yaml_config.get("model", {})
    default_model = model_config.get("default", "")
    
    # Build providers section with models
    if providers:
        json_config["providers"] = providers
    
    # Defaults section
    json_config["defaults"] = {
        "primary_model": default_model or yaml_config.get("model", ""),
        "max_turns": yaml_config.get("agent", {}).get("max_turns", 90),
        "personality": yaml_config.get("display", {}).get("personality", "kawaii")
    }
    
    # Features section (auxiliary models)
    features = {}
    for feature_name, feature_config in auxiliary.items():
        if feature_config.get("provider") or feature_config.get("model"):
            features[feature_name] = {
                "provider": feature_config.get("provider", "auto"),
                "model": feature_config.get("model", "")
            }
    if features:
        json_config["features"] = features
    
    # Toolsets
    if "toolsets" in yaml_config:
        json_config["toolsets"] = yaml_config["toolsets"]
    
    # Terminal config
    terminal = yaml_config.get("terminal", {})
    if terminal:
        json_config["terminal"] = {
            "backend": terminal.get("backend", "local"),
            "timeout": terminal.get("timeout", 180),
            "docker_image": terminal.get("docker_image", ""),
            "persistent_shell": terminal.get("persistent_shell", True)
        }
    
    # Browser config
    browser = yaml_config.get("browser", {})
    if browser:
        json_config["browser"] = {
            "inactivity_timeout": browser.get("inactivity_timeout", 120),
            "command_timeout": browser.get("command_timeout", 30)
        }
    
    # Display config
    display = yaml_config.get("display", {})
    if display:
        json_config["display"] = {
            "compact": display.get("compact", False),
            "streaming": display.get("streaming", True),
            "show_cost": display.get("show_cost", False),
            "skin": display.get("skin", "default")
        }
    
    # Memory config
    memory = yaml_config.get("memory", {})
    if memory:
        json_config["memory"] = {
            "enabled": memory.get("memory_enabled", True),
            "char_limit": memory.get("memory_char_limit", 2200)
        }
    
    # Security config
    security = yaml_config.get("security", {})
    if security:
        json_config["security"] = {
            "redact_secrets": security.get("redact_secrets", True),
            "tirith_enabled": security.get("tirith_enabled", True)
        }
    
    # Platform configs (feishu, telegram, discord, etc.)
    platforms = {}
    platform_keys = ["feishu", "telegram", "discord", "slack", "whatsapp", "signal", "qqbot"]
    for platform in platform_keys:
        if platform in yaml_config:
            platforms[platform] = yaml_config[platform]
    if platforms:
        json_config["platforms"] = platforms
    
    return json_config


def mask_api_key(api_key: str) -> str:
    """Mask API key for display (show first and last few chars)."""
    if not api_key or len(api_key) < 10:
        return api_key
    return f"{api_key[:6]}...{api_key[-4:]}"


# =============================================================================
# Unified Config Loader (auto-detect format)
# =============================================================================

def load_config_unified() -> Dict[str, Any]:
    """
    Load configuration with auto-detection of format.
    
    Priority:
    1. config.json (new format)
    2. config.yaml (legacy format)
    3. DEFAULT_CONFIG (fallback)
    
    Returns:
        Configuration dictionary
    """
    json_path = get_config_json_path()
    yaml_path = get_config_path()
    
    # Try JSON first
    if json_path.exists():
        try:
            return load_config_json(json_path)
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}")
    
    # Fall back to YAML
    if yaml_path.exists():
        try:
            from hermes_cli.config import load_config as load_yaml_config
            return load_yaml_config()
        except Exception as e:
            print(f"Warning: Failed to load config.yaml: {e}")
    
    # Last resort: default config
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_config_show():
    """Show current configuration."""
    config = load_config_unified()
    print(json.dumps(config, indent=2, ensure_ascii=False))


def cmd_config_migrate(dry_run: bool = False):
    """Migrate config.yaml to config.json."""
    result = migrate_yaml_to_json(dry_run=dry_run)
    
    if result["success"]:
        if dry_run:
            print("Migration preview (dry run):")
            print(json.dumps(result["config"], indent=2, ensure_ascii=False))
        else:
            print("✓ Configuration migrated successfully to config.json")
    else:
        print("Migration failed:")
        for error in result["errors"]:
            print(f"  ✗ {error}")
    
    for warning in result["warnings"]:
        print(f"  ⚠ {warning}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "show":
            cmd_config_show()
        elif command == "migrate":
            dry_run = "--dry-run" in sys.argv
            cmd_config_migrate(dry_run=dry_run)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python config_json.py [show| migrate [--dry-run]]")
            sys.exit(1)
    else:
        print("Hermes JSON Config Loader")
        print("Usage: python config_json.py [show| migrate [--dry-run]]")
