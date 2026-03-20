"""Product-layer configuration for the hermes-core distribution.

This file is intentionally separate from ``config.yaml``.  Hermes' existing
config remains the source of truth for generic Hermes behavior, while
``product.yaml`` holds setup-owned deployment behavior for the local multi-user
product layer.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

from hermes_cli.config import ensure_hermes_home, get_hermes_home, _secure_dir, _secure_file


DEFAULT_PRODUCT_CONFIG: Dict[str, Any] = {
    "product": {
        "brand": {
            "name": "Hermes Core",
            "logo_path": "",
        },
        "agent": {
            "soul_template_path": "",
        },
    },
    "auth": {
        "provider": "kanidm",
        "mode": "passkey",
        "issuer_url": "",
        "client_id": "hermes-core",
        "client_secret_ref": "HERMES_PRODUCT_OIDC_CLIENT_SECRET",
    },
    "network": {
        "bind_host": "0.0.0.0",
        "public_host": "localhost",
        "app_port": 8086,
        "kanidm_port": 8443,
        "tailscale": {
            "enabled": False,
        },
    },
    "models": {
        "default_route": {
            "provider": "custom",
            "base_url": "http://127.0.0.1:8080/v1",
            "model": "qwen3.5-9b-local",
        },
    },
    "tools": {
        "enabled_profiles": ["tier1"],
        "selectable_placements": {},
    },
    "runtime": {
        "default_profile": "tier1",
        "default_toolset": "mynah-tier1",
    },
    "storage": {
        "root": "product",
        "users_root": "product/users",
    },
    "bootstrap": {
        "first_admin_username": "admin",
        "first_admin_display_name": "Administrator",
        "first_admin_reset_ttl_seconds": 86400,
    },
    "services": {
        "kanidm": {
            "mode": "docker",
            "container_name": "hermes-kanidm",
        },
    },
}


def get_product_config_path() -> Path:
    return get_hermes_home() / "product.yaml"


def get_product_storage_root(home: Path | None = None) -> Path:
    hermes_home = home or get_hermes_home()
    relative = DEFAULT_PRODUCT_CONFIG["storage"]["root"]
    return hermes_home / relative


def get_product_users_root(home: Path | None = None) -> Path:
    hermes_home = home or get_hermes_home()
    relative = DEFAULT_PRODUCT_CONFIG["storage"]["users_root"]
    return hermes_home / relative


def ensure_product_home() -> None:
    ensure_hermes_home()
    hermes_home = get_hermes_home()
    product_root = get_product_storage_root(hermes_home)
    users_root = get_product_users_root(hermes_home)
    for path in (
        product_root,
        users_root,
        product_root / "logs",
        product_root / "services",
        product_root / "bootstrap",
    ):
        path.mkdir(parents=True, exist_ok=True)
        _secure_dir(path)


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_product_config() -> Dict[str, Any]:
    ensure_product_home()
    config_path = get_product_config_path()
    config = copy.deepcopy(DEFAULT_PRODUCT_CONFIG)
    if config_path.exists():
        with open(config_path, encoding="utf-8") as handle:
            user_config = yaml.safe_load(handle) or {}
        if isinstance(user_config, dict):
            config = _deep_merge(config, user_config)
    return config


def save_product_config(config: Dict[str, Any]) -> None:
    from utils import atomic_yaml_write

    ensure_product_home()
    config_path = get_product_config_path()
    normalized = _deep_merge(DEFAULT_PRODUCT_CONFIG, config)
    atomic_yaml_write(config_path, normalized)
    _secure_file(config_path)


def initialize_product_config_file() -> Dict[str, Any]:
    config = load_product_config()
    config_path = get_product_config_path()
    if not config_path.exists():
        save_product_config(config)
    return config


def resolve_runtime_defaults(config: Dict[str, Any] | None = None) -> Dict[str, str]:
    product_config = config or load_product_config()
    runtime_cfg = product_config.get("runtime", {})
    model_cfg = product_config.get("models", {}).get("default_route", {})
    return {
        "runtime_profile": str(runtime_cfg.get("default_profile", "tier1")),
        "runtime_toolset": str(runtime_cfg.get("default_toolset", "mynah-tier1")),
        "inference_model": str(model_cfg.get("model", "qwen3.5-9b-local")),
    }
