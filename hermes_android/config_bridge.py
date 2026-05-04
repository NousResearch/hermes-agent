from __future__ import annotations

from copy import deepcopy
from typing import Any

from hermes_android.python_path import prefer_hermes_package_root

prefer_hermes_package_root()

from hermes_cli.config import load_config, save_config


def read_runtime_config() -> dict[str, Any]:
    return deepcopy(load_config())


def write_runtime_config(
    provider: str,
    model: str,
    base_url: str = "",
) -> dict[str, Any]:
    config = load_config()
    config["provider"] = provider
    config["model"] = model
    config["base_url"] = base_url
    save_config(config)
    return deepcopy(load_config())
