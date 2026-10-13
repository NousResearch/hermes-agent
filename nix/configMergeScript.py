"""Merge Nix settings into the existing Hermes YAML configuration."""

import json
import os
import stat
import sys
import tempfile
from pathlib import Path


def deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def merge_config(nix_json, config_path, yaml):
    nix_json = Path(nix_json)
    config_path = Path(config_path).resolve()

    with open(nix_json) as file:
        nix = json.load(file)

    existing = {}
    if config_path.exists():
        with open(config_path) as file:
            existing = yaml.load(file) or {}

    merged = deep_merge(existing, nix)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{config_path.name}.", suffix=".tmp", dir=config_path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w") as file:
            yaml.dump(merged, file)
            file.flush()
            os.fsync(file.fileno())
        if config_path.exists():
            os.chmod(temp_path, stat.S_IMODE(config_path.stat().st_mode))
        os.replace(temp_path, config_path)
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe", pure=True)
    # Existing configs use YAML 1.1 booleans such as yes and off.
    yaml.version = (1, 1)
    yaml.default_flow_style = False
    yaml.sort_base_mapping_type_on_output = False

    merge_config(sys.argv[1], Path(sys.argv[2]), yaml)


if __name__ == "__main__":
    main()
