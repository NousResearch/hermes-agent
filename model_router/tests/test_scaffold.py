from pathlib import Path

import yaml

from validate_router_config import validate_router_config


ROOT = Path(__file__).resolve().parent.parent


def test_readme_exists():
    assert (ROOT / "README.md").exists()


def test_router_config_is_valid_yaml_and_passes_validation():
    config_path = ROOT / "router_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result = validate_router_config(config)
    assert result["valid"] is True
    assert result["errors"] == []


def test_makefile_exists():
    assert (ROOT / "Makefile").exists()
