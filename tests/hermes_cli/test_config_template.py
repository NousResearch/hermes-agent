"""Regression tests for the install-time config template."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_config_template_leaves_model_selection_to_setup():
    """Fresh installs copy this template directly into config.yaml."""
    template_path = REPO_ROOT / "cli-config.yaml.example"
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))

    model_config = config["model"]
    default_model = model_config.get("default") or model_config.get("model") or ""
    assert default_model == ""
