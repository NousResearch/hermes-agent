"""Regression coverage for additive Kanban configuration defaults."""

import os
from unittest.mock import patch

from hermes_cli.config import load_config


def test_auto_decompose_excluded_profiles_defaults_to_empty(tmp_path):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        config = load_config()

    assert config["kanban"]["auto_decompose_excluded_profiles"] == []
