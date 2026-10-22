"""Regression coverage for first writes of dotted model IDs in reasoning overrides."""

import os
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

from hermes_cli.config import _get_nested, _set_nested, _unset_nested, set_config_value


@pytest.mark.parametrize(
    "initial_overrides",
    [
        {"gpt-6-luna": "medium"},
        {"glm-5": "medium"},
        {"glm-5": {"3-flash": "medium"}},
    ],
)
def test_first_write_of_dotted_model_id_preserves_prefix_siblings(initial_overrides):
    config = {"agent": {"reasoning_overrides": dict(initial_overrides)}}
    key = "agent.reasoning_overrides.glm-5.3-flash"

    _set_nested(config, key, "high")

    mapping = config["agent"]["reasoning_overrides"]
    assert mapping == {**initial_overrides, "glm-5.3-flash": "high"}
    assert _get_nested(config, key) == "high"
    assert _unset_nested(config, key) is True
    assert mapping == initial_overrides


def test_config_set_writes_new_dotted_override_through_real_cli_path(tmp_path):
    (tmp_path / ".env").touch()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"agent": {"reasoning_overrides": {"glm-5": "medium"}}}))

    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        set_config_value("agent.reasoning_overrides.glm-5.3-flash", "high")

    saved = yaml.safe_load(path.read_text())
    assert saved["agent"]["reasoning_overrides"] == {
        "glm-5": "medium",
        "glm-5.3-flash": "high",
    }
