"""Configuration contract for max-iteration auto-continuation."""

import os
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    ("raw", "enabled", "maximum"),
    [
        ({}, False, 0),
        ({"agent": {}}, False, 0),
        (
            {
                "agent": {
                    "auto_continue_on_max_iterations": {
                        "enabled": False,
                        "max_auto_continues": 3,
                    }
                }
            },
            False,
            3,
        ),
        (
            {
                "agent": {
                    "auto_continue_on_max_iterations": {
                        "enabled": True,
                        "max_auto_continues": "2",
                    }
                }
            },
            True,
            2,
        ),
        (
            {
                "agent": {
                    "auto_continue_on_max_iterations": {
                        "enabled": True,
                        "max_auto_continues": 0,
                    }
                }
            },
            True,
            0,
        ),
        (
            {
                "agent": {
                    "auto_continue_on_max_iterations": {
                        "enabled": True,
                        "max_auto_continues": -4,
                    }
                }
            },
            True,
            0,
        ),
        (
            {"agent": {"auto_continue_on_max_iterations": True}},
            True,
            0,
        ),
    ],
)
def test_config_normalization_is_fail_closed_and_bounded(raw, enabled, maximum):
    from agent.auto_continue import resolve_auto_continue_config

    config = resolve_auto_continue_config(raw)

    assert config.enabled is enabled
    assert config.max_auto_continues == maximum
    assert config.can_continue(used=maximum - 1) is (enabled and maximum > 0)
    assert config.can_continue(used=maximum) is False


def test_default_config_is_disabled_and_dashboard_schema_exposes_nested_fields():
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.web_server import CONFIG_SCHEMA

    config = DEFAULT_CONFIG["agent"]["auto_continue_on_max_iterations"]
    assert config["enabled"] is False
    assert config["max_auto_continues"] == 0
    assert config["prompt"] == ""

    assert CONFIG_SCHEMA["agent.auto_continue_on_max_iterations.enabled"]["type"] == "boolean"
    assert CONFIG_SCHEMA["agent.auto_continue_on_max_iterations.max_auto_continues"]["type"] == "number"
    assert CONFIG_SCHEMA["agent.auto_continue_on_max_iterations.prompt"]["type"] == "string"


def test_policy_loads_from_real_profile_config(tmp_path):
    from agent.auto_continue import load_auto_continue_config

    (tmp_path / "config.yaml").write_text(
        """
agent:
  auto_continue_on_max_iterations:
    enabled: true
    max_auto_continues: 2
    prompt: Continue from persisted state.
""".lstrip(),
        encoding="utf-8",
    )

    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        policy = load_auto_continue_config()

    assert policy.enabled is True
    assert policy.max_auto_continues == 2
    assert policy.prompt == "Continue from persisted state."


def test_policy_does_not_hide_live_config_parse_errors():
    from agent.auto_continue import load_auto_continue_config

    with (
        patch("hermes_cli.config.load_config", side_effect=RuntimeError("invalid YAML")),
        pytest.raises(RuntimeError, match="invalid YAML"),
    ):
        load_auto_continue_config()
