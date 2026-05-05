"""Tests for anticipation configuration defaults and merging."""

from unittest.mock import patch

from hermes_cli.config import DEFAULT_CONFIG, load_config


def test_default_config_has_anticipation_disabled():
    anticipation = DEFAULT_CONFIG["anticipation"]

    assert anticipation["enabled"] is False
    assert anticipation["default_permission"] == "suggest"
    assert anticipation["loops"]["stale_task_resurfacer"]["enabled"] is False
    assert anticipation["loops"]["router_monitor"]["enabled"] is False
    assert anticipation["loops"]["router_monitor"]["permission"] == "ask_to_execute"


def test_load_config_adds_anticipation_defaults_for_existing_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: test/model\n", encoding="utf-8")

    with patch.dict("os.environ", {"HERMES_HOME": str(tmp_path)}):
        config = load_config()

    assert config["model"] == "test/model"
    assert config["anticipation"]["enabled"] is False
    assert config["anticipation"]["notification_budget"]["max_per_day"] == 3
    assert config["anticipation"]["loops"]["stale_task_resurfacer"]["lookback_days"] == 14
    assert config["anticipation"]["loops"]["router_monitor"]["enabled"] is False
    assert config["anticipation"]["loops"]["router_monitor"]["schedule"] == "manual"


def test_load_config_preserves_user_anticipation_overrides(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
anticipation:
  enabled: true
  notification_budget:
    max_per_day: 1
  loops:
    stale_task_resurfacer:
      enabled: true
      min_confidence: 0.9
""".strip(),
        encoding="utf-8",
    )

    with patch.dict("os.environ", {"HERMES_HOME": str(tmp_path)}):
        config = load_config()

    assert config["anticipation"]["enabled"] is True
    assert config["anticipation"]["notification_budget"]["max_per_day"] == 1
    assert config["anticipation"]["notification_budget"]["min_minutes_between"] == 120
    loop = config["anticipation"]["loops"]["stale_task_resurfacer"]
    assert loop["enabled"] is True
    assert loop["min_confidence"] == 0.9
    assert loop["permission"] == "suggest"
