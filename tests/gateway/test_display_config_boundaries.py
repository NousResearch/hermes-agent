"""Boundary tests for values flowing from user YAML into gateway display settings."""

import pytest
import yaml


@pytest.mark.parametrize(
    ("setting", "configured", "expected"),
    [
        ("tool_progress_grouping", " separate ", "separate"),
        ("reasoning_style", " blockquote ", "blockquote"),
    ],
)
@pytest.mark.parametrize("platform_scoped", [False, True])
def test_choice_settings_ignore_surrounding_yaml_whitespace(
    tmp_path, setting, configured, expected, platform_scoped
):
    """Equivalent global and platform values retain their meaning after YAML loading."""
    from gateway.display_config import resolve_display_setting
    from hermes_cli.config_effective import load_user_config_effective

    if platform_scoped:
        text = f'display:\n  platforms:\n    telegram:\n      {setting}: "{configured}"\n'
    else:
        text = f'display:\n  {setting}: "{configured}"\n'

    config_path = tmp_path / "config.yaml"
    config_path.write_text(text, encoding="utf-8")
    user_config = load_user_config_effective(config_path)

    assert user_config == yaml.safe_load(text)
    assert resolve_display_setting(user_config, "telegram", setting) == expected
