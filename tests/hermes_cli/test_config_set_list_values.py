"""``hermes config set`` must parse list/mapping literals, not store them as strings.

Before this fix, ``hermes config set platform_toolsets.discord '["file","web"]'``
stored the value as a raw STRING. Every reader that gates on
``isinstance(..., list)`` — ``_get_platform_tools``, ``_get_enabled_set``,
``_get_disabled_set`` — then silently ignored it and fell back to its default,
so the setting looked saved but never took effect (observed in the wild as a
platform running on the wrong toolset bundle for weeks).
"""
import pytest


@pytest.fixture
def user_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    return home


def test_list_literal_is_parsed_to_list(user_home):
    from hermes_cli.config import set_config_value, read_raw_config

    set_config_value("platform_toolsets.line", '["clarify", "file", "web"]')
    raw = read_raw_config()
    assert raw["platform_toolsets"]["line"] == ["clarify", "file", "web"]


def test_mapping_literal_is_parsed_to_dict(user_home):
    from hermes_cli.config import set_config_value, read_raw_config

    set_config_value("display.tool_progress_overrides", '{"terminal": "off"}')
    raw = read_raw_config()
    assert raw["display"]["tool_progress_overrides"] == {"terminal": "off"}


def test_yaml_flow_list_is_parsed(user_home):
    from hermes_cli.config import set_config_value, read_raw_config

    set_config_value("plugins.enabled", "[model-providers/gemini]")
    raw = read_raw_config()
    assert raw["plugins"]["enabled"] == ["model-providers/gemini"]


def test_invalid_list_literal_warns_and_stores_string(user_home, capsys):
    from hermes_cli.config import set_config_value, read_raw_config

    set_config_value("platform_toolsets.line", '["unclosed')
    captured = capsys.readouterr()
    assert "not valid" in captured.err.lower() or "warning" in captured.err.lower()
    raw = read_raw_config()
    assert raw["platform_toolsets"]["line"] == '["unclosed'


def test_scalar_values_unaffected(user_home):
    from hermes_cli.config import set_config_value, read_raw_config

    set_config_value("agent.max_turns", "300")
    set_config_value("display.compact", "true")
    set_config_value("tts.provider", "edge")
    raw = read_raw_config()
    assert raw["agent"]["max_turns"] == 300
    assert raw["display"]["compact"] is True
    assert raw["tts"]["provider"] == "edge"
