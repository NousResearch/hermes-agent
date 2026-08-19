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


# ---------------------------------------------------------------------------
# Consolidated-cluster additions: multi-line YAML blocks, string-typed-key
# guard, conservative trigger, and load_config round-trip.
# ---------------------------------------------------------------------------


def test_multiline_yaml_list_is_parsed(user_home):
    """A multi-line YAML block list must be stored as a real list."""
    from hermes_cli.config import set_config_value, read_raw_config

    set_config_value(
        "custom_providers",
        "- name: foo\n  base_url: https://foo.example/v1\n"
        "- name: bar\n  base_url: https://bar.example/v1",
    )
    raw = read_raw_config()
    assert raw["custom_providers"] == [
        {"name": "foo", "base_url": "https://foo.example/v1"},
        {"name": "bar", "base_url": "https://bar.example/v1"},
    ]


def test_multiline_yaml_mapping_is_parsed(user_home):
    from hermes_cli.config import set_config_value, read_raw_config

    set_config_value(
        "display.tool_progress_overrides",
        "terminal: off\nbrowser: on",
    )
    raw = read_raw_config()
    assert raw["display"]["tool_progress_overrides"] == {
        "terminal": False,
        "browser": True,
    }


def test_string_typed_key_bracket_value_stays_string(user_home):
    """Keys whose DEFAULT_CONFIG type is str must never be coerced —
    even when the value looks like a list literal."""
    from hermes_cli.config import set_config_value, read_raw_config

    set_config_value("approvals.mode", "[off]")
    raw = read_raw_config()
    assert raw["approvals"]["mode"] == "[off]"
    assert isinstance(raw["approvals"]["mode"], str)


def test_string_typed_key_negative_number_stays_string(user_home):
    """'-5' for a string-typed key must remain the string '-5'."""
    from hermes_cli.config import set_config_value, read_raw_config

    set_config_value("approvals.mode", "-5")
    raw = read_raw_config()
    assert raw["approvals"]["mode"] == "-5"


def test_dash_prefixed_scalar_not_treated_as_list(user_home):
    """Single-line dash-prefixed scalars ('-5', '--flag') must stay strings
    for non-string-typed keys too — the over-broad leading '-' trigger from
    #88066 is deliberately avoided."""
    from hermes_cli.config import set_config_value, read_raw_config

    set_config_value("weird.flag", "--verbose")
    raw = read_raw_config()
    assert raw["weird"]["flag"] == "--verbose"


def test_plain_scalar_that_parses_to_scalar_kept_as_string(user_home):
    """If yaml.safe_load of a structured-looking value yields a plain scalar,
    keep the original string."""
    from hermes_cli.config import set_config_value, read_raw_config

    # '{}' parses to an empty dict — that IS structured, so check a value
    # that starts with '[' but parses to a scalar is impossible in YAML;
    # instead use a multi-line value whose lines don't match list/dict shape.
    set_config_value("some.note", "line one\nline two without yaml shape")
    raw = read_raw_config()
    assert raw["some"]["note"] == "line one\nline two without yaml shape"


def test_round_trip_through_load_config(user_home):
    """Structured values written by set_config_value must survive
    load_config as real lists/dicts."""
    from hermes_cli.config import set_config_value, load_config

    set_config_value("platform_toolsets.line", '["clarify", "file", "web"]')
    cfg = load_config()
    assert cfg["platform_toolsets"]["line"] == ["clarify", "file", "web"]
