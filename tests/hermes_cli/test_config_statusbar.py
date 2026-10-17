"""The statusbar key is recognized without masking existing legacy configuration."""

def test_statusbar_is_a_recognized_unseeded_display_key():
    """The classic CLI reads it, but seeding a default would shadow the legacy fallback."""
    from hermes_cli.config import _validate_config_key
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert "statusbar" not in DEFAULT_CONFIG["display"]
    assert _validate_config_key("display.statusbar") == (True, None)
    assert _validate_config_key("display.statusbara")[0] is False
