from hermes_cli.config import DEFAULT_CONFIG


def test_desktop_statusbar_defaults_on_for_upgrade_compatibility():
    assert DEFAULT_CONFIG["display"]["desktop_statusbar"] == "on"
