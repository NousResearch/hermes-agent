from hermes_cli.config import DEFAULT_CONFIG
from hermes_cli.web_server import CONFIG_SCHEMA


def test_desktop_composer_enter_sends_defaults_to_true():
    desktop = DEFAULT_CONFIG.get("desktop")
    assert isinstance(desktop, dict)
    composer = desktop.get("composer")
    assert isinstance(composer, dict)
    assert composer.get("enter_sends") is True


def test_desktop_composer_enter_sends_is_in_generated_schema():
    assert CONFIG_SCHEMA["desktop.composer.enter_sends"]["type"] == "boolean"
