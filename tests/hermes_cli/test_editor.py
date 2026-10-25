"""Editor-command resolution for the per-profile Desktop setting."""

from hermes_cli.editor import resolve_editor_command


def test_environment_editor_overrides_profile_setting():
    assert resolve_editor_command(
        config={"editor": {"command": "code --wait"}}, environ={"EDITOR": "vim"}
    ) == "vim"


def test_profile_editor_is_used_before_platform_fallback():
    assert resolve_editor_command(config={"editor": {"command": "  hx  "}}, environ={}) == "hx"


def test_profile_editor_is_exposed_by_the_desktop_config_schema():
    """The schema-driven Desktop settings page can edit the persistent command."""
    from hermes_cli.web_server_config import CONFIG_SCHEMA

    assert CONFIG_SCHEMA["editor.command"]["type"] == "string"
