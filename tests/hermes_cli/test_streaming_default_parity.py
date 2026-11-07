"""The shared display default agrees with the classic CLI loader."""

from hermes_cli.cli_config_load import _cli_config_defaults
from hermes_cli.config_defaults import DEFAULT_CONFIG


def test_display_streaming_default_matches_cli():
    cli_default = _cli_config_defaults()["display"]["streaming"]

    assert DEFAULT_CONFIG["display"]["streaming"] == cli_default
    assert cli_default is True
