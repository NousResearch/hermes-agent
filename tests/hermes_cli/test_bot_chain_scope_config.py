"""The chain-scope editor choices persist through the CLI into the runtime loader."""

import pytest

from gateway.bot_loop_guard import load_settings
from hermes_cli.config import load_config, set_config_value
from hermes_cli.web_server_config import CONFIG_SCHEMA


@pytest.mark.parametrize("scope", ["channel", "thread"])
def test_editor_scope_options_round_trip_through_config_set(scope):
    key = "gateway.bot_loop_guard.chain_scope"
    field = CONFIG_SCHEMA[key]
    assert field["type"] == "select"
    assert scope in field["options"]
    assert load_config()["gateway"]["bot_loop_guard"]["chain_scope"] in field["options"]
    set_config_value(key, scope)
    assert load_settings().chain_scope == scope
