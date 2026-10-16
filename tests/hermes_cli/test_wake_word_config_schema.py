"""The config UI offers the same wake providers the runtime resolves."""

from hermes_cli.config import load_config
from hermes_cli.web_server_config import _schema_with_dynamic_provider_options
from tools.wake_word import _PROVIDERS, _provider


def test_wake_provider_schema_accepts_default_and_resolves_all_options():
    cfg = load_config()["wake_word"]
    field = _schema_with_dynamic_provider_options()["wake_word.provider"]
    assert field["type"] == "select"
    assert cfg["provider"] in field["options"]
    assert "auto" in field["options"]
    assert {_PROVIDERS[_provider({"provider": choice})][1] for choice in field["options"]} == {
        entry[1] for entry in _PROVIDERS.values()
    }
