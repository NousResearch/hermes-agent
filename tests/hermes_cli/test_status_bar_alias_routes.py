"""Status-bar aliases obey the same routes and precedence as /model."""

import pytest
import yaml

import cli
from hermes_cli.model_switch import _load_direct_aliases


@pytest.fixture
def alias_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_REVERSE_ALIAS_CACHE", None)

    def write(config):
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        return _load_direct_aliases()

    return write


@pytest.mark.parametrize("nested", [False, True], ids=["top-level", "nested"])
def test_missing_provider_alias_is_displayed_only_on_its_route(alias_config, nested):
    config = {"model": {"provider": "anthropic"}}
    entry = {"model": "audit-model"}
    if nested:
        config["model"]["aliases"] = {"fast": entry}
    else:
        config["model_aliases"] = {"fast": entry}
    route = alias_config(config)["fast"]
    other_provider = "custom" if nested else "anthropic"

    assert cli._reverse_alias_for_display(route.model, route.provider) == "fast"
    assert cli._reverse_alias_for_display(route.model, other_provider) == route.model


@pytest.mark.parametrize("nested_name", ["fast", " FAST "], ids=["exact", "normalized"])
def test_shadowed_alias_is_never_displayed_for_the_losing_route(alias_config, nested_name):
    routes = alias_config({
        "model_aliases": {"fast": {"model": "winning-model", "provider": "anthropic"}},
        "model": {"aliases": {nested_name: "local/losing-model"}},
    })

    assert routes["fast"].model == "winning-model"
    assert cli._reverse_alias_for_display("winning-model", "anthropic") == "fast"
    assert cli._reverse_alias_for_display("losing-model", "local") == "losing-model"
