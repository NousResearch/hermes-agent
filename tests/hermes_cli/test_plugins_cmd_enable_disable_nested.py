"""Canonical keys and privilege consent through real admission."""
import shutil

import pytest
import hermes_yaml as yaml

from tests.hermes_cli.plugin_worker_support import (
    plugin_world as plugin_world,
    isolated_python as isolated_python,
)


@pytest.mark.parametrize("query", ["trace-leaf", "trace-manifest", "observability/trace-leaf"])
def test_nested_enable_disable_and_composite_use_canonical_key(plugin_world, monkeypatch, query):
    from hermes_cli import plugins_cmd
    from rich.console import Console

    world = plugin_world
    origin, _ = world.origin(name="trace-manifest")
    key = "observability/trace-leaf"
    target = world.home / "plugins" / key
    shutil.copytree(origin, target)
    (world.home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": [], "disabled": [key, "trace-leaf", "trace-manifest"]}}), encoding="utf-8")
    world.command("enable", name=query, no_allow_tool_override=True)
    assert world.enabled() == [key]
    config = yaml.safe_load((world.home / "config.yaml").read_text())
    assert config["plugins"]["disabled"] == []
    world.imports(key)
    world.command("disable", name=query)
    assert world.enabled() == []
    assert yaml.safe_load((world.home / "config.yaml").read_text())["plugins"]["disabled"] == [key]
    # The fallback menu must persist canonical keys too, not labels.
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    plugins_cmd._run_composite_fallback([key], ["trace-manifest"], {0}, {key}, [], Console())
    assert world.enabled() == [key]
    world.imports(key)


def test_ambiguous_and_unknown_names_cannot_change_config(plugin_world):
    from hermes_cli import plugins_cmd

    world = plugin_world
    for category in ("image_gen", "model-providers"):
        directory = world.home / "plugins" / category / "same-leaf"
        directory.mkdir(parents=True)
        (directory / "plugin.yaml").write_text(f"name: {category}-fixture\n", encoding="utf-8")
    before = (world.home / "config.yaml").read_bytes()
    for query in ("same-leaf", "not-installed"):
        with pytest.raises(SystemExit) as exc:
            world.command("enable", name=query)
        assert exc.value.code == 1
    assert plugins_cmd._resolve_plugin_key("image_gen/same-leaf") == "image_gen/same-leaf"
    assert (world.home / "config.yaml").read_bytes() == before


def test_dependency_free_enable_no_churn_and_tool_override_fails_closed(plugin_world, monkeypatch):
    from hermes_cli import plugins_cmd
    from pm import client, receipt

    world = plugin_world
    client.sync_venv(explicit=True)
    selected = world.selected()
    origin, sha = world.origin(surface=None)
    facts = (world.home / "config.yaml").read_bytes()
    world.command("install", identifier=origin.as_uri(), ref=sha, no_enable=True, allow_removed=True)
    assert world.selected() == selected
    assert (world.home / "config.yaml").read_bytes() == facts
    monkeypatch.setattr(plugins_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(plugins_cmd.sys.stdout, "isatty", lambda: True)
    def eof(*args, **kwargs):
        raise EOFError
    monkeypatch.setattr("rich.console.Console.input", eof)
    world.command("enable", name="plugin-worker-proof")
    config = yaml.safe_load((world.home / "config.yaml").read_text())
    assert config["plugins"]["entries"]["plugin-worker-proof"]["allow_tool_override"] is False
    assert world.selected() == selected
    assert receipt.latest()["venv_rebuild"]["ok"] is False

    bundled = world.core / "plugins/trusted-fixture"
    bundled.mkdir(parents=True)
    (bundled / "plugin.yaml").write_text("name: trusted-fixture\n", encoding="utf-8")
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: world.core / "plugins")
    def unexpected(*args, **kwargs):
        pytest.fail("bundled plugin requested privilege consent")
    monkeypatch.setattr("rich.console.Console.input", unexpected)
    world.command("enable", name="trusted-fixture")
    config = yaml.safe_load((world.home / "config.yaml").read_text())
    assert "trusted-fixture" not in config["plugins"]["entries"]
    assert world.selected() == selected