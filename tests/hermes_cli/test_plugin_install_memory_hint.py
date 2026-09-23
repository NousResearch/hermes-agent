"""Installation must identify the memory provider's actual activation switch."""
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def install_fixture(tmp_path, monkeypatch):
    from rich.console import Console
    from hermes_cli import plugins_cmd as pc, plugins_cmd_catalog as catalog
    from hermes_cli import plugins_activation
    from hermes_cli.config import save_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    save_config({"memory": {"provider": "existing"}, "plugins": {"enabled": ["other"]}})
    target = tmp_path / "plugins" / "fixture-memory"
    target.mkdir(parents=True)
    manifest = {"name": "fixture-memory", "category": "memory", "version": "1.0.0"}
    (target / "plugin.yaml").write_text("name: fixture-memory\ncategory: memory\nversion: 1.0.0\n", encoding="utf-8")
    output = StringIO()
    monkeypatch.setattr(pc, "_console", lambda: Console(file=output, width=200, color_system=None))
    monkeypatch.setattr(catalog, "raise_if_removed", lambda *args: None)
    installer = Mock(side_effect=lambda *args, **kwargs: (target, manifest, "fixture-memory"))
    monkeypatch.setattr(pc, "_install_plugin_core", installer)
    monkeypatch.setattr(catalog, "install_catalog_entry", installer)
    monkeypatch.setattr(pc, "_prompt_plugin_env_vars", lambda *args: None)
    monkeypatch.setattr(pc, "_install_python_dependencies", lambda *args, **kwargs: None)
    monkeypatch.setattr(pc, "_display_after_install", lambda *args: None)
    monkeypatch.setattr(pc, "_is_tty", lambda: True)
    asked = Mock(return_value=True)
    monkeypatch.setattr(pc, "_ask_yes", asked)
    activation = Mock(return_value={})
    monkeypatch.setattr(plugins_activation, "activate_plugin_now", activation)
    monkeypatch.setattr(plugins_activation, "activation_hint", lambda value: "generic activation")
    return SimpleNamespace(pc=pc, catalog=catalog, target=target, manifest=manifest,
                           output=output, asked=asked, activation=activation,
                           config=tmp_path / "config.yaml", installer=installer)


@pytest.mark.parametrize("enable", [False, True, None])
@pytest.mark.parametrize("source", ["manifest", "catalog"])
def test_memory_install_does_not_claim_generic_activation(install_fixture, monkeypatch, enable, source):
    f = install_fixture
    identifier = "fixture/plugin"
    if source == "catalog":
        f.manifest.pop("category")
        entry = SimpleNamespace(name="fixture-memory", category="memory", tier="community",
                                sha="a" * 40, install_identifier=identifier)
        monkeypatch.setattr(f.catalog, "resolve_catalog_name", lambda *args: entry)
        monkeypatch.setattr(f.catalog, "entry_capability_summary", lambda entry: "")
        identifier = "fixture-memory"
    before = f.config.read_bytes()

    f.pc.cmd_install(identifier, enable=enable, no_deps=True)

    assert "hermes memory setup" in f.output.getvalue()
    assert "memory.provider" in f.output.getvalue()
    assert "hermes plugins enable fixture-memory" not in f.output.getvalue()
    assert f.config.read_bytes() == before
    f.asked.assert_not_called()
    f.activation.assert_not_called()


@pytest.mark.parametrize("enable", [False, True, None])
def test_ordinary_plugin_keeps_enable_flow(install_fixture, enable):
    from hermes_cli.config import load_config

    f = install_fixture
    f.manifest.pop("category")
    f.pc.cmd_install("fixture/plugin", enable=enable, no_deps=True)

    config = load_config()
    assert config["memory"]["provider"] == "existing"
    assert "other" in config["plugins"]["enabled"]
    assert ("fixture-memory" in config["plugins"]["enabled"]) == (enable is not False)
    assert "hermes memory setup" not in f.output.getvalue()
    assert f.asked.call_count == (1 if enable is None else 0)
    assert f.activation.call_count == (0 if enable is False else 1)


def test_failed_install_does_not_print_activation_guidance(install_fixture):
    f = install_fixture
    before = f.config.read_bytes()
    f.installer.side_effect = f.pc.PluginOperationError("fixture install failed")
    with pytest.raises(SystemExit) as error:
        f.pc.cmd_install("fixture/plugin", enable=True, no_deps=True)
    assert error.value.code == 1
    assert "hermes memory setup" not in f.output.getvalue()
    assert f.config.read_bytes() == before
    f.activation.assert_not_called()
