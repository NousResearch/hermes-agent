"""``ctx.register_card_action_handler``: validation, one owner per namespace, reload and unload.

Contract: ``gateway/plugin_card_actions.py``. Plugins load through real discovery from a temp
``HERMES_HOME`` (bundled plugins pointed at an empty dir so only the fixtures load).
"""

import logging
from pathlib import Path

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest, get_plugin_manager
from hermes_cli.plugins_actions import card_action_handler


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    plugins_mod._reset_plugin_managers_for_tests()
    yield home
    plugins_mod._reset_plugin_managers_for_tests()


def _write_plugin(home: Path, name: str, *register_lines: str) -> None:
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": name, "version": "0.1.0"}), encoding="utf-8")
    body = "".join(f"    {line}\n" for line in register_lines)
    (plugin_dir / "__init__.py").write_text(f"def register(ctx):\n{body}", encoding="utf-8")


def _enable(home: Path, *names: str) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": list(names)}}), encoding="utf-8")


def _answer(namespace: str):
    """``(owning plugin, its callback's answer)`` as the gateway resolves the namespace, or None."""
    entry = card_action_handler(namespace)
    return None if entry is None else (entry["plugin"], entry["callback"](None))


def _ctx(name: str, manager: PluginManager | None = None) -> PluginContext:
    manifest = PluginManifest(name=name, version="0.1.0", description="test")
    return PluginContext(manifest=manifest, manager=manager or PluginManager())


@pytest.mark.parametrize("namespace", ["", "Brain", "-brain", ".brain", "br ain", "brain/x", "a" * 65, 7, None])
def test_an_invalid_namespace_is_refused(namespace):
    ctx = _ctx("alpha")
    with pytest.raises(ValueError, match="invalid namespace"):
        ctx.register_card_action_handler(namespace, lambda event: "x")
    assert ctx._manager._card_action_handlers == {}


def test_a_non_callable_callback_is_refused():
    ctx = _ctx("alpha")
    with pytest.raises(ValueError, match="non-callable"):
        ctx.register_card_action_handler("brain", "not callable")


@pytest.mark.parametrize("namespace", ["a", "7", "a" * 64, "k3.brain-review_2"])
def test_valid_namespaces_register(namespace):
    ctx = _ctx("alpha")
    assert ctx.register_card_action_handler(namespace, lambda event: "x") is not None
    assert ctx._manager._card_action_handlers[namespace]["plugin"] == "alpha"


def test_a_namespace_has_one_owner_across_plugins(home, caplog):
    _write_plugin(home, "alpha", 'ctx.register_card_action_handler("brain", lambda event: "alpha")')
    _write_plugin(home, "beta", 'ctx.register_card_action_handler("brain", lambda event: "beta")',
                  'ctx.register_card_action_handler("beta-only", lambda event: "beta")')
    _enable(home, "alpha", "beta")

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        owner, answer = _answer("brain")

    refused = ({"alpha", "beta"} - {owner}).pop()
    assert answer == owner
    assert any(refused in r.getMessage() and "'brain'" in r.getMessage() and owner in r.getMessage()
               for r in caplog.records if r.levelno == logging.WARNING)
    # The refused plugin still loads; only the contested namespace is withheld from it.
    assert _answer("beta-only") == ("beta", "beta")


def test_the_owner_registering_again_replaces_its_handler(home):
    _write_plugin(home, "alpha", 'ctx.register_card_action_handler("brain", lambda event: "first")',
                  'ctx.register_card_action_handler("brain", lambda event: "second")')
    _enable(home, "alpha")
    assert _answer("brain") == ("alpha", "second")


def test_a_reload_replaces_the_handler(home):
    _write_plugin(home, "alpha", 'ctx.register_card_action_handler("brain", lambda event: "v1")')
    _enable(home, "alpha")
    assert _answer("brain") == ("alpha", "v1")

    _write_plugin(home, "alpha", 'ctx.register_card_action_handler("brain", lambda event: "version two")')
    get_plugin_manager().discover_and_load(force=True)

    assert _answer("brain") == ("alpha", "version two")


def test_unloading_the_owner_unregisters_and_frees_the_namespace(home):
    _write_plugin(home, "alpha", 'ctx.register_card_action_handler("brain", lambda event: "alpha")')
    _enable(home, "alpha")
    assert _answer("brain") == ("alpha", "alpha")
    manager = get_plugin_manager()

    assert manager.unload("alpha") is True

    assert card_action_handler("brain") is None
    assert _ctx("gamma", manager).register_card_action_handler("brain", lambda event: "gamma") is not None
    assert _answer("brain") == ("gamma", "gamma")


def test_the_activation_summary_reports_card_namespaces_as_live_callbacks(home):
    from hermes_cli.plugins_activation import plugin_activation_summary
    _write_plugin(home, "alpha", 'ctx.register_card_action_handler("brain", lambda event: "alpha")')
    _enable(home, "alpha")
    manager = get_plugin_manager()
    manager.discover_and_load()

    assert "card:brain" in plugin_activation_summary(manager, "alpha")["activated_now"]["callbacks"]


def test_the_plugin_host_refuses_card_action_handlers():
    from hermes_cli.plugin_host_child import RemotePluginContext
    from hermes_cli.plugin_host_wire import PluginHostUnsupported
    remote = RemotePluginContext(None, "alpha", {"ctx_methods": ["register_card_action_handler"]})
    with pytest.raises(PluginHostUnsupported, match="card click"):
        remote.register_card_action_handler("brain", lambda event: "x")
