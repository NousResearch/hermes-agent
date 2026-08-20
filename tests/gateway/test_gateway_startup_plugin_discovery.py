"""Gateway startup must (re)discover plugins even when a stale manager exists.

The gateway boots in a process where something earlier (a CLI entry point, an
import-time side effect, a probe that ran before ``HERMES_HOME`` plugins were
readable) may already have created the global ``PluginManager`` and marked it
``_discovered=True`` with an empty registry. ``discover_plugins()`` is
idempotent, so in that state the gateway's startup discovery is a no-op and the
user's plugins never load: ``pre_gateway_dispatch`` hooks never fire and
plugin slash commands come back as unknown commands.

Startup discovery therefore has to force a rescan.
"""

import textwrap
from pathlib import Path

import yaml


def _write_user_plugin(hermes_home: Path, name: str) -> None:
    """Create an enabled user plugin under ``<hermes_home>/plugins/<name>/``.

    The plugin registers one ``pre_gateway_dispatch`` hook and one ``/collab``
    slash command — the two registries the gateway reads at dispatch time.
    """
    plugin_dir = hermes_home / "plugins" / name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump(
            {"name": name, "version": "0.1.0", "description": "startup discovery probe"}
        )
    )
    (plugin_dir / "__init__.py").write_text(
        textwrap.dedent(
            """
            def _pre_gateway_dispatch(**kwargs):
                return None


            def _collab(raw_args=""):
                return "collab ok"


            def register(ctx):
                ctx.register_hook("pre_gateway_dispatch", _pre_gateway_dispatch)
                ctx.register_command("collab", _collab, description="probe command")
            """
        ).lstrip()
    )

    # Plugins are opt-in: enable it in <HERMES_HOME>/config.yaml.
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": [name]}})
    )


def test_gateway_startup_discovery_loads_plugins_despite_stale_manager(
    tmp_path, monkeypatch
):
    import hermes_cli.plugins as plugins_mod

    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir(parents=True, exist_ok=True)
    _write_user_plugin(hermes_home, "startup_probe_plugin")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)

    # Seed the exact pre-boot state that breaks plugin loading: a global
    # manager that already believes discovery is done, with nothing loaded.
    stale = plugins_mod.PluginManager()
    stale._discovered = True
    monkeypatch.setattr(plugins_mod, "_plugin_manager", stale)
    assert not stale._hooks.get("pre_gateway_dispatch")
    assert "collab" not in stale._plugin_commands

    from gateway import run as gateway_run

    helper = getattr(gateway_run, "_discover_plugins_at_startup", None)
    if helper is None:
        # Pre-fix startup path: a plain idempotent discover_plugins() call
        # inlined in GatewayRunner.start(). Exercising it here keeps the
        # failure behavioral (hook/command missing) rather than a bare
        # AttributeError, and keeps the test honest if the helper is ever
        # dropped again.
        helper = plugins_mod.discover_plugins

    helper()

    manager = plugins_mod.get_plugin_manager()
    assert manager is stale, "startup discovery must populate the global manager"
    assert "startup_probe_plugin" in manager._plugins
    assert manager._hooks.get("pre_gateway_dispatch"), (
        "gateway startup discovery did not load the plugin's "
        "pre_gateway_dispatch hook"
    )
    assert "collab" in manager._plugin_commands, (
        "gateway startup discovery did not load the plugin's /collab command"
    )


def test_gateway_start_discovers_plugins_before_registering_adapters():
    """``GatewayRunner.start()`` must call discovery before any adapter loads.

    The behavioral test above proves the helper works; this one pins where it
    is wired in. Plugins register ``pre_gateway_dispatch`` hooks and slash
    commands that adapters consult from their first inbound message, so
    discovery has to happen before the first adapter is registered. A refactor
    that keeps the call but moves it below adapter registration reopens the
    race without failing any behavioral test.
    """
    import inspect

    from gateway.run import GatewayRunner

    source = inspect.getsource(GatewayRunner.start)

    discovery_at = source.find("_discover_plugins_at_startup()")
    assert discovery_at != -1, (
        "GatewayRunner.start() no longer calls _discover_plugins_at_startup(); "
        "gateway startup will not force a plugin rescan"
    )

    marker = "# Register the generic relay adapter"
    marker_at = source.find(marker)
    assert marker_at != -1, (
        f"anchor comment {marker!r} is gone from GatewayRunner.start(); update "
        "this test to the new first-adapter-registration marker"
    )

    assert discovery_at < marker_at, (
        "_discover_plugins_at_startup() must run before the first adapter is "
        "registered, otherwise adapters can dispatch before plugin hooks and "
        "slash commands exist"
    )
