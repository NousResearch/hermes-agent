"""Regression coverage for plugin toolset validation during CLI/gateway startup."""

from __future__ import annotations

import json
import logging
import sys
import threading
import types
from types import SimpleNamespace


def _validate_and_collect_cli_warnings(toolsets: list[str], config: dict | None = None) -> list[str]:
    """Run the gateway unknown-toolset check and the CLI startup warning printer."""
    from gateway.session_local_inventory import _unknown_toolsets
    from hermes_cli.cli_init_mixin import _adopt_gateway_toolsets

    cfg = config or {}
    unknown = _unknown_toolsets(toolsets, cfg)
    warnings: list[str] = []
    cli = SimpleNamespace(
        enabled_toolsets=list(toolsets),
        _console_print=lambda message, *_args, **_kwargs: warnings.append(str(message)),
    )
    store = SimpleNamespace(current={"info": {"enabled_toolsets": list(toolsets), "unknown_toolsets": unknown}})
    _adopt_gateway_toolsets(cli, store)
    return warnings


def test_discovery_allows_same_thread_register_reentry(monkeypatch):
    """A plugin register callback may transitively ensure discovery."""
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    sweeps = []

    def _reentrant_sweep():
        sweeps.append("started")
        manager.discover_and_load()
        sweeps.append("finished")

    monkeypatch.setattr(manager, "_discover_and_load_inner", _reentrant_sweep)

    manager.discover_and_load()

    assert sweeps == ["started", "finished"]
    assert manager._discovered is True
    assert manager._discovery_owner is None


def test_public_discovery_reentry_does_not_join_a_waiting_thread(monkeypatch):
    """An owner must not join a background caller waiting on that owner."""
    from hermes_cli import plugins as plugins_mod

    manager = plugins_mod.PluginManager()

    class WaitingBackgroundThread:
        @staticmethod
        def is_alive():
            return True

        @staticmethod
        def join(timeout=None):
            raise AssertionError("discovery owner joined its own waiter")

    def _reentrant_sweep():
        monkeypatch.setattr(
            plugins_mod,
            "_background_discovery_thread",
            WaitingBackgroundThread(),
        )
        plugins_mod.discover_plugins()

    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(manager, "_discover_and_load_inner", _reentrant_sweep)

    manager.discover_and_load()

    assert manager._discovered is True


def test_waiter_retries_after_discovery_owner_fails(monkeypatch):
    """A concurrent caller retries after the active sweep fails."""
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    first_sweep_entered = threading.Event()
    allow_first_failure = threading.Event()
    waiter_started = threading.Event()
    calls = []
    errors = []

    def _sweep():
        calls.append(threading.current_thread().name)
        if len(calls) == 1:
            first_sweep_entered.set()
            if not allow_first_failure.wait(timeout=5):
                raise RuntimeError("waiter never reached discovery")
            raise RuntimeError("first sweep failed")

    def _run_owner():
        try:
            manager.discover_and_load()
        except RuntimeError as exc:
            errors.append(str(exc))

    def _run_waiter():
        waiter_started.set()
        manager.discover_and_load()

    monkeypatch.setattr(manager, "_discover_and_load_inner", _sweep)
    owner = threading.Thread(target=_run_owner, name="owner", daemon=True)
    waiter = threading.Thread(target=_run_waiter, name="waiter", daemon=True)
    owner.start()
    assert first_sweep_entered.wait(timeout=5)
    waiter.start()
    assert waiter_started.wait(timeout=5)
    assert calls == ["owner"]

    allow_first_failure.set()
    owner.join(timeout=5)
    waiter.join(timeout=5)

    assert not owner.is_alive()
    assert not waiter.is_alive()
    assert errors == ["first sweep failed"]
    assert calls == ["owner", "waiter"]
    assert manager._discovered is True
    assert manager._discovery_owner is None


def test_untracked_discovery_serves_persisted_toolset_keys_without_waiting(
    tmp_path, monkeypatch
):
    """Direct prewarm keeps config resolution overlapped with discovery."""
    plugin_toolset = "cached_direct_prewarm_tools"
    hermes_home = tmp_path / "hermes-home"
    cache_dir = hermes_home / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "plugin_toolset_keys.json").write_text(
        json.dumps({"toolset_keys": [plugin_toolset], "portable_mcp": []}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import hermes_cli.plugins as plugins_mod

    manager = plugins_mod.PluginManager()
    discovery_entered = threading.Event()
    allow_discovery = threading.Event()
    keys_returned = threading.Event()
    observed = {}

    def paused_discovery():
        discovery_entered.set()
        if not allow_discovery.wait(timeout=5):
            raise RuntimeError("cached toolset lookup blocked plugin discovery")

    monkeypatch.setattr(manager, "_discover_and_load_inner", paused_discovery)
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "_background_discovery_thread", None)

    prewarm = threading.Thread(
        target=plugins_mod.discover_plugins,
        name="tool-prewarm",
        daemon=True,
    )

    def read_cached_keys():
        observed["keys"] = plugins_mod.get_plugin_toolset_keys_nowait()
        keys_returned.set()

    reader = threading.Thread(target=read_cached_keys, daemon=True)
    prewarm.start()
    assert discovery_entered.wait(timeout=5), "prewarm discovery did not start"
    reader.start()

    try:
        assert keys_returned.wait(timeout=2), "cached key lookup joined prewarm"
        assert observed["keys"] == {plugin_toolset}
    finally:
        allow_discovery.set()
        prewarm.join(timeout=5)
        reader.join(timeout=5)


def test_untracked_tool_prewarm_discovery_finishes_before_validation(
    monkeypatch,
):
    """Direct ``cli.py -w`` discovery must have one completion boundary.

    That path imports ``model_tools`` from an untracked ``tool-prewarm``
    thread. Model the resulting direct ``discover_plugins()`` call while a
    plugin is paused in registration, then validate on the main thread.
    """
    plugin_toolset = "direct_prewarm_tools"
    plugin_tool = "direct_prewarm_health"
    unknown_toolset = "genuinely_missing_toolset"

    import hermes_cli.plugins as plugins_mod
    from tools.registry import registry

    manager = plugins_mod.PluginManager()
    discovery_entered = threading.Event()
    validation_discovery_returned = threading.Event()
    registration_complete = threading.Event()
    observed = {}

    def load_prewarmed_plugin():
        discovery_entered.set()
        observed["validation_returned_before_registration"] = (
            validation_discovery_returned.wait(timeout=0.2)
        )
        registry.register(
            name=plugin_tool,
            toolset=plugin_toolset,
            schema={
                "name": plugin_tool,
                "description": "Read-only health check",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda _args, **_kwargs: '{"status":"ok"}',
        )
        manager._plugin_tool_names.add(plugin_tool)
        registration_complete.set()

    monkeypatch.setattr(manager, "_discover_and_load_inner", load_prewarmed_plugin)
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "_background_discovery_thread", None)

    validation_thread_id = threading.get_ident()
    original_discover = plugins_mod.discover_plugins

    def observe_validation_discovery(*args, **kwargs):
        result = original_discover(*args, **kwargs)
        if threading.get_ident() == validation_thread_id:
            validation_discovery_returned.set()
        return result

    monkeypatch.setattr(plugins_mod, "discover_plugins", observe_validation_discovery)

    prewarm = threading.Thread(
        target=plugins_mod.discover_plugins,
        name="tool-prewarm",
        daemon=True,
    )
    prewarm.start()
    assert discovery_entered.wait(timeout=5), "prewarm discovery did not start"

    try:
        warnings = _validate_and_collect_cli_warnings([plugin_toolset, unknown_toolset])

        assert registration_complete.wait(timeout=5)
        assert observed["validation_returned_before_registration"] is False
        unknown_warnings = [msg for msg in warnings if "Unknown toolsets" in msg]
        assert len(unknown_warnings) == 1
        assert unknown_toolset in unknown_warnings[0]
        assert plugin_toolset not in unknown_warnings[0]
    finally:
        validation_discovery_returned.set()
        prewarm.join(timeout=5)
        registry.deregister(plugin_tool)


def test_completed_discovery_skips_redundant_join(monkeypatch):
    """A completed background discovery must not force a redundant join."""
    plugin_toolset = "completed_discovery_tools"
    plugin_tool = "completed_discovery_health"

    import hermes_cli.plugins as plugins_mod
    from tools.registry import registry

    manager = plugins_mod.PluginManager()
    manager._discovered = True
    registry.register(
        name=plugin_tool,
        toolset=plugin_toolset,
        schema={
            "name": plugin_tool,
            "description": "Read-only health check",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda _args, **_kwargs: '{"status":"ok"}',
    )
    manager._plugin_tool_names.add(plugin_tool)
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "_background_discovery_thread", None)

    def fail_on_join(*args, **kwargs):
        raise AssertionError("validation joined discovery although it had completed")

    monkeypatch.setattr(plugins_mod, "_join_background_discovery", fail_on_join)

    try:
        warnings = _validate_and_collect_cli_warnings([plugin_toolset])
        assert not any("Unknown toolsets" in msg for msg in warnings)
    finally:
        registry.deregister(plugin_tool)


def test_config_selected_plugin_toolset_waits_for_discovery_before_warning(
    tmp_path, monkeypatch, caplog
):
    """A cached plugin key must be validated against completed discovery."""
    plugin_name = "startup-validation-plugin"
    plugin_toolset = "startup_validation_tools"
    plugin_tool = "startup_validation_health"
    unknown_toolset = "genuinely_unknown_startup_toolset"
    hermes_home = tmp_path / "hermes-home"
    plugin_dir = hermes_home / "plugins" / plugin_name
    cache_dir = hermes_home / "cache"
    plugin_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)

    discovery_entered = threading.Event()
    allow_registration = threading.Event()
    registration_complete = threading.Event()
    validation_joined_discovery = threading.Event()
    gate_module = types.ModuleType("_startup_validation_plugin_gate")
    gate_module.__dict__.update(
        discovery_entered=discovery_entered,
        allow_registration=allow_registration,
        registration_complete=registration_complete,
    )
    monkeypatch.setitem(sys.modules, gate_module.__name__, gate_module)

    (plugin_dir / "plugin.yaml").write_text(
        "\n".join([
            f"name: {plugin_name}",
            'version: "0.1.0"',
            "description: Startup validation test plugin",
            "",
        ]),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "from _startup_validation_plugin_gate import (\n"
        "    allow_registration, discovery_entered, registration_complete,\n"
        ")\n\n"
        "def register(ctx):\n"
        "    discovery_entered.set()\n"
        "    if not allow_registration.wait(timeout=10):\n"
        "        raise RuntimeError('validation never reached plugin discovery')\n"
        "    ctx.register_tool(\n"
        f"        name={plugin_tool!r},\n"
        f"        toolset={plugin_toolset!r},\n"
        "        schema={\n"
        f"            'name': {plugin_tool!r},\n"
        "            'description': 'Read-only health check',\n"
        "            'parameters': {'type': 'object', 'properties': {}},\n"
        "        },\n"
        '        handler=lambda args, **kwargs: \'{"status": "ok"}\',\n'
        "    )\n"
        "    registration_complete.set()\n",
        encoding="utf-8",
    )
    config = {
        "plugins": {"enabled": [plugin_name]},
        "platform_toolsets": {"cli": [plugin_toolset]},
        "known_plugin_toolsets": {"cli": [plugin_toolset]},
    }
    (hermes_home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        f"    - {plugin_name}\n"
        "platform_toolsets:\n"
        "  cli:\n"
        f"    - {plugin_toolset}\n"
        "known_plugin_toolsets:\n"
        "  cli:\n"
        f"    - {plugin_toolset}\n",
        encoding="utf-8",
    )
    (cache_dir / "plugin_toolset_keys.json").write_text(
        json.dumps({"toolset_keys": [plugin_toolset], "portable_mcp": []}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import hermes_cli.plugins as plugins_mod
    from hermes_cli.tools_config import _get_platform_tools
    from tools.registry import registry

    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    monkeypatch.setattr(plugins_mod, "_plugin_manager", None)
    monkeypatch.setattr(plugins_mod, "_background_discovery_thread", None)

    try:
        plugins_mod.start_background_plugin_discovery()
        assert discovery_entered.wait(timeout=5), "plugin discovery did not start"

        selected = _get_platform_tools(config, "cli")
        assert plugin_toolset in selected
        assert not registration_complete.is_set()
        assert not any(
            "no valid toolsets configured" in record.getMessage()
            and plugin_toolset in record.getMessage()
            for record in caplog.records
        )

        original_join = plugins_mod._join_background_discovery

        def release_then_join_discovery(*args, **kwargs):
            validation_joined_discovery.set()
            allow_registration.set()
            return original_join(*args, **kwargs)

        monkeypatch.setattr(
            plugins_mod, "_join_background_discovery", release_then_join_discovery
        )

        warnings = _validate_and_collect_cli_warnings([plugin_toolset, unknown_toolset], config)

        assert validation_joined_discovery.is_set()
        assert registration_complete.is_set()
        unknown_warnings = [msg for msg in warnings if "Unknown toolsets" in msg]
        assert len(unknown_warnings) == 1
        assert unknown_toolset in unknown_warnings[0]
        assert plugin_toolset not in unknown_warnings[0]
    finally:
        allow_registration.set()
        thread = plugins_mod._background_discovery_thread
        if thread is not None:
            thread.join(timeout=5)
        registry.deregister(plugin_tool)


def test_discovery_failure_during_validation_logs_warning(monkeypatch, caplog):
    """A plugin discovery failure during toolset validation must log a warning."""
    import hermes_cli.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "plugin_discovery_in_flight", lambda: True)
    monkeypatch.setattr(plugins_mod, "get_plugin_toolset_keys_nowait", set)
    monkeypatch.setattr(plugins_mod, "get_portable_mcp_server_names_nowait", set)

    def _boom():
        raise RuntimeError("broken plugin manifest")

    monkeypatch.setattr(plugins_mod, "discover_plugins", _boom)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.toolset_validation"):
        warnings = _validate_and_collect_cli_warnings(["missing_plugin_toolset"])

    assert any(
        "plugin discovery failed during toolset validation" in record.getMessage()
        for record in caplog.records
    )
    assert any("missing_plugin_toolset" in msg for msg in warnings)
