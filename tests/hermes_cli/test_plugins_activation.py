"""Regression tests for concurrent plugin activation summaries."""

from types import SimpleNamespace

from hermes_cli.plugins_activation import plugin_activation_summary


def test_activation_summary_snapshots_handler_registries_before_factory_iteration():
    manager = SimpleNamespace(
        _plugins={"plugin": SimpleNamespace(manifest=SimpleNamespace(name="plugin"), tools_registered=[])},
        _ownership_ledger={"plugin": []},
        _portable_mcp_server_plugins={},
    )

    class MutatingFactories:
        def __iter__(self):
            manager._platform_handler_factories["registered-during-summary"] = []
            return iter(((object(), "plugin"),))

    manager._platform_handler_factories = {"initial": MutatingFactories()}

    summary = plugin_activation_summary(manager, "plugin")

    assert summary["activated_now"]["callbacks"] == ["initial"]
