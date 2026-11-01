"""Plugins whose own code imports siblings by absolute path must register by that same path.

The loader executes a bundled platform package under a synthetic ``hermes_plugins.platforms__<name>``
name. A package whose adapter (or standalone sender / tools) imports ``plugins.platforms.<name>.*``
would then hold two adapter classes: the registry's and the one the rest of the code sees
(Yuanbao's ``get_active()`` singleton, isinstance checks, monkeypatches).
"""

import importlib

import pytest


@pytest.mark.parametrize("platform, cls", [("signal", "SignalAdapter"), ("yuanbao", "YuanbaoAdapter")])
def test_registry_factory_is_the_canonical_adapter_class(platform, cls):
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins

    discover_plugins()
    entry = platform_registry.get(platform)
    canonical = getattr(importlib.import_module(f"plugins.platforms.{platform}.adapter"), cls)
    assert entry.adapter_factory is canonical, entry.adapter_factory.__module__
