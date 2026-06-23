"""Import-boundary tests for the OpenViking memory plugin package split."""

from __future__ import annotations

import importlib


def test_openviking_split_modules_are_importable():
    for module_name in (
        "plugins.memory.openviking.client",
        "plugins.memory.openviking.config",
        "plugins.memory.openviking.provider",
        "plugins.memory.openviking.schemas",
        "plugins.memory.openviking.setup",
        "plugins.memory.openviking.tools",
        "plugins.memory.openviking.transcript",
    ):
        assert importlib.import_module(module_name)


def test_openviking_package_facade_keeps_existing_public_imports():
    openviking = importlib.import_module("plugins.memory.openviking")

    assert openviking.OpenVikingMemoryProvider.__name__ == "OpenVikingMemoryProvider"
    assert openviking._VikingClient.__name__ == "_VikingClient"
    assert isinstance(openviking._DEFERRED_COMMIT_TIMEOUT, float)
    assert openviking.SEARCH_SCHEMA["name"] == "viking_search"
    assert openviking.ADD_RESOURCE_SCHEMA["name"] == "viking_add_resource"
    assert callable(openviking.register)
