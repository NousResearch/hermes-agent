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


def test_openviking_facade_reads_live_last_active_provider(monkeypatch):
    openviking = importlib.import_module("plugins.memory.openviking")
    provider_module = importlib.import_module("plugins.memory.openviking.provider")
    sentinel = object()

    assert "_last_active_provider" not in vars(openviking)

    monkeypatch.setattr(provider_module, "_last_active_provider", sentinel)

    assert openviking._last_active_provider is sentinel


def test_openviking_facade_client_patch_is_used_by_provider_new_client(monkeypatch):
    openviking = importlib.import_module("plugins.memory.openviking")
    created = []

    class FakeVikingClient:
        def __init__(self, endpoint, api_key, *, account="", user="", agent=""):
            created.append(
                {
                    "endpoint": endpoint,
                    "api_key": api_key,
                    "account": account,
                    "user": user,
                    "agent": agent,
                }
            )

    monkeypatch.setattr(openviking, "_VikingClient", FakeVikingClient)

    provider = openviking.OpenVikingMemoryProvider()
    provider._endpoint = "http://openviking.test"
    provider._api_key = "test-key"
    provider._account = "test-account"
    provider._user = "test-user"
    provider._agent = "test-agent"

    assert provider._new_client().__class__ is FakeVikingClient
    assert created == [
        {
            "endpoint": "http://openviking.test",
            "api_key": "test-key",
            "account": "test-account",
            "user": "test-user",
            "agent": "test-agent",
        }
    ]
