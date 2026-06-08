"""Regression tests for the Vity memory provider."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from plugins.memory.vity import VityMemoryProvider, _load_config


def test_config_only_loads_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("MAXIMEM_API_KEY", "mx_test_key")
    monkeypatch.setenv("VITY_CHANNEL", "ignored")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    assert _load_config() == {"api_key": "mx_test_key"}


def test_config_schema_has_no_channel():
    schema = VityMemoryProvider().get_config_schema()

    assert [item["key"] for item in schema] == ["api_key"]


def test_initialize_ignores_gateway_identity(monkeypatch):
    provider = VityMemoryProvider()
    monkeypatch.setattr(
        "plugins.memory.vity._load_config",
        lambda: {"api_key": "mx_test_key"},
    )
    monkeypatch.setattr(provider, "queue_prefetch", MagicMock())

    provider.initialize("session-1", channel="room-1", user_id="user-1")

    assert provider._api_key == "mx_test_key"
    assert not hasattr(provider, "_channel")


def test_client_is_created_with_api_key_only(monkeypatch):
    constructor = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "maximem_vity", SimpleNamespace(VityClient=constructor))
    provider = VityMemoryProvider()
    provider._api_key = "mx_test_key"

    provider._get_client()

    constructor.assert_called_once_with(api_key="mx_test_key")


def test_system_prompt_describes_api_key_scope():
    block = VityMemoryProvider().system_prompt_block()

    assert "configured Maximem API key" in block
    assert "Channel:" not in block
