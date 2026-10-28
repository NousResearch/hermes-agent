from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.platforms.base import resolve_channel_toolsets
from gateway.run import GatewayRunner
from model_tools import get_tool_definitions
from plugins.platforms.discord.adapter import DiscordAdapter


def _adapter(extra):
    adapter = object.__new__(DiscordAdapter)
    adapter.config = MagicMock()
    adapter.config.extra = extra
    return adapter


def _runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda source: adapter
    return runner


def test_exact_thread_empty_binding_resolves_to_zero_tools():
    extra = {
        "channel_toolset_bindings": [
            {"id": "parent", "toolsets": ["web"]},
            {"id": "thread", "toolsets": []},
        ]
    }
    source = SimpleNamespace(chat_id="thread", parent_chat_id="parent")
    adapter = _adapter(extra)

    assert resolve_channel_toolsets(extra, "thread", "parent") == []
    assert adapter.toolsets_for_source(source) == []

    enabled = GatewayRunner._resolve_enabled_toolsets_for_source(
        _runner(adapter),
        {"platform_toolsets": {"discord": ["hermes-discord"]}},
        source,
        "discord",
    )
    assert enabled == []
    assert get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True) == []


def test_parent_binding_is_inherited():
    extra = {"channel_toolset_bindings": [{"id": "parent", "toolsets": ["web"]}]}
    source = SimpleNamespace(chat_id="thread", parent_chat_id="parent")
    assert _adapter(extra).toolsets_for_source(source) == ["web"]


def test_exact_binding_wins_over_parent():
    extra = {
        "channel_toolset_bindings": [
            {"id": "parent", "toolsets": ["terminal"]},
            {"id": "thread", "toolsets": ["web"]},
        ]
    }
    assert resolve_channel_toolsets(extra, "thread", "parent") == ["web"]


def test_no_binding_preserves_platform_defaults():
    source = SimpleNamespace(chat_id="thread", parent_chat_id="parent")
    adapter = _adapter({})
    assert adapter.toolsets_for_source(source) is None
    enabled = GatewayRunner._resolve_enabled_toolsets_for_source(
        _runner(adapter),
        {"platform_toolsets": {"discord": ["web"]}},
        source,
        "discord",
    )
    assert "web" in enabled
