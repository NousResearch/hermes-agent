"""A required ``pre_gateway_dispatch`` gate (``fail_closed=True``) loaded through the real loaders.

The plugin lives on disk in a temporary ``HERMES_HOME``. Through general discovery and through
the memory-provider loader alike, the registration stays required: a gate that raises refuses the
message instead of letting it through.
"""

import textwrap

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from hermes_cli import plugins
from plugins.memory import load_memory_provider

GATE = textwrap.dedent('''\
    def gate(event, **kwargs):
        raise RuntimeError("policy backend unavailable")

    def register(ctx):
        ctx.register_hook("pre_gateway_dispatch", gate, fail_closed=True)
''')

MEMORY_GATE = textwrap.dedent('''\
    from agent.memory_provider import MemoryProvider

    class Provider(MemoryProvider):
        name = "policy"
        def is_available(self): return True
        def initialize(self, session_id, **kwargs): pass
        def get_tool_schemas(self): return []

    def gate(event, **kwargs):
        raise RuntimeError("policy backend unavailable")

    def register(ctx):
        ctx.register_memory_provider(Provider())
        ctx.register_hook("pre_gateway_dispatch", gate, fail_closed=True)
''')


def _install(home, monkeypatch, source, config):
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(home / "empty"))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    monkeypatch.chdir(home)
    (home / "config.yaml").write_text(config)
    plugin = home / "plugins" / "policy"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: policy\nversion: 1.0.0\n")
    (plugin / "__init__.py").write_text(source)
    plugins._reset_plugin_managers_for_tests()


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)})
    runner.session_store = None
    runner._is_user_authorized_for_source = lambda _source: True
    runner._admit_bot_message_for_source = lambda _source: True
    runner._admit_bot_chain = lambda _event: True
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._hm_estop_gate = lambda _event, _source, _internal: "dispatched"
    return runner


def _event():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", user_id="7", user_name="t")
    return MessageEvent(text="hello", message_type=MessageType.TEXT, source=source, message_id="m1")


@pytest.mark.asyncio
async def test_a_discovered_gate_that_raises_refuses_the_message(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, GATE, "plugins:\n  enabled: [policy]\n")
    try:
        manager = plugins.get_plugin_manager()
        manager.discover_and_load()
        assert manager.has_fail_closed_hook("pre_gateway_dispatch")
        assert await _runner()._handle_message(_event()) is None
    finally:
        plugins._reset_plugin_managers_for_tests()


@pytest.mark.asyncio
async def test_a_memory_provider_keeps_its_gate_required(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, MEMORY_GATE, "plugins:\n  enabled: []\nmemory:\n  provider: policy\n")
    try:
        manager = plugins.get_plugin_manager()
        assert load_memory_provider("policy") is not None
        assert manager.has_fail_closed_hook("pre_gateway_dispatch")
        assert await _runner()._handle_message(_event()) is None
    finally:
        plugins._reset_plugin_managers_for_tests()
