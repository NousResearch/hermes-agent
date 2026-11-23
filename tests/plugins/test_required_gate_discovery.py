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
from tests.hermes_cli.test_plugin_host import _home_with_plugins

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

# Decides on the event's text and sender; "handle" makes it reach into the live gateway runner.
HOSTED_GATE = textwrap.dedent('''\
    def gate(event, gateway=None, **kwargs):
        if "handle" in event.text:
            gateway.adapters
        refused = "deny" in event.text or event.source.user_id != "7"
        return {"action": "skip", "reason": "policy"} if refused else {"action": "allow"}

    def register(ctx):
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


def _event(text="hello", user_id="7"):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", user_id=user_id, user_name="t")
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id="m1")


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


@pytest.mark.platforms("any")  # the host is a child process
@pytest.mark.parametrize("isolation", ["host", "in_process"])
@pytest.mark.asyncio
async def test_a_gate_decides_on_text_and_sender_in_either_isolation_mode(tmp_path, monkeypatch, isolation):
    """Under ``plugins.isolation: host`` the gate runs in the plugin host: the event's text and source
    arrive as plain data and its decision is required all the same. The gateway runner arrives as a
    placeholder there (``HOST_DEGRADED_HOOKS``), so a gate that reaches into it fails, and refuses."""
    _home_with_plugins(tmp_path, monkeypatch, {"policy": HOSTED_GATE}, isolation=isolation)
    plugins._reset_plugin_managers_for_tests()
    manager = plugins.get_plugin_manager()
    manager.discover_and_load()
    try:
        assert manager._plugins["policy"].error is None and manager.has_fail_closed_hook("pre_gateway_dispatch")
        runner = _runner()
        assert await runner._handle_message(_event()) == "dispatched"
        assert await runner._handle_message(_event("deny this")) is None
        assert await runner._handle_message(_event(user_id="8")) is None
        assert await runner._handle_message(_event("handle")) is None
    finally:
        manager.unload()
        if isolation == "host":
            manager._plugin_host().shutdown()
        plugins._reset_plugin_managers_for_tests()
