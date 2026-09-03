from __future__ import annotations

import os
from pathlib import Path

import yaml

from agent import realtime_voice_registry
from hermes_cli.plugins import PluginManager


def test_plugin_registers_realtime_voice_provider_end_to_end():
    hermes_home = Path(os.environ["HERMES_HOME"])
    plugin_dir = hermes_home / "plugins" / "fake-realtime-voice"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "fake-realtime-voice",
                "version": "0.1.0",
                "description": "Test realtime voice provider",
            }
        )
    )
    (plugin_dir / "__init__.py").write_text(
        """from agent.realtime_voice import RealtimeSession, RealtimeVoiceProvider

class Session(RealtimeSession):
    async def send_audio(self, pcm): pass
    async def events(self):
        if False: yield
    async def submit_tool_result(self, call_id, output): pass
    async def cancel_response(self): pass
    async def close(self): pass

class Provider(RealtimeVoiceProvider):
    @property
    def name(self): return "fake-duplex"
    async def open_session(self, *, instructions, tools, voice=None): return Session()

def register(ctx):
    ctx.register_realtime_voice_provider(Provider())
"""
    )
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["fake-realtime-voice"]}})
    )
    realtime_voice_registry._reset_for_tests()

    manager = PluginManager()
    manager.discover_and_load()

    plugin = manager._plugins["fake-realtime-voice"]
    assert plugin.enabled is True, plugin.error
    assert realtime_voice_registry.get_provider("fake-duplex") is not None

    manager.unload()
    assert realtime_voice_registry.get_provider("fake-duplex") is None
