"""End-to-end tests for PluginContext.register_realtime_voice_provider()."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _write_plugin(
    root: Path,
    name: str,
    *,
    manifest_extra: Dict[str, Any] | None = None,
    register_body: str = "pass",
) -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "version": "0.1.0",
        "description": f"Test plugin {name}",
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    (plugin_dir / "plugin.yaml").write_text(yaml.dump(manifest))
    (plugin_dir / "__init__.py").write_text(
        f"def register(ctx):\n    {register_body}\n"
    )
    return plugin_dir


def _enable(hermes_home: Path, name: str) -> None:
    cfg_path = hermes_home / "config.yaml"
    cfg: dict = {}
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            cfg = {}
    plugins_cfg = cfg.setdefault("plugins", {})
    enabled = plugins_cfg.setdefault("enabled", [])
    if isinstance(enabled, list) and name not in enabled:
        enabled.append(name)
    cfg_path.write_text(yaml.safe_dump(cfg))


class TestRegisterRealtimeVoiceProvider:
    def test_accepts_valid_provider(self):
        from hermes_cli.plugins import PluginManager

        from agent import realtime_voice_registry

        realtime_voice_registry._reset_for_tests()
        hermes_home = Path(os.environ["HERMES_HOME"])
        _write_plugin(
            hermes_home / "plugins",
            "my-realtime-plugin",
            register_body=(
                "from agent.realtime_voice_provider import RealtimeVoiceProvider, RealtimeVoiceSession\n"
                "    class S(RealtimeVoiceSession):\n"
                "        async def send_audio(self, audio, **kw): pass\n"
                "        async def send_text(self, text, **kw): pass\n"
                "        async def send_tool_result(self, call_id, output, **kw): pass\n"
                "        def events(self):\n"
                "            async def stream():\n"
                "                if False: yield {}\n"
                "            return stream()\n"
                "        async def close(self): pass\n"
                "    class P(RealtimeVoiceProvider):\n"
                "        @property\n"
                "        def name(self): return 'fake-realtime'\n"
                "        async def open_session(self, **kw): return S()\n"
                "    ctx.register_realtime_voice_provider(P())"
            ),
        )
        _enable(hermes_home, "my-realtime-plugin")

        manager = PluginManager()
        manager.discover_and_load()

        assert manager._plugins["my-realtime-plugin"].enabled is True, (
            f"Plugin failed to load: {manager._plugins['my-realtime-plugin'].error}"
        )
        assert realtime_voice_registry.get_provider("fake-realtime") is not None

        realtime_voice_registry._reset_for_tests()

    def test_rejects_non_provider(self, caplog):
        from hermes_cli.plugins import PluginManager

        from agent import realtime_voice_registry

        realtime_voice_registry._reset_for_tests()
        hermes_home = Path(os.environ["HERMES_HOME"])
        _write_plugin(
            hermes_home / "plugins",
            "bad-realtime-plugin",
            register_body="ctx.register_realtime_voice_provider('not a provider')",
        )
        _enable(hermes_home, "bad-realtime-plugin")

        with caplog.at_level("WARNING"):
            manager = PluginManager()
            manager.discover_and_load()

        assert manager._plugins["bad-realtime-plugin"].enabled is True
        assert realtime_voice_registry.list_providers() == []
        assert "does not inherit from RealtimeVoiceProvider" in caplog.text

        realtime_voice_registry._reset_for_tests()

    def test_rejects_incompatible_provider_api(self, caplog):
        from hermes_cli.plugins import PluginManager

        from agent import realtime_voice_registry

        realtime_voice_registry._reset_for_tests()
        hermes_home = Path(os.environ["HERMES_HOME"])
        _write_plugin(
            hermes_home / "plugins",
            "old-realtime-plugin",
            register_body=(
                "from agent.realtime_voice_provider import RealtimeVoiceProvider, RealtimeVoiceSession\n"
                "    class S(RealtimeVoiceSession):\n"
                "        async def send_audio(self, audio, **kw): pass\n"
                "        async def send_text(self, text, **kw): pass\n"
                "        async def send_tool_result(self, call_id, output, **kw): pass\n"
                "        def events(self):\n"
                "            async def stream():\n"
                "                if False: yield {}\n"
                "            return stream()\n"
                "        async def close(self): pass\n"
                "    class P(RealtimeVoiceProvider):\n"
                "        api_version = 0\n"
                "        @property\n"
                "        def name(self): return 'old-realtime'\n"
                "        async def open_session(self, **kw): return S()\n"
                "    ctx.register_realtime_voice_provider(P())"
            ),
        )
        _enable(hermes_home, "old-realtime-plugin")

        with caplog.at_level("WARNING"):
            manager = PluginManager()
            manager.discover_and_load()

        assert manager._plugins["old-realtime-plugin"].enabled is True
        assert realtime_voice_registry.get_provider("old-realtime") is None
        assert "targets API v0" in caplog.text

        realtime_voice_registry._reset_for_tests()
