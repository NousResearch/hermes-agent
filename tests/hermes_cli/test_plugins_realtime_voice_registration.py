"""End-to-end tests for PluginContext.register_realtime_voice_provider().

Drops a fake plugin into ``$HERMES_HOME/plugins/``, runs
``PluginManager().discover_and_load()``, and asserts the registry outcome —
the same structure as ``test_plugins_tts_registration.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

_FAKE_PROVIDER_BODY = (
    "from agent.realtime_voice_provider import RealtimeVoiceProvider, RealtimeVoiceSession\n"
    "    class S(RealtimeVoiceSession):\n"
    "        async def send_audio(self, audio): pass\n"
    "        def _events(self):\n"
    "            async def stream():\n"
    "                if False: yield None\n"
    "            return stream()\n"
    "        async def _close(self): pass\n"
    "    class P(RealtimeVoiceProvider):\n"
    "        {extra}\n"
    "        @property\n"
    "        def name(self): return '{name}'\n"
    "        async def open_session(self, setup): return S()\n"
)


def _provider_body(name: str, *, extra: str = "pass", tail: str) -> str:
    return _FAKE_PROVIDER_BODY.format(name=name, extra=extra) + "    " + tail


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
    (plugin_dir / "plugin.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        f"def register(ctx):\n    {register_body}\n", encoding="utf-8"
    )
    return plugin_dir


def _enable(hermes_home: Path, name: str) -> None:
    cfg_path = hermes_home / "config.yaml"
    cfg: dict = {}
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}
    plugins_cfg = cfg.setdefault("plugins", {})
    enabled = plugins_cfg.setdefault("enabled", [])
    if isinstance(enabled, list) and name not in enabled:
        enabled.append(name)
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_registry():
    from agent import realtime_voice_registry

    realtime_voice_registry._reset_for_tests()
    yield
    realtime_voice_registry._reset_for_tests()


class TestRegisterRealtimeVoiceProvider:
    def test_accepts_valid_provider_and_unload_removes_it(self):
        from hermes_cli.plugins import PluginManager

        from agent import realtime_voice_registry

        hermes_home = Path(os.environ["HERMES_HOME"])
        _write_plugin(
            hermes_home / "plugins",
            "my-realtime-plugin",
            register_body=_provider_body(
                "fake-realtime", tail="ctx.register_realtime_voice_provider(P())"
            ),
        )
        _enable(hermes_home, "my-realtime-plugin")

        manager = PluginManager()
        manager.discover_and_load()

        assert manager._plugins["my-realtime-plugin"].enabled is True, (
            f"Plugin failed to load: {manager._plugins['my-realtime-plugin'].error}"
        )
        assert realtime_voice_registry.get_provider("fake-realtime") is not None

        assert manager.unload("my-realtime-plugin") is True
        assert realtime_voice_registry.get_provider("fake-realtime") is None

    def test_unload_restores_previous_exact_registration(self):
        from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
        from tests.agent.test_realtime_voice_registry import _FakeProvider

        from agent import realtime_voice_registry

        manager = PluginManager()
        first = _FakeProvider("shared-realtime")
        second = _FakeProvider("shared-realtime")
        first_context = PluginContext(PluginManifest(name="first-plugin"), manager)
        second_context = PluginContext(PluginManifest(name="second-plugin"), manager)

        assert first_context.register_realtime_voice_provider(first) is not None
        assert second_context.register_realtime_voice_provider(second) is not None
        assert realtime_voice_registry.get_provider("shared-realtime") is second

        assert manager.unload("second-plugin") is True
        assert realtime_voice_registry.get_provider("shared-realtime") is first

        assert manager.unload("first-plugin") is True
        assert realtime_voice_registry.get_provider("shared-realtime") is None

    def test_registration_is_scoped_to_the_manager_home(self):
        from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
        from tests.agent.test_realtime_voice_registry import _FakeProvider

        from agent import realtime_voice_registry

        manager = PluginManager()
        provider = _FakeProvider("scoped-realtime")
        context = PluginContext(PluginManifest(name="scoped-plugin"), manager)

        assert context.register_realtime_voice_provider(provider) is not None
        assert (
            realtime_voice_registry.get_provider("scoped-realtime", scope=manager.scope_key)
            is provider
        )
        assert realtime_voice_registry.snapshot_registration("scoped-realtime") is None
        manager.unload("scoped-plugin")

    def test_register_then_raise_rolls_back_provider(self):
        from hermes_cli.plugins import PluginManager

        from agent import realtime_voice_registry

        hermes_home = Path(os.environ["HERMES_HOME"])
        _write_plugin(
            hermes_home / "plugins",
            "failing-realtime-plugin",
            register_body=_provider_body(
                "failed-realtime",
                tail=(
                    "assert ctx.register_realtime_voice_provider(P()) is not None\n"
                    "    raise RuntimeError('registration failed')"
                ),
            ),
        )
        _enable(hermes_home, "failing-realtime-plugin")

        manager = PluginManager()
        manager.discover_and_load()

        assert manager._plugins["failing-realtime-plugin"].enabled is False
        assert realtime_voice_registry.get_provider("failed-realtime") is None

    def test_rejects_non_provider(self, caplog):
        from hermes_cli.plugins import PluginManager

        from agent import realtime_voice_registry

        hermes_home = Path(os.environ["HERMES_HOME"])
        _write_plugin(
            hermes_home / "plugins",
            "bad-realtime-plugin",
            register_body=(
                "assert ctx.register_realtime_voice_provider('not a provider') is None"
            ),
        )
        _enable(hermes_home, "bad-realtime-plugin")

        with caplog.at_level("WARNING"):
            manager = PluginManager()
            manager.discover_and_load()

        assert manager._plugins["bad-realtime-plugin"].enabled is True
        # Only bundled backends may be registered; the string never was.
        assert all(
            type(provider).__module__.startswith("plugins.")
            for provider in realtime_voice_registry.list_providers(scope=manager.scope_key)
        )
        assert "does not inherit from RealtimeVoiceProvider" in caplog.text

    def test_rejects_incompatible_provider_api(self, caplog):
        from hermes_cli.plugins import PluginManager

        from agent import realtime_voice_registry

        hermes_home = Path(os.environ["HERMES_HOME"])
        _write_plugin(
            hermes_home / "plugins",
            "old-realtime-plugin",
            register_body=_provider_body(
                "old-realtime",
                extra="api_version = 0",
                tail="assert ctx.register_realtime_voice_provider(P()) is None",
            ),
        )
        _enable(hermes_home, "old-realtime-plugin")

        with caplog.at_level("WARNING"):
            manager = PluginManager()
            manager.discover_and_load()

        assert manager._plugins["old-realtime-plugin"].enabled is True
        assert realtime_voice_registry.get_provider("old-realtime") is None
        assert "targets API v0" in caplog.text
