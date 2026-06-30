"""Tests for PluginContext.register_transcription_provider().

Exercises the plugin context hook end-to-end: drops a fake plugin into
``$HERMES_HOME/plugins/``, runs ``PluginManager().discover_and_load()``,
and asserts the registration result.

Mirrors the shape of ``test_plugins_tts_registration.py`` (companion
TTS hook from issue #30398).
"""

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


class TestRegisterTranscriptionProvider:
    def test_accepts_valid_provider(self):
        from hermes_cli.plugins import PluginManager

        from agent import transcription_registry
        transcription_registry._reset_for_tests()

        hermes_home = Path(os.environ["HERMES_HOME"])
        _write_plugin(
            hermes_home / "plugins",
            "my-stt-plugin",
            register_body=(
                "from agent.transcription_provider import TranscriptionProvider\n"
                "    class P(TranscriptionProvider):\n"
                "        @property\n"
                "        def name(self): return 'fake-stt'\n"
                "        def transcribe(self, file_path, **kw):\n"
                "            return {'success': True, 'transcript': 'hi', 'provider': 'fake-stt'}\n"
                "    ctx.register_transcription_provider(P())"
            ),
        )
        _enable(hermes_home, "my-stt-plugin")

        mgr = PluginManager()
        mgr.discover_and_load()

        assert mgr._plugins["my-stt-plugin"].enabled is True, (
            f"Plugin failed to load: {mgr._plugins['my-stt-plugin'].error}"
        )
        assert transcription_registry.get_provider("fake-stt") is not None

        transcription_registry._reset_for_tests()

    def test_rejects_non_provider(self, caplog):
        from hermes_cli.plugins import PluginManager

        from agent import transcription_registry
        transcription_registry._reset_for_tests()

        hermes_home = Path(os.environ["HERMES_HOME"])
        _write_plugin(
            hermes_home / "plugins",
            "bad-stt-plugin",
            register_body="ctx.register_transcription_provider('not a provider')",
        )
        _enable(hermes_home, "bad-stt-plugin")

        with caplog.at_level("WARNING"):
            mgr = PluginManager()
            mgr.discover_and_load()

        assert mgr._plugins["bad-stt-plugin"].enabled is True
        assert transcription_registry.get_provider("not a provider") is None
        assert "not a provider" not in {
            provider.name for provider in transcription_registry.list_providers()
        }
        assert "does not inherit from TranscriptionProvider" in caplog.text

        transcription_registry._reset_for_tests()

    def test_rejects_builtin_shadow(self, caplog):
        from hermes_cli.plugins import PluginManager

        from agent import transcription_registry
        transcription_registry._reset_for_tests()

        hermes_home = Path(os.environ["HERMES_HOME"])
        _write_plugin(
            hermes_home / "plugins",
            "shadow-stt-plugin",
            register_body=(
                "from agent.transcription_provider import TranscriptionProvider\n"
                "    class P(TranscriptionProvider):\n"
                "        @property\n"
                "        def name(self): return 'openai'\n"
                "        def transcribe(self, file_path, **kw):\n"
                "            return {'success': True, 'transcript': 'hi'}\n"
                "    ctx.register_transcription_provider(P())"
            ),
        )
        _enable(hermes_home, "shadow-stt-plugin")

        with caplog.at_level("WARNING"):
            mgr = PluginManager()
            mgr.discover_and_load()

        # Plugin still loaded normally — built-in shadowing is a warning,
        # not an exception. The registry rejects the entry though.
        assert mgr._plugins["shadow-stt-plugin"].enabled is True
        assert transcription_registry.get_provider("openai") is None
        assert "shadows a built-in name" in caplog.text

        transcription_registry._reset_for_tests()
