"""Tests for PluginContext.register_tts_provider() (issue #30398).

Exercises the plugin context hook end-to-end: drops a fake plugin into
``$HERMES_HOME/plugins/``, runs ``PluginManager().discover_and_load()``,
and asserts the registration result.

Mirrors the structure of
``tests/hermes_cli/test_plugin_scanner_recursion.py::TestRegisterImageGenProvider``.
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


class TestRegisterTTSProvider:
    """End-to-end: a fake plugin registers via the hook, ends up in the registry."""

    def test_accepts_valid_provider(self):
        from hermes_cli.plugins import PluginManager

        from agent import tts_registry
        tts_registry._reset_for_tests()

        hermes_home = Path(os.environ["HERMES_HOME"])
        _write_plugin(
            hermes_home / "plugins",
            "my-tts-plugin",
            register_body=(
                "from agent.tts_provider import TTSProvider\n"
                "    class P(TTSProvider):\n"
                "        @property\n"
                "        def name(self): return 'fake-tts'\n"
                "        def synthesize(self, text, output_path, **kw):\n"
                "            return output_path\n"
                "    ctx.register_tts_provider(P())"
            ),
        )
        _enable(hermes_home, "my-tts-plugin")

        mgr = PluginManager()
        mgr.discover_and_load()

        assert mgr._plugins["my-tts-plugin"].enabled is True, (
            f"Plugin failed to load: {mgr._plugins['my-tts-plugin'].error}"
        )
        assert tts_registry.get_provider("fake-tts") is not None

        tts_registry._reset_for_tests()

    def test_rejects_non_provider(self, caplog):
        """A plugin that passes a non-TTSProvider gets a warning, no exception."""
        from hermes_cli.plugins import PluginManager

        from agent import tts_registry
        tts_registry._reset_for_tests()

        hermes_home = Path(os.environ["HERMES_HOME"])
        _write_plugin(
            hermes_home / "plugins",
            "bad-tts-plugin",
            register_body="ctx.register_tts_provider('not a provider')",
        )
        _enable(hermes_home, "bad-tts-plugin")

        with caplog.at_level("WARNING"):
            mgr = PluginManager()
            mgr.discover_and_load()

        # Plugin loaded (register returned normally), but the bad plugin's
        # non-provider was rejected — only bundled backends (e.g. cloudflare)
        # may be present from discover_and_load().
        assert mgr._plugins["bad-tts-plugin"].enabled is True
        assert tts_registry.get_provider("not a provider") is None
        registered_names = {p.name for p in tts_registry.list_providers()}
        assert "not a provider" not in registered_names
        assert "does not inherit from TTSProvider" in caplog.text

        tts_registry._reset_for_tests()

    def test_rejects_builtin_shadow(self, caplog):
        """A plugin trying to register a name colliding with a built-in is silently
        rejected by the underlying registry — both with a registry-level warning
        AND with the registry remaining empty (plugin still loads OK).
        """
        from hermes_cli.plugins import PluginManager

        from agent import tts_registry
        tts_registry._reset_for_tests()

        hermes_home = Path(os.environ["HERMES_HOME"])
        _write_plugin(
            hermes_home / "plugins",
            "shadow-tts-plugin",
            register_body=(
                "from agent.tts_provider import TTSProvider\n"
                "    class P(TTSProvider):\n"
                "        @property\n"
                "        def name(self): return 'edge'\n"
                "        def synthesize(self, text, output_path, **kw):\n"
                "            return output_path\n"
                "    ctx.register_tts_provider(P())"
            ),
        )
        _enable(hermes_home, "shadow-tts-plugin")

        with caplog.at_level("WARNING"):
            mgr = PluginManager()
            mgr.discover_and_load()

        # Plugin still loaded normally — built-in shadowing is a warning,
        # not an exception. The registry rejects the entry though.
        assert mgr._plugins["shadow-tts-plugin"].enabled is True
        assert tts_registry.get_provider("edge") is None
        assert "shadows a built-in name" in caplog.text

        tts_registry._reset_for_tests()
