"""Regression for #126933: per-provider plugin registrations log at DEBUG.

The shared ``_register_scoped_provider`` path used to emit one INFO line per
provider on every process start (image_gen / video_gen / web / browser /
terminal / secret / TTS / transcription). Memory providers were already
DEBUG; keep the single ``Plugin discovery complete`` INFO summary.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import hermes_yaml as yaml


def _write_plugin(
    root: Path,
    name: str,
    *,
    register_body: str,
    manifest_extra: Dict[str, Any] | None = None,
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
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump(manifest))
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


class TestPluginRegistrationLogLevel:
    def test_scoped_provider_registration_is_debug_not_info(self, caplog):
        from hermes_cli.plugins import PluginManager
        from agent import tts_registry

        tts_registry._reset_for_tests()

        hermes_home = Path(os.environ["HERMES_HOME"])
        _write_plugin(
            hermes_home / "plugins",
            "quiet-tts-plugin",
            register_body=(
                "from agent.tts_provider import TTSProvider\n"
                "    class P(TTSProvider):\n"
                "        @property\n"
                "        def name(self): return 'quiet-tts'\n"
                "        def synthesize(self, text, output_path, **kw):\n"
                "            return output_path\n"
                "    ctx.register_tts_provider(P())"
            ),
        )
        _enable(hermes_home, "quiet-tts-plugin")

        with caplog.at_level(logging.DEBUG, logger="hermes_cli.plugins"):
            PluginManager().discover_and_load()

        registration_records = [
            r
            for r in caplog.records
            if r.name == "hermes_cli.plugins"
            and "registered" in r.getMessage()
            and "quiet-tts" in r.getMessage()
        ]
        assert registration_records, "expected a registration log for quiet-tts"
        assert all(r.levelno == logging.DEBUG for r in registration_records), [
            (r.levelname, r.getMessage()) for r in registration_records
        ]

        info_registration = [
            r.getMessage()
            for r in caplog.records
            if r.name == "hermes_cli.plugins"
            and r.levelno == logging.INFO
            and "registered" in r.getMessage().lower()
        ]
        assert info_registration == []
