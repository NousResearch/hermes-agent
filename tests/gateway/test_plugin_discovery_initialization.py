"""Plugin discovery must not run while ``gateway.run`` is partially imported."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


def test_gateway_import_defers_user_plugin_discovery_until_explicit_startup(tmp_path):
    hermes_home = tmp_path / "hermes-home"
    plugin_dir = hermes_home / "plugins" / "gateway_importer"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "gateway_importer", "version": "0.1.0"}),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    from gateway.run import GatewayRunner\n"
        "    ctx.register_middleware('llm_request', lambda **kwargs: None)\n",
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({
            "plugins": {"enabled": ["gateway_importer"]},
            "auxiliary": {"title_generation": {"provider": "auto"}},
        }),
        encoding="utf-8",
    )

    script = """
import gateway.run
from hermes_cli.plugins import discover_plugins, get_plugin_manager

manager = get_plugin_manager()
assert manager._discovered is False, "gateway import triggered plugin discovery"
discover_plugins()
loaded = manager._plugins["gateway_importer"]
assert loaded.enabled is True, loaded.error
assert loaded.error is None
assert manager.has_middleware("llm_request")
"""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"subprocess failed ({result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
