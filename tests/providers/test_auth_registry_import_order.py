"""Regression coverage for provider auth registration during TUI imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_tui_import_exposes_auth_registry_to_provider_plugins(tmp_path):
    """Provider discovery must not see a partially initialized auth module."""
    hermes_home = tmp_path / ".hermes"
    plugin_dir = hermes_home / "plugins" / "model-providers" / "import-order-probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "\n"
        "profile = ProviderProfile(\n"
        "    name='import-order-probe',\n"
        "    display_name='Profile fallback',\n"
        "    env_vars=('IMPORT_ORDER_PROBE_KEY',),\n"
        "    base_url='https://profile.example/v1',\n"
        "    auth_type='api_key',\n"
        ")\n"
        "register_provider(profile)\n"
        "\n"
        "from hermes_cli.auth import PROVIDER_REGISTRY, ProviderConfig\n"
        "PROVIDER_REGISTRY['import-order-probe'] = ProviderConfig(\n"
        "    id='import-order-probe',\n"
        "    name='Plugin injection',\n"
        "    auth_type='api_key',\n"
        "    inference_base_url='https://plugin.example/v1',\n"
        "    api_key_env_vars=('IMPORT_ORDER_PROBE_KEY',),\n"
        ")\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_PROFILE", None)
    env["PYTHONPATH"] = os.pathsep.join([
        str(REPO_ROOT),
        env.get("PYTHONPATH", ""),
    ]).rstrip(os.pathsep)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST; "
            "from hermes_cli.auth import PROVIDER_REGISTRY; "
            "cfg = PROVIDER_REGISTRY['import-order-probe']; "
            "assert cfg.name == 'Plugin injection', cfg; "
            "assert cfg.inference_base_url == 'https://plugin.example/v1', cfg; "
            "assert 'IMPORT_ORDER_PROBE_KEY' in _HERMES_PROVIDER_ENV_BLOCKLIST",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "partially initialized module 'hermes_cli.auth'" not in probe.stderr
