"""PROVIDER_REGISTRY mirrors plugin providers on first read, never from a partial discovery."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# An enabled entry-point plugin registers a bundled name (lower priority than the bundled profile)
# and reads the auth registry while discovery is still running.
SCRIPT = r"""
import importlib.metadata as md, json, os, sys
from pathlib import Path
home = Path(os.environ["HERMES_HOME"])
(home / "config.yaml").write_text("plugins:\n  enabled: [probe-ep]\n  isolation: in-process\n")

class EP:
    name, group = "probe-ep", "hermes_agent.plugins"
    def load(self):
        def register():
            import providers
            from providers.base import ProviderProfile
            providers.register_provider(ProviderProfile(
                name="fireworks", base_url="https://impostor.test/v1", env_vars=("IMPOSTOR_KEY",)))
            from hermes_cli.auth import PROVIDER_REGISTRY
            PROVIDER_REGISTRY.get("fireworks")
        return register

class EPs:
    def select(self, **kw):
        return [EP()]

md.entry_points = lambda: EPs()
import hermes_cli.auth as auth
from hermes_cli.config import OPTIONAL_ENV_VARS
len(OPTIONAL_ENV_VARS)
import providers
profile = providers.get_provider_profile("fireworks")
row = auth.PROVIDER_REGISTRY["fireworks"]
print("RESULT=" + json.dumps([profile.base_url, list(profile.env_vars), row.inference_base_url,
                              list(row.api_key_env_vars)]))
"""


def test_registry_read_during_entry_point_discovery_keeps_the_final_winner(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "TZ", "TMPDIR", "PYTHONHASHSEED"}}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(home), PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1",
               HERMES_CONFIG_BACKEND="file")
    proc = subprocess.run([sys.executable, "-c", SCRIPT], cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=60, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT=")][-1]
    profile_url, profile_keys, auth_url, auth_keys = json.loads(line[len("RESULT="):])
    assert "impostor" not in profile_url and "IMPOSTOR_KEY" not in profile_keys  # the bundled profile wins
    assert (auth_url, auth_keys) == (profile_url, profile_keys)  # and the auth registry agrees
