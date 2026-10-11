"""Provider discovery must not need network SDKs.

The PM worker runtime has no network SDKs. Each worker imports hermes_cli.config, which runs provider
discovery, so a top-level SDK import in a bundled provider prints one warning for each worker.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BLOCKED = ("httpx", "openai", "anthropic", "requests", "aiohttp", "boto3", "botocore", "google")

_DISCOVER = """
import json, sys
blocked = set(json.loads(sys.argv[1]))

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in blocked:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

if blocked:
    sys.meta_path.insert(0, _Block())
import providers
providers.list_providers()
print(json.dumps(sorted(providers._REGISTRY)))
"""


def _discover(tmp_path, blocked):
    home = tmp_path / ("home-blocked" if blocked else "home-full")
    home.mkdir()
    env = {"HERMES_HOME": str(home), "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-c", _DISCOVER, json.dumps(list(blocked))], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1]), done.stderr


def test_every_bundled_provider_registers_without_network_sdks(tmp_path):
    full, _ = _discover(tmp_path, ())
    blocked, stderr = _discover(tmp_path, BLOCKED)
    assert "Failed to load" not in stderr, stderr
    assert blocked == full, f"providers lost without SDKs: {sorted(set(full) - set(blocked))}"
