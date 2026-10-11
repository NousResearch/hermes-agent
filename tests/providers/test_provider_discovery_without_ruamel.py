"""Bundled provider discovery must not need ruamel.

``custom`` and ``openrouter`` import repo ``utils``, which pulled
``hermes_yaml`` (and ruamel) at module import. Any interpreter without
ruamel — the bare tools python, before bootstrap activates the app venv —
silently lost both providers from the registry (#134107).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


_DISCOVER = """
import importlib.abc
import json
import sys

sys.path.insert(0, sys.argv[1])

blocked = set(json.loads(sys.argv[2]))


class _Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


if blocked:
    sys.meta_path.insert(0, _Block())

import providers
providers.list_providers()
print(json.dumps(sorted(providers._REGISTRY)))
"""


def _discover(repo, tmp_path, blocked):
    home = tmp_path / ("home-blocked" if blocked else "home-full")
    home.mkdir()
    env = dict(os.environ, HOME=str(home), HERMES_HOME=str(home / "hermes"))
    done = subprocess.run(
        [sys.executable, "-I", "-B", "-c", _DISCOVER, str(repo), json.dumps(list(blocked))],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Failed to load" not in done.stderr, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_every_bundled_provider_registers_without_ruamel(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    full = _discover(repo, tmp_path, ())
    blocked = _discover(repo, tmp_path, ("ruamel",))
    assert blocked == full, f"providers lost without ruamel: {sorted(set(full) - set(blocked))}"
