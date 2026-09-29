"""A bare PM interpreter can probe the proxy without loading the app."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_native_proxy_probe_needs_no_application_yaml(tmp_path):
    script = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
sys.modules['ruamel'] = None
from pm.registry import get_package
result = get_package('iron-proxy')._probe_env()
assert 'MY_PRIVATE_TOKEN' not in result
assert 'agent.proxy_sources.iron_proxy' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        env={**os.environ, "HERMES_HOME": str(tmp_path), "MY_PRIVATE_TOKEN": "sentinel"},
        capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stderr
