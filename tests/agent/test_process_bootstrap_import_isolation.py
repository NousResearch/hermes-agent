"""Process HTTP helpers must be importable without launching Hermes.

A foreign service can reuse ``agent.process_bootstrap`` for its HTTP transport.  Importing that
library helper must not import ``hermes_bootstrap``: the latter is a launcher entry point whose
module body may select Hermes' managed interpreter and re-exec the process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_external_interpreter_http_helper_import_does_not_bootstrap(tmp_path):
    """A clean child can build the real HTTP helper without running the launcher bootstrap."""
    root = Path(__file__).resolve().parents[2]
    script = """
import json
import sys
from agent.process_bootstrap import build_keepalive_http_client

client = build_keepalive_http_client("https://example.invalid/v1")
if client is not None:
    client.close()
print(json.dumps({
    "bootstrap_loaded": "hermes_bootstrap" in sys.modules,
    "network_loaded": "hermes_network" in sys.modules,
}))
"""
    env = {
        "HOME": str(tmp_path),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(root),
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "HERMES_DISABLE_LAZY_INSTALLS": "1",
        "TMPDIR": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.splitlines()[-1])
    assert payload == {"bootstrap_loaded": False, "network_loaded": True}
