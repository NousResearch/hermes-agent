"""Import-hygiene guard: ``import gateway`` must not require httpx.

The gateway package is imported by minimal consumers that install only the
lightweight wire deps (websockets/aiohttp/pyyaml/requests) — notably the
gateway-gateway cross-repo live E2E suite, which drives
``gateway.relay.ws_transport`` from a venv without the full agent dependency
set. Any module reachable from ``import gateway`` that eagerly imports
``hermes_cli.auth`` (and therefore httpx) breaks every one of those consumers
at import time.

Regression: agent/conversation_compression.py grew a module-level
``from agent.auxiliary_client import AuxiliaryExplicitCancellation`` which
pulled in agent.credential_pool → hermes_cli.auth → httpx during
``import gateway``.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Runs in a fresh interpreter so previously-imported modules in the test
# process can't mask an eager import. A meta-path hook makes httpx (and the
# heavyweight SDK clients) unimportable, mimicking the minimal E2E venv.
_PROBE = """
import sys

BLOCKED = {"httpx", "openai", "anthropic"}

class _Blocker:
    def find_module(self, fullname, path=None):
        if fullname.split(".")[0] in BLOCKED:
            return self
        return None

    def load_module(self, fullname):
        raise ImportError(f"blocked by import-hygiene test: {fullname}")

sys.meta_path.insert(0, _Blocker())

from gateway.relay.ws_transport import WebSocketRelayTransport  # noqa: F401
import gateway  # noqa: F401

leaked = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
if leaked:
    raise SystemExit(f"blocked modules leaked into sys.modules: {leaked}")
print("import-hygiene-ok")
"""


def test_import_gateway_does_not_require_httpx():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        "`import gateway` dragged in a blocked heavyweight dependency "
        "(httpx/openai/anthropic). Minimal consumers (cross-repo E2E venvs) "
        "import the gateway package without the full agent deps — keep those "
        f"imports lazy.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "import-hygiene-ok" in result.stdout
