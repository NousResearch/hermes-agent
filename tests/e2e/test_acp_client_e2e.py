"""E2E smoke test for the ACP client transport.

Spawns a real ACP-compliant server (claude-acp-bridge) and performs one
complete prompt roundtrip through ACPClient + ACPClientSession.

Gated behind the HERMES_E2E_ACP environment variable — must be set to "1"
(or any non-empty truthy value) to run. Requires:
  - ``claude-acp-bridge`` installed and on PATH (pip install claude-acp-bridge)
  - A working Claude CLI configured for the subprocess (model auth token)

Run:
    HERMES_E2E_ACP=1 pytest tests/e2e/test_acp_client_e2e.py -v

Why this test:
  The unit tests (test_acp_client.py, test_acp_client_session.py) use fake
  subprocesses and mock clients. This smoke test validates the actual
  JSON-RPC wire protocol against a real ACP-compliant server implementation,
  catching encoding bugs, missing fields, and protocol-version mismatches.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

# Guard: only run when explicitly enabled
_E2E_ENABLED = bool(os.environ.get("HERMES_E2E_ACP", "").strip())

pytestmark = pytest.mark.skipif(
    not _E2E_ENABLED,
    reason=(
        "ACP E2E test disabled. Set HERMES_E2E_ACP=1 to enable. "
        "Requires claude-acp-bridge + Claude CLI auth."
    ),
)

# Path to the claude-acp-bridge package (fallback to installed binary)
_BRIDGE_PACKAGE_DIR = os.environ.get(
    "HERMES_ACP_BRIDGE_DIR",
    "/Users/guillaume/dev/claude-acp-bridge",
)
_BRIDGE_COMMAND = os.environ.get("HERMES_ACP_BRIDGE_COMMAND", "claude-acp-bridge")


def _bridge_available() -> bool:
    """Return True if the bridge binary / module can be found."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import claude_acp_bridge"],
            cwd=_BRIDGE_PACKAGE_DIR,
            capture_output=True,
            timeout=5,
        )
        return proc.returncode == 0
    except Exception:
        return False


@pytest.fixture(scope="module")
def bridge_command() -> list[str]:
    """Return the command to spawn the ACP server for testing."""
    if os.path.isdir(_BRIDGE_PACKAGE_DIR):
        return [sys.executable, "-m", "claude_acp_bridge"]
    return [_BRIDGE_COMMAND]


class TestACPClientE2E:
    def test_initialize_handshake(self, bridge_command: list[str]) -> None:
        """Verify the wire protocol: initialize returns valid InitializeResponse."""
        from agent.transports.acp_client import ACPClient

        with ACPClient(command=bridge_command[0], args=bridge_command[1:]) as client:
            result = client.initialize(
                client_name="hermes-e2e-test",
                client_version="0.1",
                timeout=10.0,
            )
            assert result is not None
            # ACP spec: InitializeResponse must include protocolVersion
            assert "protocolVersion" in result or result.get("protocol_version") is not None

    def test_session_new_and_close(self, bridge_command: list[str]) -> None:
        """Verify session/new returns a sessionId and session/close succeeds."""
        from agent.transports.acp_client_session import ACPClientSession

        with ACPClientSession(command=bridge_command[0], args=bridge_command[1:]) as session:
            sid = session.ensure_started(cwd=os.getcwd())
            assert sid
            assert len(sid) >= 8  # UUID format

    def test_prompt_roundtrip(self, bridge_command: list[str]) -> None:
        """Full roundtrip: session/new → session/prompt → text response."""
        from agent.transports.acp_client_session import ACPClientSession

        deltas: list[str] = []

        with ACPClientSession(
            command=bridge_command[0],
            args=bridge_command[1:],
            on_delta=deltas.append,
        ) as session:
            result = session.run_turn(
                "Reply with exactly: HERMES_ACP_OK",
                cwd=os.getcwd(),
                turn_timeout=120.0,
            )

        assert result.error is None, f"Turn error: {result.error}"
        assert result.should_retire is False
        # Either the streamed text or the final_text should contain our marker
        full_text = result.final_text + "".join(deltas)
        assert "HERMES_ACP_OK" in full_text, (
            f"Expected marker not found. final_text={result.final_text!r} "
            f"deltas={deltas!r}"
        )
