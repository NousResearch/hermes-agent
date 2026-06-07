"""E2E smoke test for the ACP client transport.

Spawns a real ACP-compliant server and performs one complete prompt roundtrip
through ACPClient + ACPClientSession.

Gated behind two environment variables:
  - ``HERMES_E2E_ACP=1``           must be set to enable the tests
  - ``HERMES_E2E_ACP_COMMAND``     path or name of the ACP binary to spawn

There is no default binary — reviewers must explicitly supply their own
ACP-compliant server. For example, using the reference implementation:

    npm install -g @agentclientprotocol/claude-agent-acp

    HERMES_E2E_ACP=1 HERMES_E2E_ACP_COMMAND=claude-agent-acp \\
      pytest tests/e2e/test_acp_client_e2e.py -v

Additional args can be forwarded to the binary via ``HERMES_E2E_ACP_ARGS``
(space-separated):

    HERMES_E2E_ACP=1 HERMES_E2E_ACP_COMMAND=claude-agent-acp \\
      HERMES_E2E_ACP_ARGS="--stdio --model claude-sonnet-4-5" \\
      pytest tests/e2e/test_acp_client_e2e.py -v

Why this test:
  The unit tests (test_acp_client.py, test_acp_client_session.py) use fake
  subprocesses and mock clients. This smoke test validates the actual
  JSON-RPC wire protocol against a real ACP-compliant server implementation,
  catching encoding bugs, missing fields, and protocol-version mismatches.
"""

from __future__ import annotations

import os
import shlex
import subprocess

import pytest

# ---------------------------------------------------------------------------
# Guard: only run E2E tests when explicitly enabled
# ---------------------------------------------------------------------------

_E2E_ENABLED = bool(os.environ.get("HERMES_E2E_ACP", "").strip())

_E2E_SKIP_MARK = pytest.mark.skipif(
    not _E2E_ENABLED,
    reason=(
        "ACP E2E test disabled. Set HERMES_E2E_ACP=1 to enable. "
        "Also requires HERMES_E2E_ACP_COMMAND pointing to an ACP-compliant binary."
    ),
)

# ---------------------------------------------------------------------------
# ACP command resolution — explicit env var, no default
# ---------------------------------------------------------------------------

_ACP_COMMAND = os.environ.get("HERMES_E2E_ACP_COMMAND", "").strip()
_ACP_ARGS_RAW = os.environ.get("HERMES_E2E_ACP_ARGS", "").strip()
_ACP_EXTRA_ARGS: list[str] = shlex.split(_ACP_ARGS_RAW) if _ACP_ARGS_RAW else []


def _acp_command_available() -> bool:
    """Return True if HERMES_E2E_ACP_COMMAND is set and the binary responds."""
    if not _ACP_COMMAND:
        return False
    try:
        proc = subprocess.run(
            [_ACP_COMMAND, "--version"],
            capture_output=True,
            timeout=5,
        )
        return proc.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures (only used by E2E tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def acp_command() -> list[str]:
    """Return the command list to spawn the ACP server for testing.

    Skips the test module when:
    - ``HERMES_E2E_ACP_COMMAND`` is not set, or
    - the binary is not reachable on PATH.

    This produces a clear skip reason instead of a confusing FileNotFoundError
    at test time. Reviewers must supply their own ACP binary explicitly.
    """
    if not _ACP_COMMAND:
        pytest.skip(
            "HERMES_E2E_ACP_COMMAND is not set. "
            "Point it at an ACP-compliant binary to run E2E tests. "
            "Example: HERMES_E2E_ACP_COMMAND=claude-agent-acp"
        )
    if not _acp_command_available():
        pytest.skip(
            f"ACP binary not found or did not respond: {_ACP_COMMAND!r}. "
            "Ensure the binary is installed and on PATH."
        )
    return [_ACP_COMMAND] + _ACP_EXTRA_ARGS


# ---------------------------------------------------------------------------
# E2E tests — gated by _E2E_SKIP_MARK
# ---------------------------------------------------------------------------


@_E2E_SKIP_MARK
class TestACPClientE2E:
    def test_initialize_handshake(self, acp_command: list[str]) -> None:
        """Verify the wire protocol: initialize returns valid InitializeResponse."""
        from agent.transports.acp_client import ACPClient

        with ACPClient(command=acp_command[0], args=acp_command[1:]) as client:
            result = client.initialize(
                client_name="hermes-e2e-test",
                client_version="0.1",
                timeout=10.0,
            )
            assert result is not None
            # ACP spec: InitializeResponse must include protocolVersion
            assert "protocolVersion" in result or result.get("protocol_version") is not None

    def test_session_new_and_close(self, acp_command: list[str]) -> None:
        """Verify session/new returns a sessionId and session/close succeeds."""
        from agent.transports.acp_client_session import ACPClientSession

        with ACPClientSession(command=acp_command[0], args=acp_command[1:]) as session:
            sid = session.ensure_started(cwd=os.getcwd())
            assert sid
            assert len(sid) >= 8  # UUID format

    def test_prompt_roundtrip(self, acp_command: list[str]) -> None:
        """Full roundtrip: session/new → session/prompt → text response."""
        from agent.transports.acp_client_session import ACPClientSession

        deltas: list[str] = []

        with ACPClientSession(
            command=acp_command[0],
            args=acp_command[1:],
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


# ---------------------------------------------------------------------------
# Skip-message regression guard — runs unconditionally
# ---------------------------------------------------------------------------


def test_acp_command_available_returns_false_when_unset() -> None:
    """Confirm _acp_command_available() returns False when _ACP_COMMAND is empty.

    This test runs unconditionally to protect against regressions where the
    skip guard logic is accidentally removed. It patches the module-level
    variable to simulate a missing HERMES_E2E_ACP_COMMAND.
    """
    import tests.e2e.test_acp_client_e2e as e2e_mod

    original = e2e_mod._ACP_COMMAND
    try:
        e2e_mod._ACP_COMMAND = ""
        result = e2e_mod._acp_command_available()
        assert result is False, (
            "_acp_command_available() must return False when _ACP_COMMAND is empty; "
            "the E2E fixture relies on this to produce a clear skip message."
        )
    finally:
        e2e_mod._ACP_COMMAND = original
