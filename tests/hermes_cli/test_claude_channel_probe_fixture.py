from __future__ import annotations

import json
from pathlib import Path
import subprocess


FIXTURE = Path(__file__).parent / "fixtures" / "claude_channel_probe"


def test_fixture_pins_reviewed_sdk_and_declares_no_permission_relay():
    package = json.loads((FIXTURE / "package.json").read_text())
    source = (FIXTURE / "channel.mjs").read_text()

    assert package["dependencies"] == {"@modelcontextprotocol/sdk": "1.32.2"}
    assert '"claude/channel": {}' in source
    assert "claude/channel/permission" not in source


def test_fixture_binds_route_and_registers_before_listening():
    config = (FIXTURE / "mcp.json.template").read_text()
    source = (FIXTURE / "channel.mjs").read_text()
    contract = (FIXTURE / "contract.mjs").read_text()
    for name in (
        "TICKET_REF", "ASSIGNMENT_ID", "ATTEMPT_ID", "OWNER_IDENTITY",
        "TICKET_REVISION", "AUTHORITY_EVENT_SHA256", "CORRELATION_ID",
        "ROUTE_NONCE",
    ):
        assert f"ESN0407_{name}" in config
        assert f"ESN0407_{name}" in source + contract
    assert source.index("await mcp.connect") < source.index("localServer.listen")


def test_fixture_cleanup_destroys_clients_and_unlinks_owned_socket():
    source = (FIXTURE / "channel.mjs").read_text()
    contract = (FIXTURE / "contract.mjs").read_text()

    assert "for (const socket of activeSockets) socket.destroy()" in contract
    assert 'process.once("exit", () => unlinkOwnedSocket(socketPath))' in source
    assert "if (stat.isSocket()) fs.unlinkSync(socketPath)" in contract


def test_fixture_runtime_contract_suite_passes():
    result = subprocess.run(
        ["node", "--test", "contract.test.mjs"],
        cwd=FIXTURE,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
