"""Buzz ACP credentials cross the private owner boundary without becoming daemon credentials."""
import asyncio
from contextlib import AsyncExitStack
import json
import shlex
import sys
from threading import Barrier

import pytest

from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.gateway.fixtures.authority_controls_peer import ModelPeer
from tests.waiting import async_wait_until


@pytest.fixture
def daemon_env():
    return {
        "BUZZ_PRIVATE_KEY": "fixture-daemon",
        "BUZZ_MANAGED_AGENT": "fixture-desktop-marker",
        "BUZZ_AMBIENT_ONLY": "fixture-daemon-only",
    }


class ConcurrentTerminalPeer(ModelPeer):
    def do_GET(self):
        if self.path != "/terminal-rendezvous":
            return super().do_GET()
        # The terminal children overlap, independent of the order of model requests.
        self.server.turns.wait(timeout=15)
        self.send_response(204)
        self.end_headers()


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_buzz_acp_terminal_environment_is_private_to_each_turn(daemon, tmp_path, model_peer):
    model_peer.RequestHandlerClass = ConcurrentTerminalPeer
    model_peer.turns = Barrier(4)
    # The test model gets only the success from the command, never the fixture values.
    probe = tmp_path / "inspect_env.py"
    probe.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "from urllib.request import urlopen\n"
        f"with urlopen('http://127.0.0.1:{model_peer.server_port}/terminal-rendezvous', timeout=20):\n"
        "    Path('observed.json').write_text(json.dumps({\n"
        "        k: v for k, v in os.environ.items() if k.startswith('BUZZ_')}))\n",
        encoding="utf-8",
    )
    model_peer.command = shlex.join([sys.executable, str(probe)])
    home, _, _, _ = daemon
    (home / ".env").write_text("BUZZ_PRIVATE_KEY=fixture-dotenv\n", encoding="utf-8")

    async with AsyncExitStack() as stack:
        ws = await stack.enter_async_context(viewer(daemon))
        prompts = []
        cases = []
        for identity in ("alpha", "bravo", "editor"):
            workspace = tmp_path / identity
            workspace.mkdir()
            credentials = {
                "BUZZ_PRIVATE_KEY": f"fixture-key-{identity}",
                "BUZZ_AUTH_TAG": f"fixture-tag-{identity}",
                "BUZZ_RELAY_URL": f"wss://{identity}.example.invalid",
                "BUZZ_ACP_AGENT_OWNER": f"fixture-owner-{identity}",
            }
            acp = await stack.enter_async_context(editor(daemon, workspace, launch_env=credentials))
            name = "example-editor" if identity == "editor" else "buzz-acp"
            assert "result" in await acp.rpc("initialize", protocolVersion=2, clientCapabilities={},
                                            clientInfo={"name": name, "version": "1.0.0"})
            created = await acp.rpc("session/new", cwd=str(workspace), mcpServers=[])
            assert "result" in created, created
            prompts.append((acp, created["result"]["sessionId"]))
            cases.append((workspace, {} if identity == "editor" else credentials))
        cli = tmp_path / "cli"
        cli.mkdir()
        cli_id = (await ws.rpc("session.create", cwd=str(cli), source="cli", request_id="cli-buzz-probe"))["session_id"]
        cases.append((cli, {}))
        accepted = await ws.rpc("prompt.submit", session_id=cli_id, input_id="observe", text="Observe the environment")
        results = await asyncio.gather(*(acp.rpc("session/prompt", sessionId=sid,
            prompt=[{"type": "text", "text": "Observe the environment"}]) for acp, sid in prompts))
        assert all(reply.get("result", {}).get("stopReason") == "end_turn" for reply in results), results

        async def cli_finished():
            receipt = await ws.rpc("prompt.receipt", session_id=cli_id, admission_id=accepted["admission_id"])
            return receipt if receipt["status"] == "terminal" else None

        assert (await async_wait_until(cli_finished))["outcome"] == "completed"
        for workspace, expected in cases:
            assert json.loads((workspace / "observed.json").read_text()) == expected
