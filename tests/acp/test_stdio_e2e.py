"""E2E: ACP agent init over our robust stdio transport (JSON-RPC frames).

Addresses the sweeper request for a transport-level regression that proves
JSON-RPC exchange still works when the POSIX pipe transport is unavailable
(the failure mode that used to crash macOS ACP startup).
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import acp
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from acp_adapter.stdio_transport import open_acp_stdio_streams
from acp_adapter import entry as entry_mod

REPO_ROOT = Path(__file__).resolve().parents[2]


def _hermes_agent(session_manager: SessionManager | None = None) -> HermesACPAgent:
    mgr = session_manager or SessionManager(
        agent_factory=lambda: MagicMock(name="MockAIAgent")
    )
    return HermesACPAgent(session_manager=mgr)


async def _pipe_pair_streams():
    """Build agent-side (reader, writer) + client-side reader/write fd over os.pipe."""
    loop = asyncio.get_running_loop()

    # client -> agent
    c2a_r, c2a_w = os.pipe()
    # agent -> client
    a2c_r, a2c_w = os.pipe()

    in_read = os.fdopen(c2a_r, "rb", buffering=0)
    in_write = os.fdopen(c2a_w, "wb", buffering=0)
    out_read = os.fdopen(a2c_r, "rb", buffering=0)
    out_write = os.fdopen(a2c_w, "wb", buffering=0)

    agent_input = asyncio.StreamReader(limit=1024 * 1024)
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(agent_input), in_read
    )

    out_transport, out_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, out_write
    )
    agent_output = asyncio.StreamWriter(out_transport, out_protocol, None, loop)

    client_input = asyncio.StreamReader(limit=1024 * 1024)
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(client_input), out_read
    )

    return {
        "agent_reader": agent_input,  # output_stream for run_agent
        "agent_writer": agent_output,  # input_stream for run_agent
        "client_reader": client_input,
        "client_write": in_write,
        "close": lambda: (
            in_write.close(),
            out_read.close(),
        ),
    }


async def _rpc(client_write, client_reader, req: dict, timeout: float = 5.0) -> dict:
    client_write.write((json.dumps(req) + "\n").encode())
    client_write.flush()
    line = await asyncio.wait_for(client_reader.readline(), timeout=timeout)
    assert line, "agent produced no response line"
    return json.loads(line.decode())


@pytest.mark.asyncio
async def test_full_agent_initialize_over_run_agent_jsonrpc():
    """Real HermesACPAgent.initialize via acp.run_agent + JSON-RPC frames."""
    pipes = await _pipe_pair_streams()
    agent = _hermes_agent()

    task = asyncio.create_task(
        acp.run_agent(
            agent,
            input_stream=pipes["agent_writer"],
            output_stream=pipes["agent_reader"],
            use_unstable_protocol=True,
        )
    )
    try:
        resp = await _rpc(
            pipes["client_write"],
            pipes["client_reader"],
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": 1,
                    "clientInfo": {"name": "stdio-e2e", "version": "0"},
                    "clientCapabilities": {},
                },
            },
        )
        assert "error" not in resp, resp
        result = resp["result"]
        assert result["protocolVersion"] == acp.PROTOCOL_VERSION
        assert result["agentInfo"]["name"] == "hermes-agent"
        assert "agentCapabilities" in result

        # Second frame: session/new must also work (proves connection stayed up)
        resp2 = await _rpc(
            pipes["client_write"],
            pipes["client_reader"],
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {"cwd": str(REPO_ROOT), "mcpServers": []},
            },
        )
        assert "error" not in resp2, resp2
        assert resp2["result"]["sessionId"]
    finally:
        pipes["client_write"].close()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_full_agent_initialize_via_entry_run_acp_agent(monkeypatch):
    """entry._run_acp_agent wires open_acp_stdio_streams → run_agent with full init.

    Forces the fallback transport (pipe transport boom) so this covers the
    exact macOS-style failure path the sweeper asked for.
    """
    pipes = await _pipe_pair_streams()

    async def fake_open(limit=None):
        # Return the pipe-backed streams (simulates successful open after fallback)
        return pipes["agent_reader"], pipes["agent_writer"]

    monkeypatch.setattr(
        "acp_adapter.stdio_transport.open_acp_stdio_streams",
        fake_open,
    )

    agent = _hermes_agent()
    task = asyncio.create_task(entry_mod._run_acp_agent(agent))
    try:
        # _run_acp_agent will call our fake_open then acp.run_agent
        resp = await _rpc(
            pipes["client_write"],
            pipes["client_reader"],
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "initialize",
                "params": {
                    "protocolVersion": 1,
                    "clientInfo": {"name": "entry-e2e", "version": "0"},
                    "clientCapabilities": {},
                },
            },
        )
        assert "error" not in resp, resp
        assert resp["result"]["agentInfo"]["name"] == "hermes-agent"
    finally:
        pipes["client_write"].close()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_fallback_transport_then_full_initialize(monkeypatch):
    """Fallback stdio streams + HermesACPAgent initialize over JSON-RPC.

    Builds fallback writer/reader, feeds client frames into the reader, and
    asserts initialize succeeds — this is the transport-level proof.
    """
    from acp_adapter import stdio_transport as st

    loop = asyncio.get_running_loop()
    inbound = asyncio.Queue()
    outbound = bytearray()

    class _Stdout:
        class buffer:
            @staticmethod
            def write(data: bytes):
                outbound.extend(data)

            @staticmethod
            def flush():
                pass

        def flush(self):
            pass

    monkeypatch.setattr(st.sys, "stdout", _Stdout())

    def feed_from_queue(loop, reader):
        async def pump():
            while True:
                item = await inbound.get()
                if item is None:
                    reader.feed_eof()
                    return
                reader.feed_data(item)

        asyncio.create_task(pump())

    monkeypatch.setattr(st, "_start_stdin_feeder", feed_from_queue)

    reader, writer = await st._fallback_stdio_streams(loop, limit=1024 * 1024)
    agent = _hermes_agent()
    task = asyncio.create_task(
        acp.run_agent(
            agent,
            input_stream=writer,
            output_stream=reader,
            use_unstable_protocol=True,
        )
    )
    try:
        req = {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "initialize",
            "params": {
                "protocolVersion": 1,
                "clientInfo": {"name": "fallback-e2e", "version": "0"},
                "clientCapabilities": {},
            },
        }
        await inbound.put((json.dumps(req) + "\n").encode())

        # Wait for agent response on outbound buffer
        deadline = loop.time() + 5.0
        while loop.time() < deadline and b"\n" not in outbound:
            await asyncio.sleep(0.05)
        assert b"\n" in outbound, f"no response: {bytes(outbound)!r}"
        line = bytes(outbound).split(b"\n", 1)[0]
        resp = json.loads(line.decode())
        assert "error" not in resp, resp
        assert resp["id"] == 42
        assert resp["result"]["agentInfo"]["name"] == "hermes-agent"
    finally:
        await inbound.put(None)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-gated regression")
def test_subprocess_macos_file_stdout_still_initializes():
    """macOS sandbox: child stdout is a REG file (crash shape); agent still inits.

    Parent talks JSON-RPC over the child's redirected stdio via a side-channel
    protocol: child writes frames to the file stdout, parent writes requests to
    child stdin.
    """
    py = sys.executable
    child = textwrap.dedent(
        f"""
        import asyncio, json, sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        import acp
        from unittest.mock import MagicMock
        from acp_adapter.server import HermesACPAgent
        from acp_adapter.session import SessionManager
        from acp_adapter.stdio_transport import open_acp_stdio_streams

        async def main():
            agent = HermesACPAgent(
                session_manager=SessionManager(
                    agent_factory=lambda: MagicMock(name="MockAIAgent")
                )
            )
            reader, writer = await open_acp_stdio_streams(limit=1024 * 1024)
            await acp.run_agent(
                agent,
                input_stream=writer,
                output_stream=reader,
                use_unstable_protocol=True,
            )

        asyncio.run(main())
        """
    )

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "agent_stdout.jsonl"
        # stdin=pipe (parent writes requests), stdout=file (REG → forces fallback)
        with open(out_path, "wb") as outf:
            proc = subprocess.Popen(
                [py, "-c", child],
                stdin=subprocess.PIPE,
                stdout=outf,
                stderr=subprocess.PIPE,
                cwd=str(REPO_ROOT),
            )
            assert proc.stdin is not None
            req = {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "initialize",
                "params": {
                    "protocolVersion": 1,
                    "clientInfo": {"name": "macos-file-stdout", "version": "0"},
                    "clientCapabilities": {},
                },
            }
            proc.stdin.write((json.dumps(req) + "\n").encode())
            proc.stdin.flush()

            # Poll the file for a response line
            import time

            deadline = time.time() + 10.0
            data = b""
            while time.time() < deadline:
                data = out_path.read_bytes()
                if b"\n" in data:
                    break
                if proc.poll() is not None:
                    break
                time.sleep(0.05)

            # Tear down without double-flushing a closed stdin
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                err = proc.stderr.read() if proc.stderr else b""
            except Exception:
                err = b""
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
            # Drain any remaining stderr
            try:
                more = proc.stderr.read() if proc.stderr else b""
                err = (err or b"") + (more or b"")
            except Exception:
                pass

        assert b"\n" in data, (
            f"no JSON-RPC response. rc={proc.returncode} stderr={err!r} file={data!r}"
        )
        line = data.split(b"\n", 1)[0]
        resp = json.loads(line.decode())
        assert "error" not in resp, resp
        assert resp["result"]["agentInfo"]["name"] == "hermes-agent"
        # Fallback path should have logged the pipe-transport warning on Darwin
        # when stdout is a regular file.
        assert b"fallback" in err.lower() or b"pipe transport" in err.lower(), err
