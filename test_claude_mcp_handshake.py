#!/usr/bin/env python3
import asyncio
import os
import json
import logging
import time

# Configure logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_mcp_handshake_subprocess():
    logger.info("Starting Claude MCP handshake test with asyncio.create_subprocess_exec and 'script'")

    claude_cli_path = os.getenv("CLAUDE_CLI_PATH", "/Users/tusker/.local/bin/claude") # Adjust if claude is in PATH
    command_args = [claude_cli_path, "mcp", "serve"]
    script_command = " ".join(command_args)

    # Use 'script -q -e -c' to provide a pseudo-TTY which claude mcp serve sometimes needs
    # /dev/null is often used as the file argument to 'script' when not saving output.
    proc = await asyncio.create_subprocess_exec(
        'script', '-q', '-e', '-c', script_command, '/dev/null',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=dict(os.environ, CLAUDE_CODE_SIMPLE='1', TERM='xterm-256color'),
    )

    logger.debug(f"Subprocess started with PID: {proc.pid}")

    async def _read_stream(stream, name):
        buffer = b""
        while True:
            try:
                data = await stream.read(4096)
                if not data:
                    break
                buffer += data
                # Process lines from buffer
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    logger.debug(f"{name} RECV: {line!r}")
                    if line.startswith(b"{"): # Attempt to parse JSON
                        return json.loads(line.decode('utf-8'))
            except Exception as e:
                logger.error(f"Error reading from {name} stream: {e}")
                break
        return None

    # Wait a moment for `claude mcp serve` to start up and potentially print its own logs
    # We'll try to drain startup output without blocking.
    logger.debug("Draining initial stdout/stderr from subprocess...")
    startup_drain_task = asyncio.create_task(_read_stream(proc.stdout, "SUBPROCESS_STDOUT"))
    stderr_drain_task = asyncio.create_task(_read_stream(proc.stderr, "SUBPROCESS_STDERR"))
    await asyncio.sleep(1) # Give it 1 second to print anything

    # Check if the process is still alive by polling its return code
    # if proc.returncode is not None:
    #     logger.error(f"Claude MCP serve exited prematurely with code {proc.returncode} during startup.")
    #     if await stderr_drain_task:
    #         logger.error(f"Stderr output from early exit: {await stderr_drain_task}")
    #     else:
    #         logger.error("No stderr for early exit.")
    #     return False

    # Construct the initialize JSON-RPC request
    initialize_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26", # Important: Match Claude CLI protocol version
            "capabilities": {},
            "clientInfo": {"name": "TestClient", "version": "1.0"},
        },
    }
    initialize_json = json.dumps(initialize_request) + "\n"
    logger.info(f"Sending initialize request: {initialize_json.strip()}")

    try:
        proc.stdin.write(initialize_json.encode('utf-8'))
        await proc.stdin.drain()
        logger.info("Initialize request sent.")

        # Read response from stdout
        response_task = asyncio.create_task(_read_stream(proc.stdout, "MCP_RESPONSE"))
        response = await asyncio.wait_for(response_task, timeout=5) # Wait for 5 seconds for a response

        if response:
            logger.info(f"Received MCP response: {response}")
            if response.get("id") == 1 and "result" in response:
                logger.info("Successfully received 'initialize' response!")
                return True
            else:
                logger.error(f"Received unexpected response to initialize: {response}")
                return False
        else:
            logger.error("Did not receive a response from Claude MCP serve within timeout.")
            #logger.error(f"Remaining stdout: {proc.stdout.read()!r}")
            #logger.error(f"Remaining stderr: {proc.stderr.read()!r}")
            return False

    except asyncio.TimeoutError:
        logger.error("Timed out waiting for MCP initialize response.")
        return False
    except Exception as e:
        logger.error(f"Error during MCP communication: {e}")
        return False
    finally:
        logger.info("Terminating subprocess.")
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=5)
        except Exception as e:
            logger.warning(f"Error terminating process: {e}")
        logger.info(f"Subprocess exited with code: {proc.returncode}")

if __name__ == "__main__":
    result = asyncio.run(test_mcp_handshake_subprocess())
    logger.info(f"Test result: {result}")
    if not result:
        print("\nFAILURE: The Claude MCP handshake test failed.")
        print("Please check the logs above for errors from 'claude mcp serve' or communication issues.")
        print("Ensure 'claude' CLI is in your PATH or set CLAUDE_CLI_PATH environment variable.")
    else:
        print("\nSUCCESS: The Claude MCP handshake test seems to have worked.")
