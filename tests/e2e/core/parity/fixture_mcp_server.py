"""Tiny stdio MCP server used by the entrypoint-parity suite.

Exposes one tool, ``parity_canary``, whose result is the value of the
``PARITY_MCP_CANARY`` env var — so a test can prove a REAL tool call round-tripped
through the real Hermes MCP client (discovery, transport, liveness checks, result
plumbing) instead of a registered-but-dead schema.

``PARITY_MCP_SPAWN_GRANDCHILD=1`` makes the server fork a long-lived grandchild at
startup (the shape of npx/uvx wrappers and servers with worker helpers) so
shutdown tests can prove Hermes reaps the whole process tree, not just its direct
child. ``PARITY_MCP_DEATH_TOOL=1`` adds ``parity_die``, which kills the server while its
own call is in flight, leaving a helper holding the stdio pipe open so no EOF ever
arrives (the fast-death supervisor's target shape, #81995).
Every PID the server owns is appended to ``PARITY_MCP_PID_LOG`` so the test
can check liveness of exactly the processes this fixture created.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time


def _log_pid(kind: str, pid: int) -> None:
    path = os.environ.get("PARITY_MCP_PID_LOG")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{kind} {pid}\n")


def _spawn_grandchild() -> None:
    # Inherits the environment (and so the orphan-scan tag). Reads nothing,
    # writes nothing, lives until killed: an orphan if the tree is not reaped.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time\nwhile True: time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _log_pid("grandchild", child.pid)


def main() -> None:
    from mcp.server import MCPServer

    _log_pid("server", os.getpid())
    if os.environ.get("PARITY_MCP_SPAWN_GRANDCHILD") == "1":
        _spawn_grandchild()

    server = MCPServer("parity")

    @server.tool()
    def parity_canary(nonce: str = "") -> str:
        """Return the parity fixture canary (echoing the caller's nonce)."""
        return f"{os.environ.get('PARITY_MCP_CANARY', 'NO-CANARY')}:{nonce}"

    if os.environ.get("PARITY_MCP_DEATH_TOOL") == "1":
        @server.tool()
        def parity_die(nonce: str = "") -> str:
            """Crash this MCP server mid-call: the RPC never gets a response."""
            # A helper that inherits our stdio keeps the pipe open after we die (the
            # npx/uvx-wrapper shape), so the client never sees EOF: only the child
            # liveness watch can end the call.
            holder = subprocess.Popen([sys.executable, "-c", "import time\nwhile True: time.sleep(60)"])
            _log_pid("pipe_holder", holder.pid)
            _log_pid("dying_server", os.getpid())
            threading.Timer(0.3, lambda: os._exit(3)).start()
            time.sleep(3600)
            return "unreachable"

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
