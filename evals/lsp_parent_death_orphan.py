#!/usr/bin/env python3
"""Repro: an LSP language server outlives the Hermes process that spawned it.

``LSPClient._spawn`` starts the language server with ``start_new_session=True``
(deliberate: it keeps the gateway's process group safe from MCP ``killpg``
sweeps).  The consequence is that the server is NOT in the parent's process
group and carries no parent-death signal, so every ``os._exit`` path in Hermes —
the CLI exit watchdog (``hermes_cli/cli_shutdown.py::_arm_exit_watchdog``), the
one-shot / kanban-worker exit (``hermes_cli/cli_single_query.py``), the TUI
gateway hard exit (``tui_gateway/entry.py::_hard_exit``) — leaves a live server
holding the whole workspace index in memory, reparented to PID 1.

``agent/lsp/__init__.py`` documented the opposite ("SIGKILL/os._exit skip atexit
— fine, the kernel reaps the stateless servers with their parent").  The kernel
reaps a child with its parent only while the child stays in the parent's process
group / session; a session-detached child is reparented and runs on.

Run it against a checkout:

    PYTHONPATH=<repo> python3 evals/lsp_parent_death_orphan.py

Exit code: 0 when the server is reaped with its parent, 1 when it survives.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# A stand-in language server: long-lived, dies only when killed.  A real pyright/gopls behaves
# identically here — the mechanism under test is the process-tree semantics of the spawn.
SERVER_SLEEP_S = 120
OBSERVE_S = 5.0

PARENT_PROGRAM = '''\
import asyncio, os, sys

from agent.lsp.client import LSPClient


async def main() -> int:
    client = LSPClient(
        server_id="repro-server",
        workspace_root=os.getcwd(),
        command=[sys.executable, "-c", "import time; time.sleep({sleep_s})"],
        env={{}},
        cwd=os.getcwd(),
    )
    await client._spawn()          # the real spawn path (start_new_session=True)
    print(f"PARENT {{os.getpid()}} SERVER {{client._proc.pid}}", flush=True)
    # How the CLI exit watchdog / one-shot / kanban worker exit:
    os._exit(0)


raise SystemExit(asyncio.run(main()))
'''


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _ppid(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as fh:
            return int(fh.read().split()[3])
    except (OSError, IndexError, ValueError):
        return None


def main() -> int:
    if sys.platform == "win32":
        print("POSIX-only repro (start_new_session is a no-op on Windows).")
        return 0

    with tempfile.TemporaryDirectory(prefix="lsp-orphan-") as workdir:
        parent_file = Path(workdir) / "orphan_parent.py"
        parent_file.write_text(PARENT_PROGRAM.format(sleep_s=SERVER_SLEEP_S), encoding="utf-8")

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if p)
        proc = subprocess.run(
            [sys.executable, str(parent_file)],
            cwd=workdir, env=env, capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            print("parent failed to run the spawn path")
            print("stdout:", proc.stdout)
            print("stderr:", proc.stderr[-2000:])
            return 2

        words = proc.stdout.split()
        server_pid = int(words[words.index("SERVER") + 1])
        print(f"parent exited via os._exit(0); server pid {server_pid}")

        time.sleep(OBSERVE_S)
        if not _alive(server_pid):
            print("RESULT: server reaped with its parent — correct.")
            return 0

        print(f"RESULT: server SURVIVED its parent; ppid={_ppid(server_pid)} "
              f"(1 = reparented to init / orphaned)")
        try:
            os.kill(server_pid, 9)  # clean up after ourselves
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
