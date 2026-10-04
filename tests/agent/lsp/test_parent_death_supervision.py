"""The language server must not outlive the Hermes process that spawned it.

``LSPClient._spawn`` starts the server with ``start_new_session=True`` (deliberately: it keeps
the gateway safe from MCP ``killpg`` sweeps), which means the kernel's parent-death semantics do
not apply — the server is session-detached, so it reparents to init and keeps running when Hermes
dies without its cleanup path (the ``os._exit`` exits: CLI exit watchdog, one-shot / kanban
worker, TUI hard exit).  Each spawned group is therefore registered with the process-wide
parent-death supervisor (``tools/mcp_death_supervisor.py``), the same one that already covers
stdio MCP servers, which kills a registered group when this process dies by any means.

Live orphan evidence this covers: ``evals/lsp_parent_death_orphan.py``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent.lsp import client as client_module
from agent.lsp.client import LSPClient

pytestmark = pytest.mark.platforms("posix")  # process groups and the supervisor are POSIX-only

_SLEEPER = "import time; time.sleep(120)"

# A process that registers its spawned server group, then dies the way the exit watchdog does.
_PARENT_PROGRAM = """
import os, subprocess, sys

from agent.lsp.client import _register_process_group_with_parent_death_supervisor

child = subprocess.Popen([sys.executable, "-c", {sleeper!r}], start_new_session=True)
registered = _register_process_group_with_parent_death_supervisor(os.getpgid(child.pid))
print(f"{{child.pid}} {{int(registered)}}", flush=True)
os._exit(0)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_gone(pid: int, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


def _make_client(workspace: Path) -> LSPClient:
    return LSPClient(
        server_id="sleeper",
        workspace_root=str(workspace),
        command=[sys.executable, "-c", _SLEEPER],
        env={},
        cwd=str(workspace),
    )


@pytest.mark.asyncio
async def test_spawn_registers_the_server_group_for_parent_death(tmp_path: Path, monkeypatch):
    """``_spawn`` hands the live server's process group to the supervisor."""
    registered: list[int] = []
    monkeypatch.setattr(
        client_module, "_register_process_group_with_parent_death_supervisor",
        lambda pgid: registered.append(pgid) or True,
    )

    client = _make_client(tmp_path)
    try:
        await client._spawn()
        proc = client._proc
        assert proc is not None
        assert registered == [os.getpgid(proc.pid)]
        assert client._supervised_pgid == registered[0]
    finally:
        await client._cleanup_process()


def test_release_unregisters_a_group_that_is_gone(tmp_path: Path, monkeypatch):
    """No survivors left → the registration is dropped (nothing stale to signal later)."""
    released: list[int] = []
    monkeypatch.setattr(client_module, "_unregister_process_group_with_parent_death_supervisor",
                        released.append)
    monkeypatch.setattr(client_module, "_process_group_alive", lambda pgid: False)

    client = _make_client(tmp_path)
    client._supervised_pgid = 4242
    client._release_process_group_supervision()

    assert released == [4242]
    assert client._supervised_pgid is None


def test_release_keeps_coverage_when_a_member_survived_teardown(tmp_path: Path, monkeypatch):
    """A group with live members stays registered: the supervisor is the only thing left that can
    reap an escaped descendant, and it prunes the registration itself once the group empties."""
    released: list[int] = []
    monkeypatch.setattr(client_module, "_unregister_process_group_with_parent_death_supervisor",
                        released.append)
    monkeypatch.setattr(client_module, "_process_group_alive", lambda pgid: True)

    client = _make_client(tmp_path)
    client._supervised_pgid = 4242
    client._release_process_group_supervision()

    assert released == []


@pytest.mark.asyncio
async def test_cleanup_releases_supervision(tmp_path: Path, monkeypatch):
    """The graceful path (``shutdown`` → ``_cleanup_process``) drops coverage once it has reaped."""
    released: list[int] = []
    monkeypatch.setattr(client_module, "_register_process_group_with_parent_death_supervisor",
                        lambda pgid: True)
    monkeypatch.setattr(client_module, "_unregister_process_group_with_parent_death_supervisor",
                        released.append)

    client = _make_client(tmp_path)
    await client._spawn()
    pgid = client._supervised_pgid
    assert pgid is not None
    await client._cleanup_process()
    assert released == [pgid]


def test_supervisor_reaps_the_group_when_the_owner_dies_ungracefully(tmp_path: Path):
    """End to end: a registered server group dies with its owner's ``os._exit``.

    Runs the real supervisor subprocess (no mocks): the owner registers a server group, exits via
    ``os._exit(0)`` the way the CLI exit watchdog does, and the server must not survive it.  This is
    the regression that left pyright processes reparented to init on a long-lived gateway.
    """
    program = _PARENT_PROGRAM.format(sleeper=_SLEEPER)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(Path(__file__).resolve().parents[3]), env.get("PYTHONPATH", "")) if p)
    proc = subprocess.run([sys.executable, "-c", program], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-2000:]

    pid_text, registered = proc.stdout.split()[-2:]
    server_pid = int(pid_text)
    assert registered == "1", "registration failed, so this proves nothing"

    try:
        assert _wait_gone(server_pid), (
            f"language server {server_pid} outlived its owner — the parent-death supervisor did "
            f"not reap the registered group")
    finally:
        if _alive(server_pid):  # never leave the failure behind
            os.kill(server_pid, 9)
