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

import asyncio
import contextlib
import os
import signal
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
        assert client._supervised_pgids == {registered[0]}
    finally:
        await client._cleanup_process()


def test_release_unregisters_a_group_that_is_gone(tmp_path: Path, monkeypatch):
    """No survivors left → the registration is dropped (nothing stale to signal later)."""
    released: list[int] = []
    monkeypatch.setattr(client_module, "_unregister_process_group_with_parent_death_supervisor",
                        released.append)
    monkeypatch.setattr(client_module, "_process_group_alive", lambda pgid: False)

    client = _make_client(tmp_path)
    client._supervised_pgids = {4242}
    client._release_process_group_supervision(4242)

    assert released == [4242]
    assert client._supervised_pgids == set()


def test_release_keeps_coverage_when_a_member_survived_teardown(tmp_path: Path, monkeypatch):
    """A group with live members stays registered: the supervisor is the only thing left that can
    reap an escaped descendant.  The pgid stays on the client so a later spawn or teardown releases
    it once the group empties."""
    released: list[int] = []
    monkeypatch.setattr(client_module, "_unregister_process_group_with_parent_death_supervisor",
                        released.append)
    monkeypatch.setattr(client_module, "_process_group_alive", lambda pgid: True)

    client = _make_client(tmp_path)
    client._supervised_pgids = {4242}
    client._release_process_group_supervision(4242)

    assert released == []
    assert client._supervised_pgids == {4242}


@pytest.mark.asyncio
async def test_reentered_spawn_releases_the_superseded_attempts_group(tmp_path: Path, monkeypatch):
    """A second spawn cycle must not orphan a registration retained by the first.

    ``start()`` is documented re-call-to-retry and a failed handshake leaves it re-callable, so the
    first attempt's group can still be covered when a second attempt registers its own.  With a
    single overwritten slot the earlier pgid was lost: the supervisor kept it registered forever,
    and once that group emptied it was a stale, recyclable pgid — the residual
    process-group-reuse risk ``tools/mcp_death_supervisor.py`` documents.

    Assertions are on the supervisor's own record, never on client internals, so this same test
    body fails against the pre-fix implementation for the right reason.
    """
    supervised: set[int] = set()
    dead: set[int] = set()  # a group here has no members left; everything else counts as live
    monkeypatch.setattr(client_module, "_register_process_group_with_parent_death_supervisor",
                        lambda pgid: supervised.add(pgid) or True)
    monkeypatch.setattr(client_module, "_unregister_process_group_with_parent_death_supervisor",
                        supervised.discard)
    monkeypatch.setattr(client_module, "_process_group_alive", lambda pgid: pgid not in dead)

    client = _make_client(tmp_path)

    # Cycle 1: a member survives teardown, so coverage is retained -- exactly what the PR intends.
    await client._spawn()
    proc = client._proc
    assert proc is not None
    first = os.getpgid(proc.pid)
    await client._cleanup_process()
    assert supervised == {first}, "a group with a live member must stay covered"

    # That member now exits on its own: the group is empty while the registration is still there.
    dead.add(first)

    # Cycle 2: the retry registers its own group and must release the superseded registration.
    await client._spawn()
    proc = client._proc
    assert proc is not None
    second = os.getpgid(proc.pid)
    assert second != first
    assert second in supervised
    assert first not in supervised, (
        f"registration {first} was orphaned by the re-entered spawn; the supervisor would still "
        f"signal that pgid on parent death and could hit an unrelated recycled group")

    # Keep the second group covered through its own teardown for symmetry.
    await client._cleanup_process()


def test_reentry_sweep_releases_a_retained_group_that_since_emptied(tmp_path: Path, monkeypatch):
    """The retained pgid from a superseded attempt is released once its group has emptied."""
    released: list[int] = []
    monkeypatch.setattr(client_module, "_unregister_process_group_with_parent_death_supervisor",
                        released.append)

    client = _make_client(tmp_path)
    client._supervised_pgids = {4242, 4243}
    monkeypatch.setattr(client_module, "_process_group_alive", lambda pgid: pgid == 4243)

    client._release_dead_process_group_supervisions()

    assert released == [4242]
    assert client._supervised_pgids == {4243}


@pytest.mark.asyncio
async def test_cleanup_releases_supervision(tmp_path: Path, monkeypatch):
    """The graceful path (``shutdown`` → ``_cleanup_process``) drops coverage once it has reaped."""
    supervised: set[int] = set()
    dead: set[int] = set()
    monkeypatch.setattr(client_module, "_register_process_group_with_parent_death_supervisor",
                        lambda pgid: supervised.add(pgid) or True)
    monkeypatch.setattr(client_module, "_unregister_process_group_with_parent_death_supervisor",
                        supervised.discard)
    monkeypatch.setattr(client_module, "_process_group_alive", lambda pgid: pgid not in dead)

    client = _make_client(tmp_path)
    await client._spawn()
    proc = client._proc
    assert proc is not None
    pgid = os.getpgid(proc.pid)
    assert supervised == {pgid}
    dead.add(pgid)  # the kill during cleanup leaves nothing behind
    await client._cleanup_process()
    assert supervised == set()


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


# A launcher that forks a long-lived grandchild into its own (already new) session and exits, so the
# group outlives its leader -- the escaped-descendant case the retained-coverage branch exists for.
_ORPHANING_LAUNCHER = (
    "import subprocess, sys\n"
    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
)


@pytest.mark.asyncio
async def test_escaped_descendant_stays_covered_until_its_group_empties(tmp_path: Path, monkeypatch):
    """Retained coverage must survive teardown, then be released once the group is finally empty.

    Real processes (no mocks of the supervisor helpers): the leader exits immediately, leaving a
    grandchild in the group, so ``_cleanup_process`` finds the group alive and keeps it registered.
    Killing the survivor makes the group empty, and the next teardown must then drop the
    registration — the pgid is stale at that point and could be recycled.
    """
    registered: set[int] = set()
    released: set[int] = set()
    real_register = client_module._register_process_group_with_parent_death_supervisor
    real_unregister = client_module._unregister_process_group_with_parent_death_supervisor

    def _register(pgid: int) -> bool:
        ok = real_register(pgid)
        if ok:
            registered.add(pgid)
        return ok

    def _unregister(pgid: int) -> None:
        released.add(pgid)
        real_unregister(pgid)

    monkeypatch.setattr(client_module, "_register_process_group_with_parent_death_supervisor", _register)
    monkeypatch.setattr(client_module, "_unregister_process_group_with_parent_death_supervisor", _unregister)

    client = LSPClient(
        server_id="launcher", workspace_root=str(tmp_path),
        command=[sys.executable, "-c", _ORPHANING_LAUNCHER], env={}, cwd=str(tmp_path),
    )
    await client._spawn()
    proc = client._proc
    assert proc is not None
    pgid = os.getpgid(proc.pid)
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=5.0)  # the launcher itself exits at once

    try:
        # The descendant is alive: the group outlives teardown and must stay covered.
        assert client_module._process_group_alive(pgid), "the grandchild should still hold the group"
        await client._cleanup_process()
        assert pgid in registered and pgid not in released

        # The descendant now exits on its own: the next teardown releases the stale registration.
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(pgid, signal.SIGKILL)  # windows-footgun: ok — POSIX-only test module
        deadline = time.monotonic() + 10.0
        while client_module._process_group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not client_module._process_group_alive(pgid)

        await client._cleanup_process()
        assert pgid in released, "an emptied retained group must be unregistered, not left stale"
        assert pgid not in client._supervised_pgids
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            if client_module._process_group_alive(pgid):
                os.killpg(pgid, signal.SIGKILL)  # windows-footgun: ok — POSIX-only test module
