"""LSP servers die with the Hermes process that started them, however it dies.

``LSPClient`` starts each language server in its own session/process group so a killpg aimed at
the server can never reach Hermes. That also took the server's descendants out of Hermes's kill
domain: the server itself exits on stdin EOF when Hermes dies, but a descendant that neither holds
nor watches that pipe (a wrapper's background child, a SIGTERM-ignoring worker) outlived a
SIGKILLed / OOM-killed Hermes forever. ``_spawn`` now hands the group to the per-process
owner-death supervisor; teardown hands it back once the group is empty.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import psutil
import pytest

from agent.lsp import client as lsp_client
from agent.lsp.client import LSPClient
from hermes_cli import process_identity
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("posix")

MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")
REPO = str(Path(__file__).resolve().parents[3])


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _client(workspace: Path, script: str, **extra_env: str) -> LSPClient:
    env = {"MOCK_LSP_SCRIPT": script, "PYTHONPATH": os.environ.get("PYTHONPATH", ""), **extra_env}
    return LSPClient(server_id=f"mock-{script}", workspace_root=str(workspace),
                     command=[sys.executable, MOCK_SERVER], env=env, cwd=str(workspace))


@pytest.fixture
def recorded(monkeypatch):
    calls: list = []
    monkeypatch.setattr(lsp_client, "own_process_groups", lambda pgids: calls.append(("own", list(pgids))))
    monkeypatch.setattr(process_identity, "release_process_groups", lambda pgids: calls.append(("release", list(pgids))))
    monkeypatch.setattr(lsp_client, "register_child", lambda pid, purpose: calls.append(("ledger", pid, purpose)))
    return calls


@pytest.mark.asyncio
async def test_the_server_group_is_owned_on_spawn_and_released_on_shutdown(tmp_path, recorded):
    client = _client(tmp_path, "clean")
    await client.start()
    pid = client._proc.pid
    try:
        assert os.getpgid(pid) == pid  # its own group: a killpg of it can never reach Hermes
        assert recorded == [("own", [pid]), ("ledger", pid, "lsp-server")]
    finally:
        await client.shutdown()
    assert recorded[-1] == ("release", [pid])


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass  # the child reparents to init once its leader is killed
async def test_a_surviving_group_member_stays_owned(tmp_path, recorded, monkeypatch):
    """Teardown that could not empty the group must not drop coverage: whatever survived still
    has to die with Hermes."""
    monkeypatch.setattr("agent.deadline.kill_process_tree", lambda pid, **kw: True)  # kills nothing
    child_pid_file = tmp_path / "child.pid"
    client = _client(tmp_path, "slow_tree", MOCK_LSP_CHILD_PID=str(child_pid_file))
    await client.start()
    leader = client._proc.pid
    child = int(wait_until(lambda: child_pid_file.exists() and child_pid_file.read_text(encoding="utf-8"),
                           timeout=15, what="the server's SIGTERM-ignoring child to start"))
    try:
        monkeypatch.setattr(lsp_client, "SHUTDOWN_GRACE", 0.1)
        client._proc.kill()  # the leader goes; its child stays in the group
        await client.shutdown()
        assert not [c for c in recorded if c[0] == "release"]
    finally:
        for pid in (child, leader):
            try:
                os.kill(pid, 9)  # windows-footgun: ok — posix-marked test
            except ProcessLookupError:
                pass


def _owner(tmp_path: Path, script: str) -> tuple[subprocess.Popen, int, int]:
    """A real Hermes-like owner: starts an LSPClient on the mock server's ``slow_tree`` script (a
    leader plus a SIGTERM-ignoring child that never touches the stdio pipes) and prints both pids."""
    child_pid_file = tmp_path / "child.pid"
    ws = tmp_path / "ws"
    ws.mkdir()
    src = textwrap.dedent(f"""
        import asyncio, os, sys, time
        from agent.lsp.client import LSPClient
        async def main():
            c = LSPClient(server_id="mock", workspace_root={str(ws)!r}, command=[sys.executable, {MOCK_SERVER!r}],
                          env={{"MOCK_LSP_SCRIPT": {script!r}, "MOCK_LSP_CHILD_PID": {str(child_pid_file)!r}}})
            await c.start()
            while not os.path.exists({str(child_pid_file)!r}):
                await asyncio.sleep(0.05)
            print(c._proc.pid, open({str(child_pid_file)!r}, encoding="utf-8").read(), flush=True)
            await asyncio.sleep(300)
        asyncio.run(main())
    """)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HERMES_HOME=str(home), HOME=str(home), PYTHONPATH=REPO)
    owner = subprocess.Popen([sys.executable, "-c", src], env=env, cwd=REPO, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = owner.stdout.readline().split()
    if len(line) != 2:
        owner.kill()
        pytest.fail(f"owner did not start the server: {line!r} {owner.communicate(timeout=10)[1][-2000:]}")
    return owner, int(line[0]), int(line[1])


def _cleanup(owner: subprocess.Popen, *pids: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, 9)  # windows-footgun: ok — posix-marked test
        except ProcessLookupError:
            pass
    if owner.poll() is None:
        owner.kill()
    owner.communicate(timeout=10)
    process_identity._supervised_pgids.clear()


# The server tree reparents to init once its owner dies, so cleanup leaves this test's subtree.
@pytest.mark.live_system_guard_bypass
def test_a_sigkilled_owner_takes_its_whole_server_tree_with_it(tmp_path):
    """End to end through the real supervisor: the owner dies by SIGKILL (no Python cleanup runs).
    The leader exits on stdin EOF by itself; its SIGTERM-ignoring child, which holds no pipe, used
    to survive forever (measured on main: alive 30 s later, 3/3 trials)."""
    owner, leader, child = _owner(tmp_path, "slow_tree")
    try:
        owner.kill()
        owner.wait(timeout=10)
        wait_until(lambda: not _alive(leader) and not _alive(child), timeout=15,
                   what="the LSP server and its descendant to die with the kill -9'd Hermes process")
    finally:
        _cleanup(owner, leader, child)


@pytest.mark.live_system_guard_bypass
def test_a_live_owner_keeps_its_server_tree(tmp_path):
    """Control: while the owner lives, the supervisor touches nothing."""
    owner, leader, child = _owner(tmp_path, "slow_tree")
    try:
        supervisor = [p for p in psutil.Process(owner.pid).children() if "process_supervisor.sh" in " ".join(p.cmdline())]
        assert supervisor, "the owner should have started its owner-death supervisor"
        # Give a wrongly-firing supervisor its full TERM -> 3 s -> KILL window to act.
        with pytest.raises(AssertionError):
            wait_until(lambda: not _alive(leader) or not _alive(child), timeout=5, what="(control) a server death")
        assert _alive(leader) and _alive(child) and owner.poll() is None
    finally:
        _cleanup(owner, leader, child)
