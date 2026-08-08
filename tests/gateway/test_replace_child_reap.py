"""Tests for --replace child-process reaping (POSIX taskkill /T parity).

On Windows, ``terminate_pid(force=True)`` tree-kills via ``taskkill /T``.
On POSIX, ``--replace`` historically signalled only the recorded gateway PID,
so adapter subprocesses that survived their parent kept holding scoped token
locks and blocked the replacement gateway.  ``_snapshot_gateway_children`` /
``reap_gateway_children`` close that gap: the replacer snapshots the old
gateway's descendants while it is still alive, and reaps them (best-effort,
identity-aware) only after the main PID is confirmed dead.

Also asserts the takeover/reap machinery stays gated on explicit --replace.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway import status
from gateway.config import GatewayConfig


class _FakeChild:
    """Minimal psutil.Process stand-in for reap tests."""

    def __init__(self, pid, *, running=True, ppid=1, zombie=False):
        self.pid = pid
        self._running = running
        self._ppid = ppid
        self._zombie = zombie
        self.terminated = False
        self.killed = False

    def is_running(self):
        return self._running

    def status(self):
        return "zombie" if self._zombie else "sleeping"

    def ppid(self):
        return self._ppid

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def _fake_psutil(monkeypatch, *, wait_gone=None, wait_alive=None):
    """Install a stub psutil module for gateway.status's local imports."""
    fake = MagicMock()
    fake.STATUS_ZOMBIE = "zombie"
    fake.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    fake.wait_procs = MagicMock(
        side_effect=lambda live, timeout: (
            wait_gone if wait_gone is not None else list(live),
            wait_alive if wait_alive is not None else [],
        )
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)
    return fake


class TestReapGatewayChildren:
    def test_reaps_orphaned_children_sigterm_then_wait(self, monkeypatch):
        fake = _fake_psutil(monkeypatch)
        orphans = [_FakeChild(101, ppid=1), _FakeChild(102, ppid=1)]

        reaped = status.reap_gateway_children(orphans, parent_pid=42)

        assert reaped == 2
        assert all(c.terminated for c in orphans)
        assert not any(c.killed for c in orphans)
        fake.wait_procs.assert_called_once()

    def test_survivors_of_sigterm_get_sigkill(self, monkeypatch):
        stubborn = _FakeChild(103, ppid=1)
        _fake_psutil(monkeypatch, wait_gone=[], wait_alive=[stubborn])

        reaped = status.reap_gateway_children([stubborn], parent_pid=42)

        assert stubborn.terminated
        assert stubborn.killed
        assert reaped == 1


class TestSnapshotGatewayChildren:
    def test_snapshot_walks_descendants_recursively(self, monkeypatch):
        fake = _fake_psutil(monkeypatch)
        kids = [_FakeChild(201), _FakeChild(202)]
        fake.Process.return_value.children.return_value = kids

        assert status._snapshot_gateway_children(42) == kids
        fake.Process.assert_called_once_with(42)
        fake.Process.return_value.children.assert_called_once_with(recursive=True)


class TestScopedLockTakeoverReapsChildren:
    """take_over_scoped_lock_holder reaps the dead owner's orphans (POSIX)."""

    @staticmethod
    def _owner_record(target_home: Path, *, pid: int = 4242, start_time: int = 123):
        target_home.mkdir(parents=True, exist_ok=True)
        record = {
            "pid": pid,
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway", "run"],
            "start_time": start_time,
            "hermes_home": str(target_home),
        }
        (target_home / "gateway.pid").write_text(json.dumps(record))
        return record

    def _verified_owner_env(self, tmp_path, monkeypatch, *, alive_polls):
        replacer_home = tmp_path / "replacer"
        target_home = tmp_path / "target"
        replacer_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(replacer_home))
        record = self._owner_record(target_home)
        alive = iter(alive_polls)
        monkeypatch.setattr(status, "_pid_exists", lambda _pid: next(alive))
        monkeypatch.setattr(status, "_get_process_start_time", lambda _pid: 123)
        monkeypatch.setattr(
            status,
            "_read_process_cmdline",
            lambda _pid: "python -m hermes_cli.main gateway run",
        )
        return record

    def test_successful_takeover_snapshots_then_reaps(self, tmp_path, monkeypatch):
        record = self._verified_owner_env(
            tmp_path, monkeypatch, alive_polls=[True, True, False]
        )
        kids = [_FakeChild(301, ppid=1)]
        events = []
        monkeypatch.setattr(
            status,
            "_snapshot_gateway_children",
            lambda pid: events.append(("snapshot", pid)) or kids,
        )
        monkeypatch.setattr(
            status,
            "reap_gateway_children",
            lambda children, *, parent_pid, timeout=5.0: events.append(
                ("reap", parent_pid, children)
            )
            or len(children),
        )
        monkeypatch.setattr(
            status,
            "terminate_pid",
            lambda pid, *, force=False: events.append(("terminate", pid, force)),
        )

        assert status.take_over_scoped_lock_holder(record, graceful_attempts=1) == 4242
        # Snapshot taken while owner alive, BEFORE terminate; reap after exit.
        assert events == [
            ("snapshot", 4242),
            ("terminate", 4242, False),
            ("reap", 4242, kids),
        ]


@pytest.mark.asyncio
async def test_start_gateway_replace_reaps_old_gateway_children_posix(
    monkeypatch, tmp_path
):
    """--replace snapshots the old gateway's children before SIGTERM and
    reaps them after the main PID is confirmed dead (POSIX path)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    events = []
    kids = [_FakeChild(401, ppid=1)]

    class _CleanExitRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.exit_code = None
            self.adapters = {}

        async def start(self):
            assert self._platform_lock_takeover_on_start is True
            return True

        async def stop(self):
            return None

    _pid_state = {"alive": True}
    monkeypatch.setattr(
        "gateway.status.get_running_pid",
        lambda: 42 if _pid_state["alive"] else None,
    )
    monkeypatch.setattr(
        "gateway.status.remove_pid_file",
        lambda: _pid_state.update(alive=False),
    )
    monkeypatch.setattr(
        "gateway.status.release_all_scoped_locks", lambda **kwargs: 0
    )
    monkeypatch.setattr(
        "gateway.status._snapshot_gateway_children",
        lambda pid: events.append(("snapshot", pid)) or kids,
    )
    monkeypatch.setattr(
        "gateway.status.reap_gateway_children",
        lambda children, *, parent_pid, timeout=5.0: events.append(
            ("reap", parent_pid, children)
        )
        or len(children),
    )

    def _mock_terminate_pid(pid, force=False):
        events.append(("terminate", pid, force))
        _pid_state["alive"] = False

    monkeypatch.setattr("gateway.status.terminate_pid", _mock_terminate_pid)
    monkeypatch.setattr(
        "gateway.status._pid_exists", lambda pid: _pid_state["alive"]
    )
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr(
        "hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path
    )
    monkeypatch.setattr(
        "hermes_logging._add_rotating_handler", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("gateway.run.GatewayRunner", _CleanExitRunner)

    from gateway.run import start_gateway

    ok = await start_gateway(config=GatewayConfig(), replace=True, verbosity=None)

    assert ok is True
    # Snapshot precedes the SIGTERM; reap runs only after the PID is dead.
    assert events == [
        ("snapshot", 42),
        ("terminate", 42, False),
        ("reap", 42, kids),
    ]


