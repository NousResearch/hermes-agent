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
        monkeypatch.setattr(status, "_IS_WINDOWS", False)
        fake = _fake_psutil(monkeypatch)
        orphans = [_FakeChild(101, ppid=1), _FakeChild(102, ppid=1)]

        reaped = status.reap_gateway_children(orphans, parent_pid=42)

        assert reaped == 2
        assert all(c.terminated for c in orphans)
        assert not any(c.killed for c in orphans)
        fake.wait_procs.assert_called_once()

    def test_survivors_of_sigterm_get_sigkill(self, monkeypatch):
        monkeypatch.setattr(status, "_IS_WINDOWS", False)
        stubborn = _FakeChild(103, ppid=1)
        _fake_psutil(monkeypatch, wait_gone=[], wait_alive=[stubborn])

        reaped = status.reap_gateway_children([stubborn], parent_pid=42)

        assert stubborn.terminated
        assert stubborn.killed
        assert reaped == 1

    def test_child_still_parented_to_live_parent_is_skipped(self, monkeypatch):
        """If a child's ppid still equals the old gateway PID, the parent is
        alive and the child is not an orphan — never signal it."""
        monkeypatch.setattr(status, "_IS_WINDOWS", False)
        _fake_psutil(monkeypatch)
        child = _FakeChild(104, ppid=42)

        reaped = status.reap_gateway_children([child], parent_pid=42)

        assert reaped == 0
        assert not child.terminated
        assert not child.killed

    def test_dead_and_zombie_children_are_skipped(self, monkeypatch):
        monkeypatch.setattr(status, "_IS_WINDOWS", False)
        _fake_psutil(monkeypatch)
        dead = _FakeChild(105, running=False)
        zombie = _FakeChild(106, zombie=True)

        assert status.reap_gateway_children([dead, zombie], parent_pid=42) == 0
        assert not dead.terminated and not zombie.terminated

    def test_noop_on_windows_and_empty_snapshot(self, monkeypatch):
        monkeypatch.setattr(status, "_IS_WINDOWS", True)
        child = _FakeChild(107, ppid=1)
        assert status.reap_gateway_children([child], parent_pid=42) == 0
        assert not child.terminated

        monkeypatch.setattr(status, "_IS_WINDOWS", False)
        assert status.reap_gateway_children([], parent_pid=42) == 0

    def test_never_raises_when_psutil_explodes(self, monkeypatch):
        monkeypatch.setattr(status, "_IS_WINDOWS", False)
        fake = _fake_psutil(monkeypatch)
        fake.wait_procs.side_effect = RuntimeError("boom")
        child = _FakeChild(108, ppid=1)

        # Must swallow and return best-effort count, not raise.
        assert status.reap_gateway_children([child], parent_pid=42) == 0


class TestSnapshotGatewayChildren:
    def test_snapshot_walks_descendants_recursively(self, monkeypatch):
        monkeypatch.setattr(status, "_IS_WINDOWS", False)
        fake = _fake_psutil(monkeypatch)
        kids = [_FakeChild(201), _FakeChild(202)]
        fake.Process.return_value.children.return_value = kids

        assert status._snapshot_gateway_children(42) == kids
        fake.Process.assert_called_once_with(42)
        fake.Process.return_value.children.assert_called_once_with(recursive=True)

    def test_snapshot_returns_empty_on_windows_or_error(self, monkeypatch):
        monkeypatch.setattr(status, "_IS_WINDOWS", True)
        assert status._snapshot_gateway_children(42) == []

        monkeypatch.setattr(status, "_IS_WINDOWS", False)
        fake = _fake_psutil(monkeypatch)
        fake.Process.side_effect = RuntimeError("gone")
        assert status._snapshot_gateway_children(42) == []


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

    def test_failed_takeover_does_not_reap(self, tmp_path, monkeypatch):
        # Owner never exits; safe_to_force stays False via unknown start time.
        record = self._verified_owner_env(
            tmp_path, monkeypatch, alive_polls=[True] * 50
        )
        monkeypatch.setattr(status, "_snapshot_gateway_children", lambda pid: [])
        starts = iter([123, 123] + [None] * 50)
        monkeypatch.setattr(
            status, "_get_process_start_time", lambda _pid: next(starts)
        )
        reap = MagicMock()
        monkeypatch.setattr(status, "reap_gateway_children", reap)
        monkeypatch.setattr(status, "terminate_pid", lambda pid, *, force=False: None)
        monkeypatch.setattr(status.time, "sleep", lambda _s: None)

        assert (
            status.take_over_scoped_lock_holder(
                record, graceful_attempts=1, force_attempts=1
            )
            is None
        )
        reap.assert_not_called()

    def test_unverified_holder_is_never_snapshotted_or_signalled(
        self, tmp_path, monkeypatch
    ):
        """A non-gateway lock record fails identity validation: no snapshot,
        no terminate, no reap — regardless of --replace intent upstream."""
        record = {"pid": 4242, "kind": "something-else", "start_time": 123}
        snapshot = MagicMock()
        terminate = MagicMock()
        reap = MagicMock()
        monkeypatch.setattr(status, "_snapshot_gateway_children", snapshot)
        monkeypatch.setattr(status, "terminate_pid", terminate)
        monkeypatch.setattr(status, "reap_gateway_children", reap)

        assert status.take_over_scoped_lock_holder(record) is None
        snapshot.assert_not_called()
        terminate.assert_not_called()
        reap.assert_not_called()


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


@pytest.mark.asyncio
async def test_start_gateway_without_replace_never_touches_old_gateway(
    monkeypatch, tmp_path
):
    """Without --replace an existing gateway aborts startup: no takeover
    authority is armed, no snapshot/terminate/reap ever runs."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    snapshot = MagicMock()
    terminate = MagicMock()
    reap = MagicMock()
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 42)
    monkeypatch.setattr("gateway.status._snapshot_gateway_children", snapshot)
    monkeypatch.setattr("gateway.status.terminate_pid", terminate)
    monkeypatch.setattr("gateway.status.reap_gateway_children", reap)
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)

    class _RunnerShouldNotStart:
        def __init__(self, config):
            raise AssertionError("must not start while another gateway runs")

    monkeypatch.setattr("gateway.run.GatewayRunner", _RunnerShouldNotStart)

    from gateway.run import start_gateway

    ok = await start_gateway(config=GatewayConfig(), replace=False, verbosity=None)

    assert ok is False
    snapshot.assert_not_called()
    terminate.assert_not_called()
    reap.assert_not_called()
