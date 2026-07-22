"""Unit tests for Phase 2c host-global claude_cli concurrency semaphore.

Covers:
  * Nth concurrent acquire blocks then falls back (timeout → ClaudeCliConcurrencyError)
  * Slot released on completion
  * Slot released on failure / exception path
  * Stale-slot reaping (dead PID)
  * Cap disabled (max_concurrent=0 / unbounded)
  * Cross-thread last-slot exclusivity

No live ``claude`` / network calls.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agent.transports import claude_cli_concurrency as conc
from agent.transports.claude_cli import ClaudeCliConcurrencyError


@pytest.fixture(autouse=True)
def _slot_dir(tmp_path, monkeypatch):
    slot_dir = tmp_path / "claude_cli_slots"
    monkeypatch.setenv("HERMES_CLAUDE_CLI_SLOT_DIR", str(slot_dir))
    # Avoid config.yaml bleed from the host.
    monkeypatch.delenv("HERMES_CLAUDE_CLI_MAX_CONCURRENT", raising=False)
    monkeypatch.delenv("HERMES_CLAUDE_CLI_ACQUIRE_TIMEOUT", raising=False)
    return slot_dir


def test_resolve_defaults(monkeypatch):
    monkeypatch.setattr(conc, "_load_model_claude_cli_cfg", lambda: {})
    assert conc.resolve_claude_cli_max_concurrent() == conc.DEFAULT_MAX_CONCURRENT
    assert (
        conc.resolve_claude_cli_acquire_timeout()
        == conc.DEFAULT_ACQUIRE_TIMEOUT_SECONDS
    )


def test_resolve_from_config(monkeypatch):
    monkeypatch.setattr(
        conc,
        "_load_model_claude_cli_cfg",
        lambda: {"max_concurrent": 2, "acquire_timeout_seconds": 1.5},
    )
    assert conc.resolve_claude_cli_max_concurrent() == 2
    assert conc.resolve_claude_cli_acquire_timeout() == 1.5


def test_resolve_zero_means_unbounded(monkeypatch):
    monkeypatch.setattr(
        conc, "_load_model_claude_cli_cfg", lambda: {"max_concurrent": 0}
    )
    assert conc.resolve_claude_cli_max_concurrent() is None


def test_env_bridge_overrides_config(monkeypatch):
    monkeypatch.setattr(
        conc, "_load_model_claude_cli_cfg", lambda: {"max_concurrent": 9}
    )
    monkeypatch.setenv("HERMES_CLAUDE_CLI_MAX_CONCURRENT", "1")
    monkeypatch.setenv("HERMES_CLAUDE_CLI_ACQUIRE_TIMEOUT", "0.2")
    assert conc.resolve_claude_cli_max_concurrent() == 1
    assert conc.resolve_claude_cli_acquire_timeout() == 0.2


def test_nth_concurrent_try_acquire_blocks(tmp_path, monkeypatch):
    """With max=1, second try_acquire fails immediately (slot full)."""
    lease1, msg1 = conc.try_acquire_claude_cli_slot(max_concurrent=1)
    assert msg1 is None and lease1 is not None and lease1.enabled

    lease2, msg2 = conc.try_acquire_claude_cli_slot(max_concurrent=1)
    assert lease2 is None
    assert msg2 is not None
    assert "concurrency cap" in msg2

    lease1.release()
    lease3, msg3 = conc.try_acquire_claude_cli_slot(max_concurrent=1)
    assert msg3 is None and lease3 is not None
    lease3.release()
    assert conc.claude_cli_slot_registry_snapshot() == []


def test_acquire_timeout_raises_concurrency_error(monkeypatch):
    lease1, _ = conc.try_acquire_claude_cli_slot(max_concurrent=1)
    assert lease1 is not None
    try:
        with pytest.raises(ClaudeCliConcurrencyError) as ei:
            conc.acquire_claude_cli_slot(
                max_concurrent=1, timeout_seconds=0.25
            )
        err = ei.value
        assert "concurrency" in str(err).lower() or "cap" in str(err).lower()
        assert err.max_concurrent == 1
        assert err.timeout_seconds == 0.25
    finally:
        lease1.release()


def test_slot_released_on_context_manager_success():
    with conc.claude_cli_slot(max_concurrent=1, timeout_seconds=1.0) as lease:
        assert lease.enabled
        assert len(conc.claude_cli_slot_registry_snapshot()) == 1
    assert conc.claude_cli_slot_registry_snapshot() == []


def test_slot_released_on_context_manager_failure():
    with pytest.raises(RuntimeError, match="boom"):
        with conc.claude_cli_slot(max_concurrent=1, timeout_seconds=1.0):
            assert len(conc.claude_cli_slot_registry_snapshot()) == 1
            raise RuntimeError("boom")
    # Slot must not leak after exception.
    assert conc.claude_cli_slot_registry_snapshot() == []


def test_stale_slot_reaped(monkeypatch):
    """Dead-PID slots are pruned so the next acquire can proceed."""
    monkeypatch.setattr(
        "gateway.status._pid_exists",
        lambda pid: int(pid) != 99999999,
    )
    state_path = conc._state_path()
    conc._write_entries(
        state_path,
        [
            {
                "lease_id": "stale",
                "pid": 99999999,
                "started_at": 1.0,
                "updated_at": 1.0,
            }
        ],
    )
    lease, msg = conc.try_acquire_claude_cli_slot(max_concurrent=1)
    assert msg is None and lease is not None
    snap = conc.claude_cli_slot_registry_snapshot()
    assert len(snap) == 1
    assert snap[0]["lease_id"] == lease.lease_id
    lease.release()


def test_hard_exit_child_reclaimed(tmp_path, monkeypatch):
    """A process that os._exit(0) without release is reaped by the next acquire."""
    repo_root = Path(__file__).resolve().parents[2]
    slot_dir = tmp_path / "claude_cli_slots"
    env = os.environ.copy()
    env["HERMES_CLAUDE_CLI_SLOT_DIR"] = str(slot_dir)
    env["PYTHONPATH"] = str(repo_root)
    # Keep pytest seat belt happy in child by having SLOT_DIR set.
    env.pop("PYTEST_CURRENT_TEST", None)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os\n"
                "from agent.transports.claude_cli_concurrency import "
                "try_acquire_claude_cli_slot\n"
                "lease, msg = try_acquire_claude_cli_slot(max_concurrent=1)\n"
                "assert msg is None, msg\n"
                "print(os.getpid(), flush=True)\n"
                "os._exit(0)\n"
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    child_pid = int(child.stdout.strip())
    assert child_pid > 0

    lease, msg = conc.try_acquire_claude_cli_slot(max_concurrent=1)
    assert msg is None and lease is not None
    lease.release()


def test_concurrent_threads_claim_only_cap_slots():
    results = []

    def _claim(_i):
        lease, msg = conc.try_acquire_claude_cli_slot(max_concurrent=2)
        results.append((lease, msg))
        return lease

    with ThreadPoolExecutor(max_workers=8) as pool:
        leases = list(pool.map(_claim, range(8)))

    held = [l for l in leases if l is not None]
    blocked = [msg for lease, msg in results if lease is None and msg]
    try:
        assert len(held) == 2
        assert len(blocked) == 6
    finally:
        for lease in held:
            lease.release()


def test_wait_then_acquire_after_release():
    """Bounded wait succeeds when a holder releases mid-wait."""
    lease1, _ = conc.try_acquire_claude_cli_slot(max_concurrent=1)
    assert lease1 is not None

    ready = threading.Event()
    done = {}

    def _waiter():
        ready.set()
        lease = conc.acquire_claude_cli_slot(
            max_concurrent=1, timeout_seconds=3.0
        )
        done["lease"] = lease

    t = threading.Thread(target=_waiter, daemon=True)
    t.start()
    assert ready.wait(1.0)
    time.sleep(0.3)
    lease1.release()
    t.join(timeout=5.0)
    assert "lease" in done
    done["lease"].release()
    assert conc.claude_cli_slot_registry_snapshot() == []


def test_unbounded_when_cap_none():
    lease, msg = conc.try_acquire_claude_cli_slot(max_concurrent=None)
    # resolve with override None still goes to config — pass 0 via resolve:
    # max_concurrent=None means "look up config". Use 0 via env.
    assert msg is None  # either no-op or real lease
    if lease:
        lease.release()

    # Explicit override: 0 is not accepted by try_acquire's int path the same
    # way — resolve_claude_cli_max_concurrent(0) → None.
    assert conc.resolve_claude_cli_max_concurrent(0) is None
    lease2, msg2 = conc.try_acquire_claude_cli_slot(
        max_concurrent=conc.resolve_claude_cli_max_concurrent(0)
    )
    assert msg2 is None and lease2 is not None and lease2.enabled is False
    lease2.release()


def test_session_holds_slot_for_turn(tmp_path, monkeypatch):
    """ClaudeCliSession.run_turn acquires and releases the host slot."""
    from agent.transports.claude_cli import ClaudeCliSpawnConfig
    from agent.transports.claude_cli_session import ClaudeCliSession

    class _FakeClient:
        def __init__(self, **kw):
            self._closed = False

        def spawn(self, cfg: ClaudeCliSpawnConfig):
            # While the turn is in progress the slot must be held.
            snap = conc.claude_cli_slot_registry_snapshot()
            assert len(snap) == 1, snap
            return None

        def iter_stdout_lines(self, timeout=None):
            sid = "11111111-1111-1111-1111-111111111111"
            yield (
                '{"type":"system","subtype":"init","session_id":"%s"}' % sid
            )
            yield (
                '{"type":"result","subtype":"success","is_error":false,'
                '"result":"ok","session_id":"%s",'
                '"usage":{"input_tokens":1,"output_tokens":1}}' % sid
            )

        def wait(self, timeout=None):
            return 0

        def stderr_tail(self, n=20):
            return []

        def close(self):
            self._closed = True

    monkeypatch.setenv("HERMES_CLAUDE_CLI_MAX_CONCURRENT", "1")
    session = ClaudeCliSession(
        oauth_token="sk-ant-oat01-TEST",
        model="claude-opus-4-8",
        cwd=str(tmp_path),
        client_factory=lambda **kw: _FakeClient(**kw),
    )
    result = session.run_turn("hi")
    assert result.final_text == "ok"
    # Slot released after turn.
    assert conc.claude_cli_slot_registry_snapshot() == []
