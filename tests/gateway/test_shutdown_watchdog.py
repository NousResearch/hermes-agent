"""Shutdown watchdog + loop heartbeat coverage for #66892.

The drain path is asyncio-based; a frozen loop makes every asyncio timeout
structurally unable to fire. These tests pin the out-of-loop backstop
(thread watchdog) and the loop-liveness heartbeat file contract.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

from gateway.shutdown_watchdog import (
    DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S,
    arm_shutdown_watchdog,
    get_shutdown_watchdog_dump_path,
    resolve_shutdown_watchdog_delay,
)

def test_resolve_shutdown_watchdog_delay_adds_grace():
    assert resolve_shutdown_watchdog_delay(180) == 180 + DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S
    assert resolve_shutdown_watchdog_delay(0) == DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S
    assert resolve_shutdown_watchdog_delay("bad") == DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S
    assert resolve_shutdown_watchdog_delay(10, grace_s=5) == 15.0


def test_arm_shutdown_watchdog_fires_with_dump_and_exit(tmp_path):
    done = threading.Event()
    fired = threading.Event()
    dump = tmp_path / "logs" / "watchdog.log"
    snapshot_calls = []
    exit_codes = []

    def snapshot():
        snapshot_calls.append(1)
        return {"active_agents": 1, "draining": True}

    def fake_exit(code):
        exit_codes.append(code)
        fired.set()

    with patch("gateway.shutdown_watchdog.os._exit", side_effect=fake_exit):
        arm_shutdown_watchdog(
            0.15,
            done_event=done,
            snapshot_fn=snapshot,
            dump_path=dump,
            exit_code=9,
        )
        assert fired.wait(timeout=5.0), "watchdog did not fire"

    assert exit_codes == [9]
    assert snapshot_calls == [1]
    assert dump.is_file()
    text = dump.read_text(encoding="utf-8")
    assert "shutdown_watchdog_fired" in text
    assert "faulthandler dump" in text
    assert get_shutdown_watchdog_dump_path(tmp_path).name == "gateway-shutdown-watchdog.log"


def test_arm_shutdown_watchdog_exits_despite_wedged_pid_cleanup(tmp_path, monkeypatch):
    """Hard-exit must survive a hung remove_pid_file (wedged NFS/flock).

    try/except only catches raised exceptions; a blocking unlink/flock never
    returns. Without a join timeout the watchdog thread itself freezes and
    os._exit never runs — the failure mode this module exists to recover from.
    """
    monkeypatch.setattr(
        "gateway.shutdown_watchdog.DEFAULT_WATCHDOG_CLEANUP_TIMEOUT_S",
        0.15,
    )
    done = threading.Event()
    fired = threading.Event()
    blocked = threading.Event()
    exit_codes = []

    def wedged_remove_pid_file():
        blocked.set()
        threading.Event().wait()  # hang until process exit; no sleep timer

    def fake_exit(code):
        exit_codes.append(code)
        fired.set()

    with (
        patch("gateway.shutdown_watchdog.os._exit", side_effect=fake_exit),
        patch("gateway.status.remove_pid_file", side_effect=wedged_remove_pid_file),
        patch("gateway.status.release_gateway_runtime_lock"),
        patch("hermes_logging.drain_log_queue"),
        patch("gateway.lifecycle_ledger.mark_exited"),
    ):
        arm_shutdown_watchdog(
            0.1,
            done_event=done,
            dump_path=tmp_path / "dump.log",
            exit_code=11,
        )
        assert fired.wait(timeout=3.0), "watchdog hung on PID cleanup and never exited"

    assert exit_codes == [11]
    assert blocked.is_set()


def test_arm_shutdown_watchdog_exits_despite_wedged_lifecycle_mark(tmp_path, monkeypatch):
    """Hard-exit must survive a hung lifecycle-sentinel write."""
    monkeypatch.setattr(
        "gateway.shutdown_watchdog.DEFAULT_WATCHDOG_CLEANUP_TIMEOUT_S",
        0.15,
    )
    done = threading.Event()
    fired = threading.Event()
    blocked = threading.Event()
    exit_codes = []

    def wedged_mark_exited(*_args, **_kwargs):
        blocked.set()
        threading.Event().wait()

    def fake_exit(code):
        exit_codes.append(code)
        fired.set()

    with (
        patch("gateway.shutdown_watchdog.os._exit", side_effect=fake_exit),
        patch("gateway.status.remove_pid_file"),
        patch("gateway.status.release_gateway_runtime_lock"),
        patch("hermes_logging.drain_log_queue"),
        patch("gateway.lifecycle_ledger.mark_exited", side_effect=wedged_mark_exited),
    ):
        arm_shutdown_watchdog(
            0.1,
            done_event=done,
            dump_path=tmp_path / "dump.log",
            exit_code=12,
        )
        assert fired.wait(timeout=3.0), (
            "watchdog hung on lifecycle mark_exited and never exited"
        )

    assert exit_codes == [12]
    assert blocked.is_set()


def test_arm_shutdown_watchdog_exits_despite_wedged_dump_io(tmp_path, monkeypatch):
    """Hard-exit must survive hung dump-file I/O before os._exit."""
    monkeypatch.setattr(
        "gateway.shutdown_watchdog.DEFAULT_WATCHDOG_CLEANUP_TIMEOUT_S",
        0.15,
    )
    done = threading.Event()
    fired = threading.Event()
    blocked = threading.Event()
    exit_codes = []
    dump = tmp_path / "dump.log"

    def wedged_open(*_args, **_kwargs):
        blocked.set()
        threading.Event().wait()

    def fake_exit(code):
        exit_codes.append(code)
        fired.set()

    with (
        patch("gateway.shutdown_watchdog.os._exit", side_effect=fake_exit),
        patch("gateway.shutdown_watchdog.open", side_effect=wedged_open),
        patch("gateway.status.remove_pid_file"),
        patch("gateway.status.release_gateway_runtime_lock"),
        patch("hermes_logging.drain_log_queue"),
        patch("gateway.lifecycle_ledger.mark_exited"),
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback"),
    ):
        arm_shutdown_watchdog(
            0.1,
            done_event=done,
            dump_path=dump,
            exit_code=13,
        )
        assert fired.wait(timeout=3.0), "watchdog hung on dump I/O and never exited"

    assert exit_codes == [13]
    assert blocked.is_set()
