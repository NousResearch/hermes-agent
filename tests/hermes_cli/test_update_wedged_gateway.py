"""Tests for the wedged-gateway health probe + bounded escalation (#81642).

A gateway whose event loop is stalled cannot process a graceful shutdown, so
the updater's drain wait used to burn the full 180s budget ("Gateway PID X
still running after 180.0s — restart may fail") and could deadlock `hermes
update`. The fix probes the loop-liveness heartbeat file BEFORE draining and,
only when the loop is provably dead, escalates SIGTERM → SIGKILL bounded to
seconds. A busy-but-alive gateway (fresh heartbeat) must keep the full drain
path — including the in-flight cron drain floor from #86684.
"""

import asyncio
import json
import os
import socket
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_cli.gateway as gateway_cli
from gateway.shutdown_watchdog import (
    get_loop_heartbeat_path,
    get_loop_tick_socket_path,
    loop_heartbeat_forever,
    write_loop_heartbeat,
)


@pytest.fixture()
def tmp_path():
    """Short-path override for this module (macOS AF_UNIX ~104-byte limit).

    The loop-tick witness tests bind real UNIX sockets under HERMES_HOME.
    pytest's default tmp_path nests deep enough on macOS that
    ``state/gateway.loop-tick.<pid>.sock`` exceeds the sockaddr_un limit and
    ``bind()`` raises ``OSError: AF_UNIX path too long``. A mkdtemp directly
    under the system temp root keeps the socket path well under the limit
    on every platform.
    """
    return Path(tempfile.mkdtemp(prefix="hwg-"))


def _write_heartbeat(home, pid, age_s=0.0):
    """Write a heartbeat file for ``pid`` whose mtime is ``age_s`` old."""
    path = get_loop_heartbeat_path(home)
    write_loop_heartbeat(pid=pid, home=home)
    if age_s:
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))
    return path


def _mark_witness_flag(home, armed, age_s=0.0):
    """Set ``loop_tick_socket`` on the heartbeat payload; re-stamp mtime."""
    path = get_loop_heartbeat_path(home)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["loop_tick_socket"] = armed
    path.write_text(json.dumps(payload), encoding="utf-8")
    if age_s:
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))
    return path


def _silent_socket_node(path):
    """Create a socket node at ``path`` that never answers.

    Bind + listen, then close the listener WITHOUT unlinking: the node stays,
    so a probe's connect() gets ECONNREFUSED — a witness that exists but is
    silent, exactly like a dead listener (or a loop that stopped scheduling).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        srv.bind(str(path))
        srv.listen(1)
    finally:
        srv.close()


def _start_freezeable_producer(tmp_path, block_s, errors, write_stall_s=1.5):
    """Run the real heartbeat producer on a loop that can be frozen on demand.

    Returns ``(state, ready)``:

    - ``state["trigger"]`` — freezes the gateway loop synchronously for
      ``block_s`` seconds. While frozen, NO task runs on the loop — not the
      heartbeat writer, not the tick-socket handler — which is exactly the
      "loop not scheduling" condition the probe must detect (sustained).
    - ``state["thread"]`` — the producer thread; join it in ``finally``.
    - ``ready`` — set once the loop is running (socket may still be arming).

    The heartbeat write is patched to stall ``write_stall_s`` after the first
    write, so the file goes stale while the loop keeps dispatching.
    """
    freeze_evt = asyncio.Event()
    state: dict = {"loop": None, "trigger": None, "thread": None}
    ready = threading.Event()

    def stalling_write(**_kwargs):
        if not get_loop_heartbeat_path(tmp_path).exists():
            return write_loop_heartbeat(**_kwargs)
        time.sleep(write_stall_s)
        return write_loop_heartbeat(**_kwargs)

    async def freeze_gate() -> None:
        while True:
            await freeze_evt.wait()
            freeze_evt.clear()
            # Synchronous sleep: freezes the entire loop for block_s.
            time.sleep(block_s)

    async def producer() -> None:
        loop = asyncio.get_running_loop()
        state["loop"] = loop
        state["trigger"] = lambda: loop.call_soon_threadsafe(freeze_evt.set)
        with patch("gateway.shutdown_watchdog.write_loop_heartbeat", stalling_write):
            task = asyncio.create_task(
                loop_heartbeat_forever(interval_s=1.0, home=tmp_path)
            )
            gate = asyncio.create_task(freeze_gate())
            try:
                ready.set()
                while True:
                    await asyncio.sleep(3600)
            finally:
                task.cancel()
                gate.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                try:
                    await gate
                except asyncio.CancelledError:
                    pass

    def run_producer() -> None:
        try:
            asyncio.run(producer())
        except Exception as exc:  # surfaced via errors after join
            errors.append(exc)

    thread = threading.Thread(target=run_producer, daemon=True)
    thread.start()
    state["thread"] = thread
    return state, ready


def _wait_heartbeat_stale(tmp_path, stale_after, timeout_s=5.0):
    """Block until the heartbeat file is older than ``stale_after``."""
    hb_path = get_loop_heartbeat_path(tmp_path)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            age = time.time() - hb_path.stat().st_mtime
        except FileNotFoundError:
            # The socket is armed before the first write lands; the file
            # appears a tick later.
            age = 0.0
        if age > stale_after:
            return
        assert time.monotonic() < deadline, "heartbeat never went stale"
        time.sleep(0.02)


def _launchd_harness(monkeypatch, tmp_path, pid):
    """Patch the launchd_restart path so the REAL probe drives it.

    Returns the ``events`` list: "probe" is recorded by the probe wrapper
    below, "escalate" by ``_escalate_wedged_gateway``, ``("drain", t)`` by
    the graceful drain wait, "kickstart" by the relaunch. The probe itself
    is the real ``probe_gateway_loop_liveness`` (home resolved through the
    patched ``_process_hermes_home``).
    """
    events = []
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 180.0)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: pid)
    monkeypatch.setattr(
        gateway_cli, "_request_gateway_self_restart", lambda pid: False
    )
    real_probe = gateway_cli.probe_gateway_loop_liveness

    def recording_probe(pid, **kw):
        events.append("probe")
        return real_probe(pid, **kw)

    monkeypatch.setattr(gateway_cli, "probe_gateway_loop_liveness", recording_probe)
    monkeypatch.setattr(
        gateway_cli,
        "_escalate_wedged_gateway",
        lambda pid, **kw: events.append("escalate") or True,
    )
    monkeypatch.setattr(
        gateway_cli,
        "terminate_pid",
        lambda pid, force=False: events.append(("kill" if force else "term", pid)),
    )
    monkeypatch.setattr(
        gateway_cli,
        "_wait_for_gateway_exit",
        lambda timeout, force_after=None: events.append(("drain", timeout)) or True,
    )
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda *a, **k: events.append("kickstart")
        or __import__("types").SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(gateway_cli, "_clear_launchd_unsupported_marker", lambda: None)
    monkeypatch.setattr(
        "gateway.shutdown_watchdog._process_hermes_home", lambda: tmp_path
    )
    return events


class TestProbeGatewayLoopLiveness:
    def test_fresh_heartbeat_is_alive(self, tmp_path):
        """A gateway that refreshed its heartbeat recently is busy, not wedged."""
        _write_heartbeat(tmp_path, pid=4242, age_s=5.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(4242, home=tmp_path)
            == gateway_cli.GATEWAY_LOOP_ALIVE
        )

    def test_stale_heartbeat_is_wedged(self, tmp_path):
        """A heartbeat several missed beats old proves the loop is dead."""
        _write_heartbeat(tmp_path, pid=4242, age_s=600.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(4242, home=tmp_path)
            == gateway_cli.GATEWAY_LOOP_WEDGED
        )

    def test_heartbeat_just_inside_budget_is_alive(self, tmp_path):
        """Boundary: age below the stale budget must NOT classify as wedged."""
        _write_heartbeat(tmp_path, pid=4242, age_s=60.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(
                4242, stale_after=90.0, home=tmp_path
            )
            == gateway_cli.GATEWAY_LOOP_ALIVE
        )

    def test_missing_heartbeat_is_unknown(self, tmp_path):
        """No heartbeat file (older gateway, fresh start) is not evidence."""
        assert (
            gateway_cli.probe_gateway_loop_liveness(4242, home=tmp_path)
            == gateway_cli.GATEWAY_LOOP_UNKNOWN
        )

    def test_pid_mismatch_is_unknown_even_when_stale(self, tmp_path):
        """A stale file from a PREVIOUS process must not condemn the new PID."""
        _write_heartbeat(tmp_path, pid=1111, age_s=600.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(4242, home=tmp_path)
            == gateway_cli.GATEWAY_LOOP_UNKNOWN
        )

    def test_corrupt_heartbeat_is_unknown(self, tmp_path):
        path = get_loop_heartbeat_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        stamp = time.time() - 600.0
        os.utime(path, (stamp, stamp))
        assert (
            gateway_cli.probe_gateway_loop_liveness(4242, home=tmp_path)
            == gateway_cli.GATEWAY_LOOP_UNKNOWN
        )

    def test_nonpositive_pid_is_unknown(self, tmp_path):
        _write_heartbeat(tmp_path, pid=4242, age_s=600.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(0, home=tmp_path)
            == gateway_cli.GATEWAY_LOOP_UNKNOWN
        )

    def test_invalid_stale_after_falls_back_to_default(self, tmp_path):
        _write_heartbeat(tmp_path, pid=4242, age_s=600.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(
                4242, stale_after="bogus", home=tmp_path
            )
            == gateway_cli.GATEWAY_LOOP_WEDGED
        )

    def test_probe_never_raises_on_unreadable_path(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.shutdown_watchdog.get_loop_heartbeat_path",
            lambda home=None: (_ for _ in ()).throw(OSError("boom")),
        )
        assert (
            gateway_cli.probe_gateway_loop_liveness(4242)
            == gateway_cli.GATEWAY_LOOP_UNKNOWN
        )


class TestEscalateWedgedGateway:
    def test_sigterm_grace_suffices_without_sigkill(self, monkeypatch):
        """If SIGTERM lands (signal-handler thread alive), no SIGKILL is sent."""
        signals = []
        monkeypatch.setattr(
            gateway_cli,
            "terminate_pid",
            lambda pid, force=False: signals.append(("kill" if force else "term", pid)),
        )
        monkeypatch.setattr(
            gateway_cli, "_wait_for_pid_exit", lambda pid, timeout: True
        )

        assert gateway_cli._escalate_wedged_gateway(4242) is True
        assert signals == [("term", 4242)]

    def test_escalates_to_sigkill_when_sigterm_ignored(self, monkeypatch):
        signals = []
        waits = []

        def fake_wait(pid, timeout):
            waits.append(timeout)
            # First wait (SIGTERM grace) times out; second (post-SIGKILL) succeeds.
            return len(waits) > 1

        monkeypatch.setattr(
            gateway_cli,
            "terminate_pid",
            lambda pid, force=False: signals.append(("kill" if force else "term", pid)),
        )
        monkeypatch.setattr(gateway_cli, "_wait_for_pid_exit", fake_wait)

        assert gateway_cli._escalate_wedged_gateway(4242) is True
        assert signals == [("term", 4242), ("kill", 4242)]

    def test_total_wait_budget_is_bounded_well_under_drain(self, monkeypatch):
        """Worst case must be seconds, never the 180s drain budget."""
        waits = []
        monkeypatch.setattr(gateway_cli, "terminate_pid", lambda pid, force=False: None)
        monkeypatch.setattr(
            gateway_cli,
            "_wait_for_pid_exit",
            lambda pid, timeout: waits.append(timeout) or False,
        )

        assert gateway_cli._escalate_wedged_gateway(4242) is False
        assert sum(waits) < 30.0

    def test_process_already_gone_is_success(self, monkeypatch):
        def raise_gone(pid, force=False):
            raise ProcessLookupError

        monkeypatch.setattr(gateway_cli, "terminate_pid", raise_gone)
        monkeypatch.setattr(
            gateway_cli, "_wait_for_pid_exit", lambda pid, timeout: True
        )

        assert gateway_cli._escalate_wedged_gateway(4242) is True

    def test_sigkill_permission_error_does_not_raise(self, monkeypatch):
        calls = []

        def term(pid, force=False):
            calls.append(force)
            if force:
                raise PermissionError

        monkeypatch.setattr(gateway_cli, "terminate_pid", term)
        monkeypatch.setattr(
            gateway_cli, "_wait_for_pid_exit", lambda pid, timeout: False
        )

        assert gateway_cli._escalate_wedged_gateway(4242) is False
        assert calls == [False, True]


class TestLaunchdRestartWedgedIntegration:
    """launchd_restart must skip the 180s drain only for a wedged loop."""

    def _setup(self, monkeypatch, liveness):
        events = []
        monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
        monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
        monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 180.0)
        # Wait budget covers after-turn deferral + drain + headroom (#77184).
        monkeypatch.setattr(gateway_cli, "_get_restart_exit_wait_budget", lambda: 195.0)
        monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: 4242)
        monkeypatch.setattr(
            gateway_cli, "_request_gateway_self_restart", lambda pid: False
        )
        monkeypatch.setattr(
            gateway_cli,
            "probe_gateway_loop_liveness",
            lambda pid, **kw: events.append("probe") or liveness,
        )
        monkeypatch.setattr(
            gateway_cli,
            "_escalate_wedged_gateway",
            lambda pid, **kw: events.append("escalate") or True,
        )
        monkeypatch.setattr(
            gateway_cli,
            "terminate_pid",
            lambda pid, force=False: events.append("sigterm"),
        )
        # Never let a real SIGUSR1 escape to PID 4242 during tests.
        monkeypatch.setattr(
            gateway_cli,
            "_graceful_restart_via_sigusr1",
            lambda pid, timeout: events.append(("drain", pid, timeout)) or True,
        )
        # KeepAlive revival observed instantly — avoids the real 15s poll
        # (mocked subprocess.run returns empty stdout, so the PID probe
        # would otherwise burn the full observation timeout in time.sleep).
        monkeypatch.setattr(
            gateway_cli,
            "_wait_for_launchd_service_pid",
            lambda label, old_pid, timeout=10.0, *, domain: events.append("observe")
            or True,
        )
        monkeypatch.setattr(
            gateway_cli.subprocess,
            "run",
            lambda *a, **k: events.append("kickstart")
            or __import__("types").SimpleNamespace(returncode=0, stdout="", stderr=""),
        )
        monkeypatch.setattr(
            gateway_cli, "_clear_launchd_unsupported_marker", lambda: None
        )
        return events

    def test_wedged_gateway_skips_drain_and_escalates(self, monkeypatch):
        events = self._setup(monkeypatch, gateway_cli.GATEWAY_LOOP_WEDGED)
        gateway_cli.launchd_restart()
        assert "escalate" in events
        # The 180s drain wait must never run for a wedged loop.
        assert not any(isinstance(e, tuple) and e[0] == "drain" for e in events)

    def test_busy_gateway_keeps_full_drain_budget(self, monkeypatch):
        """A busy-but-alive gateway (fresh heartbeat) must NOT be escalated —
        that would bypass the in-flight cron drain floor (#86684)."""
        events = self._setup(monkeypatch, gateway_cli.GATEWAY_LOOP_ALIVE)
        gateway_cli.launchd_restart()
        assert "escalate" not in events
        assert ("drain", 4242, 195.0) in events

    def test_unknown_liveness_keeps_full_drain_budget(self, monkeypatch):
        """Ambiguity (no heartbeat) must never trigger escalation."""
        events = self._setup(monkeypatch, gateway_cli.GATEWAY_LOOP_UNKNOWN)
        gateway_cli.launchd_restart()
        assert "escalate" not in events
        assert ("drain", 4242, 195.0) in events


class TestLoopTickWitness:
    """Two-witness liveness (#90502 review).

    The heartbeat write moved off-loop, so a stale file no longer proves a
    wedged loop and a fresh file no longer proves an alive one. The loop
    answers a UNIX socket instead; the probe only escalates when BOTH
    witnesses agree the loop stopped scheduling.
    """

    def test_stalled_heartbeat_write_never_escalates_a_running_loop(
        self, tmp_path, monkeypatch
    ):
        """Producer + consumer composition.

        While the heartbeat write is stalled longer than the stale budget, a
        loop that demonstrably keeps dispatching must probe ALIVE — and a
        restart path fed by the real probe must take the graceful drain,
        never the bounded escalation. This is the exact false-positive the
        review called out: the measured fsync stall (112.6s max) exceeds the
        90s destructive-classifier threshold.
        """
        pid = os.getpid()
        block_s = 1.5
        stale_after = 1.0

        # First write lands immediately; every later write stalls like an
        # fsync on the incident filesystem, so the file ages past the budget
        # while the loop keeps running.
        def stalling_write(**_kwargs):
            if not get_loop_heartbeat_path(tmp_path).exists():
                return write_loop_heartbeat(**_kwargs)
            time.sleep(block_s)
            return write_loop_heartbeat(**_kwargs)

        errors = []

        async def producer() -> None:
            with patch(
                "gateway.shutdown_watchdog.write_loop_heartbeat", stalling_write
            ):
                task = asyncio.create_task(
                    loop_heartbeat_forever(interval_s=1.0, home=tmp_path)
                )
                try:
                    # One interval (1.0s) elapses, the second write starts and
                    # stalls; 1.35s in the file is stale but the loop ticks.
                    await asyncio.sleep(1.35)
                finally:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        def run_producer() -> None:
            try:
                asyncio.run(producer())
            except Exception as exc:  # surfaced via errors after join
                errors.append(exc)

        thread = threading.Thread(target=run_producer, daemon=True)
        thread.start()
        try:
            sock_path = get_loop_tick_socket_path(tmp_path, pid)
            deadline = time.monotonic() + 5.0
            while not sock_path.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert sock_path.exists(), "producer never armed the tick socket"

            hb_path = get_loop_heartbeat_path(tmp_path)
            deadline = time.monotonic() + 5.0
            while True:
                try:
                    age = time.time() - hb_path.stat().st_mtime
                except FileNotFoundError:
                    # The socket is armed before the first write lands; the
                    # file appears a tick later.
                    age = 0.0
                if age > stale_after:
                    break
                assert time.monotonic() < deadline, "heartbeat never went stale"
                time.sleep(0.02)

            # The file is stale but the loop answers: ALIVE, not WEDGED.
            assert (
                gateway_cli.probe_gateway_loop_liveness(
                    pid, home=tmp_path, stale_after=stale_after, tick_timeout=0.25
                )
                == gateway_cli.GATEWAY_LOOP_ALIVE
            )

            # The restart path fed by the REAL probe must drain, not escalate.
            events = []
            monkeypatch.setattr(
                gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway"
            )
            monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
            monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 180.0)
            monkeypatch.setattr(
                "gateway.status.get_running_pid", lambda *a, **k: pid
            )
            monkeypatch.setattr(
                gateway_cli, "_request_gateway_self_restart", lambda pid: False
            )
            monkeypatch.setattr(
                gateway_cli,
                "_escalate_wedged_gateway",
                lambda pid, **kw: events.append("escalate") or True,
            )
            monkeypatch.setattr(
                gateway_cli,
                "terminate_pid",
                lambda pid, force=False: events.append(("kill" if force else "term", pid)),
            )
            monkeypatch.setattr(
                gateway_cli,
                "_wait_for_gateway_exit",
                lambda timeout, force_after=None: events.append(("drain", timeout))
                or True,
            )
            monkeypatch.setattr(
                gateway_cli.subprocess,
                "run",
                lambda *a, **k: events.append("kickstart")
                or __import__("types").SimpleNamespace(
                    returncode=0, stdout="", stderr=""
                ),
            )
            monkeypatch.setattr(
                gateway_cli, "_clear_launchd_unsupported_marker", lambda: None
            )
            # The real probe resolves the state dir through this hook.
            monkeypatch.setattr(
                "gateway.shutdown_watchdog._process_hermes_home", lambda: tmp_path
            )
            gateway_cli.launchd_restart()
            assert "escalate" not in events
            assert ("drain", 180.0) in events
        finally:
            thread.join(timeout=5.0)
            assert not errors, errors

    def test_off_loop_completion_cannot_manufacture_fresh_liveness(self, tmp_path):
        """A write landing after the loop froze must not look alive.

        File fresh (a late off-loop write completed) + loop silent: the probe
        must NOT return ALIVE — the loop itself never answered, so freshness
        is not a liveness proof. UNKNOWN keeps the safe drain path while
        denying the false-fresh window the review described.
        """
        pid = 4242
        _silent_socket_node(get_loop_tick_socket_path(tmp_path, pid))
        _write_heartbeat(tmp_path, pid, age_s=5.0)
        _mark_witness_flag(tmp_path, armed=True)
        assert (
            gateway_cli.probe_gateway_loop_liveness(
                pid, home=tmp_path, tick_timeout=0.2
            )
            == gateway_cli.GATEWAY_LOOP_UNKNOWN
        )

    def test_true_wedge_requires_sustained_witness_silence(self, tmp_path):
        """Stale file + armed socket silent across the whole window: WEDGED.

        The destructive path must still exist for genuinely dead loops — a
        frozen loop stops answering the socket for every attempt in the
        bounded window, and the file goes stale. The reviewer's sustained-
        proof contract (one sample is never destructive authority) is
        satisfied here: all ``tick_strikes`` consecutive probes miss.
        """
        pid = 4242
        _silent_socket_node(get_loop_tick_socket_path(tmp_path, pid))
        _write_heartbeat(tmp_path, pid, age_s=600.0)
        _mark_witness_flag(tmp_path, armed=True, age_s=600.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(
                pid,
                home=tmp_path,
                tick_timeout=0.2,
                tick_strikes=3,
                tick_gap_s=0.0,
            )
            == gateway_cli.GATEWAY_LOOP_WEDGED
        )

    def test_single_silent_probe_is_not_destructive(self, tmp_path, monkeypatch):
        """One miss is a transient stall, never a wedge (#90502 review).

        Stale file + armed socket + a loop that misses the FIRST probe but
        answers the second: the probe must recover to ALIVE — the loop is
        demonstrably dispatching, and the stale file is just a stalled
        write. A single silent sample must never grant the bounded kill
        authority.
        """
        pid = 4242
        calls = []

        def flaky_probe(_pid, _home, timeout=1.0):
            calls.append(timeout)
            # First sample misses (transient synchronous stall), the loop
            # answers on the next attempt.
            return len(calls) > 1

        monkeypatch.setattr(gateway_cli, "_probe_loop_tick_socket", flaky_probe)
        _write_heartbeat(tmp_path, pid, age_s=600.0)
        _mark_witness_flag(tmp_path, armed=True, age_s=600.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(
                pid,
                home=tmp_path,
                tick_timeout=0.2,
                tick_strikes=2,
                tick_gap_s=0.0,
            )
            == gateway_cli.GATEWAY_LOOP_ALIVE
        )
        # First probe (miss) + exactly one sustained-window follow-up.
        assert len(calls) == 2

    def test_sustained_silence_is_required_for_wedge(self, tmp_path, monkeypatch):
        """WEDGED only after the full bounded window of consecutive misses.

        With ``tick_strikes=2`` the probe must observe BOTH samples silent
        before granting the wedge verdict; a single silent sample alone
        would previously have escalated.
        """
        pid = 4242
        calls = []
        monkeypatch.setattr(
            gateway_cli,
            "_probe_loop_tick_socket",
            lambda _pid, _home, timeout=1.0: (calls.append(timeout), False)[1],
        )
        _write_heartbeat(tmp_path, pid, age_s=600.0)
        _mark_witness_flag(tmp_path, armed=True, age_s=600.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(
                pid,
                home=tmp_path,
                tick_timeout=0.2,
                tick_strikes=2,
                tick_gap_s=0.0,
            )
            == gateway_cli.GATEWAY_LOOP_WEDGED
        )
        assert len(calls) == 2  # miss #1 (initial probe) + miss #2 (window)

    def test_witness_vanishing_mid_window_is_unknown(self, tmp_path, monkeypatch):
        """A witness that disappears mid-window is ambiguity, not a wedge.

        First sample silent (node existed), second sample finds no node:
        the witness is gone and cannot condemn the loop. UNKNOWN keeps the
        graceful drain path.
        """
        pid = 4242
        calls = []
        monkeypatch.setattr(
            gateway_cli,
            "_probe_loop_tick_socket",
            lambda _pid, _home, timeout=1.0: (calls.append(timeout), False, None)[
                len(calls)
            ],
        )
        _write_heartbeat(tmp_path, pid, age_s=600.0)
        _mark_witness_flag(tmp_path, armed=True, age_s=600.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(
                pid,
                home=tmp_path,
                tick_timeout=0.2,
                tick_strikes=2,
                tick_gap_s=0.0,
            )
            == gateway_cli.GATEWAY_LOOP_UNKNOWN
        )

    def test_armed_witness_unreachable_is_unknown(self, tmp_path):
        """Producer claims the witness is armed but no node exists: ambiguity.

        Never kill on it — the graceful drain remains the backstop.
        """
        pid = 4242
        _write_heartbeat(tmp_path, pid, age_s=600.0)
        _mark_witness_flag(tmp_path, armed=True, age_s=600.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(
                pid, home=tmp_path, tick_timeout=0.2
            )
            == gateway_cli.GATEWAY_LOOP_UNKNOWN
        )

    def test_unarmed_witness_disables_stale_escalation(self, tmp_path):
        """New producer whose bind failed: staleness is NOT proof.

        The write is off-loop, so the file can age while the loop runs; with
        no witness, a stale file must never escalate.
        """
        pid = 4242
        _write_heartbeat(tmp_path, pid, age_s=600.0)
        _mark_witness_flag(tmp_path, armed=False, age_s=600.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(
                pid, home=tmp_path, tick_timeout=0.2
            )
            == gateway_cli.GATEWAY_LOOP_UNKNOWN
        )

    def test_legacy_payload_keeps_single_witness_contract(self, tmp_path):
        """No witness flag = on-loop writer: staleness stays proof.

        Old gateways never moved the write off-loop, so their stale file
        still means a dead loop — the legacy WEDGED verdict is unchanged.
        """
        pid = 4242
        _write_heartbeat(tmp_path, pid, age_s=600.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(pid, home=tmp_path)
            == gateway_cli.GATEWAY_LOOP_WEDGED
        )
        # And a fresh legacy file stays safe even if a dead-listener node
        # exists for the PID (leftover from a newer process): the silent
        # socket denies ALIVE, and UNKNOWN never escalates — the drain path
        # keeps the full budget either way.
        _write_heartbeat(tmp_path, pid, age_s=5.0)
        _silent_socket_node(get_loop_tick_socket_path(tmp_path, pid))
        assert (
            gateway_cli.probe_gateway_loop_liveness(
                pid, home=tmp_path, tick_timeout=0.2
            )
            == gateway_cli.GATEWAY_LOOP_UNKNOWN
        )

    def test_transient_stall_below_wedge_budget_never_escalates(
        self, tmp_path, monkeypatch
    ):
        """Composed producer+consumer: transient loop stall below the budget.

        Heartbeat write stalled (file stale) + loop frozen for longer than
        one recv timeout but shorter than the wedge window: the probe must
        NOT return WEDGED — the loop answers once it unfreezes, inside the
        bounded window — and the restart path fed by the real probe must
        take the graceful drain, never the bounded escalation. This is the
        reviewer's required composed regression (#90502 review).
        """
        pid = os.getpid()
        block_s = 0.5
        stale_after = 1.0
        tick_timeout = 0.2
        tick_strikes = 3
        tick_gap_s = 0.05
        wedge_window = tick_strikes * tick_timeout + (tick_strikes - 1) * tick_gap_s
        assert tick_timeout < block_s < wedge_window, (tick_timeout, block_s, wedge_window)

        errors = []
        state, ready = _start_freezeable_producer(tmp_path, block_s, errors)
        try:
            assert ready.wait(timeout=5.0)
            sock_path = get_loop_tick_socket_path(tmp_path, pid)
            deadline = time.monotonic() + 5.0
            while not sock_path.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert sock_path.exists(), "producer never armed the tick socket"
            _wait_heartbeat_stale(tmp_path, stale_after)

            state["trigger"]()  # freeze the loop for block_s
            verdict = gateway_cli.probe_gateway_loop_liveness(
                pid,
                home=tmp_path,
                stale_after=stale_after,
                tick_timeout=tick_timeout,
                tick_strikes=tick_strikes,
                tick_gap_s=tick_gap_s,
            )
            assert verdict == gateway_cli.GATEWAY_LOOP_ALIVE, verdict

            events = _launchd_harness(monkeypatch, tmp_path, pid)
            gateway_cli.launchd_restart()
            assert "escalate" not in events
            assert ("drain", 180.0) in events
        finally:
            state["thread"].join(timeout=5.0)
            assert not errors, errors

    def test_sustained_stop_above_wedge_budget_still_escalates(
        self, tmp_path
    ):
        """Composed producer+consumer: sustained stop above the budget.

        The bounded-proof contract must not neuter the destructive path: a
        loop frozen for longer than the wedge window stays silent for every
        probe attempt, so the probe still returns WEDGED. (The WEDGED →
        bounded-escalation wiring on the restart paths is covered by
        ``TestLaunchdRestartWedgedIntegration``.)
        """
        pid = os.getpid()
        block_s = 1.2
        stale_after = 1.0
        tick_timeout = 0.2
        tick_strikes = 3
        tick_gap_s = 0.05
        wedge_window = tick_strikes * tick_timeout + (tick_strikes - 1) * tick_gap_s
        assert block_s > wedge_window, (block_s, wedge_window)

        errors = []
        state, ready = _start_freezeable_producer(tmp_path, block_s, errors)
        try:
            assert ready.wait(timeout=5.0)
            sock_path = get_loop_tick_socket_path(tmp_path, pid)
            deadline = time.monotonic() + 5.0
            while not sock_path.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert sock_path.exists(), "producer never armed the tick socket"
            _wait_heartbeat_stale(tmp_path, stale_after)

            state["trigger"]()  # freeze the loop for block_s
            verdict = gateway_cli.probe_gateway_loop_liveness(
                pid,
                home=tmp_path,
                stale_after=stale_after,
                tick_timeout=tick_timeout,
                tick_strikes=tick_strikes,
                tick_gap_s=tick_gap_s,
            )
            assert verdict == gateway_cli.GATEWAY_LOOP_WEDGED, verdict
        finally:
            state["thread"].join(timeout=5.0)
            assert not errors, errors
