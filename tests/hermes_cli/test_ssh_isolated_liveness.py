"""Child-side liveness for SSH-isolated ``hermes serve`` (#101626).

The remote backend is ``setsid``/``nohup`` detached on purpose (#91668), so
``PPID=1`` is not an orphan signal. These tests pin the actual contract:
idle grace after a missing client, exclusive per-home writer lock, and
leaving plain loopback Desktop pings disabled.
"""

from __future__ import annotations

import os

import pytest

from hermes_cli.ssh_isolated_liveness import (
    SshIsolatedIdleTracker,
    acquire_ssh_isolated_home_lock,
    ssh_isolated_idle_step,
    ssh_isolated_should_exit,
    ssh_isolated_ws_ping_window,
    track_ssh_isolated_ws,
)


def test_plain_loopback_keeps_protocol_ping_disabled():
    assert ssh_isolated_ws_ping_window(
        is_loopback=True,
        ssh_session_token="",
        default_interval=20.0,
        default_timeout=20.0,
    ) == (None, None)


def test_ssh_isolated_loopback_enables_half_open_ping():
    interval, timeout = ssh_isolated_ws_ping_window(
        is_loopback=True,
        ssh_session_token="a" * 64,
        default_interval=20.0,
        default_timeout=20.0,
    )
    assert interval and interval >= 60.0
    assert timeout and timeout >= 300.0
    assert timeout >= interval


def test_non_loopback_ping_unchanged_without_ssh_token():
    interval, timeout = ssh_isolated_ws_ping_window(
        is_loopback=False,
        ssh_session_token="",
        default_interval=20.0,
        default_timeout=25.0,
    )
    assert interval == 20.0
    assert timeout == 25.0


def test_idle_grace_does_not_exit_without_ssh_token():
    assert (
        ssh_isolated_should_exit(
            has_ssh_token=False,
            now=1_000.0,
            last_client_at=0.0,
            grace_s=10.0,
            ppid=1,
        )
        is False
    )


def test_ppid_one_is_not_an_exit_signal_while_client_is_recent():
    assert (
        ssh_isolated_should_exit(
            has_ssh_token=True,
            now=1_000.0,
            last_client_at=999.0,
            grace_s=30.0,
            ppid=1,
        )
        is False
    )


def test_ssh_isolated_exits_after_idle_grace():
    assert (
        ssh_isolated_should_exit(
            has_ssh_token=True,
            now=1_000.0,
            last_client_at=900.0,
            grace_s=30.0,
            ppid=1,
        )
        is True
    )


def test_ssh_isolated_stays_up_inside_grace_window():
    assert (
        ssh_isolated_should_exit(
            has_ssh_token=True,
            now=1_000.0,
            last_client_at=980.0,
            grace_s=30.0,
            ppid=42,
        )
        is False
    )


def test_idle_grace_does_not_exit_while_agent_turn_is_in_flight():
    assert (
        ssh_isolated_should_exit(
            has_ssh_token=True,
            now=1_000.0,
            last_client_at=0.0,
            grace_s=10.0,
            turn_in_flight=True,
        )
        is False
    )


def test_idle_step_refreshes_grace_while_turn_runs_then_exits_after():
    class Clock:
        def __init__(self):
            self.now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    tracker = SshIsolatedIdleTracker(clock=clock)
    clock.now = 50.0
    assert (
        ssh_isolated_idle_step(
            has_ssh_token=True,
            tracker=tracker,
            grace_s=10.0,
            turn_in_flight=True,
        )
        is False
    )
    clock.now = 55.0
    assert (
        ssh_isolated_idle_step(
            has_ssh_token=True,
            tracker=tracker,
            grace_s=10.0,
            turn_in_flight=False,
        )
        is False
    )
    clock.now = 70.0
    assert (
        ssh_isolated_idle_step(
            has_ssh_token=True,
            tracker=tracker,
            grace_s=10.0,
            turn_in_flight=False,
        )
        is True
    )


def test_turn_probe_exception_fails_closed_and_refreshes_grace():
    """An exception from turn_probe must fail closed (treat as turn active),
    refreshing grace and deferring shutdown until a subsequent successful
    observation proves no turn is running (#101678 / review 5096241308).
    """
    class Clock:
        def __init__(self):
            self.now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    tracker = SshIsolatedIdleTracker(clock=clock)
    grace = 10.0

    # 1. Advance clock past the idle grace window (50s > 10s grace)
    clock.now = 50.0

    # 2. turn_probe raises an exception (e.g. database error, transient failure)
    def _raising_probe():
        raise RuntimeError("database transiently locked")

    # The step MUST fail closed: shutdown is NOT requested (returns False)
    # and the tracker clock MUST be refreshed
    assert (
        ssh_isolated_idle_step(
            has_ssh_token=True,
            tracker=tracker,
            grace_s=grace,
            turn_probe=_raising_probe,
        )
        is False
    )

    # Confirm the tracker was touched/refreshed at clock.now = 50.0
    assert tracker.last_client_at() == 50.0

    # 3. 5 seconds later (clock.now = 55.0), turn probe succeeds and returns False.
    # Even though probe returns False, only 5s have elapsed since the refreshed
    # observation, which is < 10.0s grace window. Shutdown must STILL be False!
    clock.now = 55.0
    assert (
        ssh_isolated_idle_step(
            has_ssh_token=True,
            tracker=tracker,
            grace_s=grace,
            turn_probe=lambda: False,
        )
        is False
    )

    # 4. Now advance past the refreshed grace window (clock.now = 65.0 > 50.0 + 10.0)
    # with successful no-turn observation. Now shutdown MUST be requested (returns True).
    clock.now = 65.0
    assert (
        ssh_isolated_idle_step(
            has_ssh_token=True,
            tracker=tracker,
            grace_s=grace,
            turn_probe=lambda: False,
        )
        is True
    )


def test_track_ws_context_does_not_leak_on_error():
    tracker = SshIsolatedIdleTracker()
    import hermes_cli.ssh_isolated_liveness as mod

    previous = mod._idle_tracker
    mod._idle_tracker = tracker
    try:
        assert tracker.live_count() == 0
        with track_ssh_isolated_ws():
            assert tracker.live_count() == 1
            raise RuntimeError("disconnect")
    except RuntimeError:
        pass
    finally:
        mod._idle_tracker = previous
    assert tracker.live_count() == 0


@pytest.mark.linux_only
def test_second_ssh_isolated_serve_cannot_take_the_home_lock(tmp_path):
    first = acquire_ssh_isolated_home_lock(tmp_path)
    assert first is not None
    try:
        second = acquire_ssh_isolated_home_lock(tmp_path)
        assert second is None
    finally:
        os.close(first)


def test_ensure_lock_cooperative_handover_retires_idle_orphan(tmp_path, monkeypatch):
    from hermes_cli.ssh_isolated_liveness import (
        ensure_ssh_isolated_home_lock,
        read_ssh_isolated_state,
        write_ssh_isolated_state,
    )

    incumbent_fd = acquire_ssh_isolated_home_lock(tmp_path)
    assert incumbent_fd is not None

    try:
        write_ssh_isolated_state(
            tmp_path,
            pid=99999,
            state="idle",
            active_clients=0,
            turn_active=False,
        )

        signaled = []

        def fake_signal(pid):
            signaled.append(pid)
            os.close(incumbent_fd)

        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness._is_positive_flock_holder",
            lambda pid, home, state_info: True,
        )
        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness._signal_retirement", fake_signal
        )

        new_fd = ensure_ssh_isolated_home_lock(
            ssh_session_token="token123",
            hermes_home=tmp_path,
        )
        assert new_fd is not None
        assert signaled == [99999]
        os.close(new_fd)
    finally:
        try:
            os.close(incumbent_fd)
        except OSError:
            pass


def test_ensure_lock_cooperative_handover_refuses_active_incumbent(
    tmp_path, monkeypatch
):
    from hermes_cli.ssh_isolated_liveness import (
        ensure_ssh_isolated_home_lock,
        write_ssh_isolated_state,
    )

    incumbent_fd = acquire_ssh_isolated_home_lock(tmp_path)
    assert incumbent_fd is not None

    try:
        write_ssh_isolated_state(
            tmp_path,
            pid=99999,
            state="active",
            active_clients=1,
            turn_active=False,
        )

        signaled = []
        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness._signal_retirement",
            lambda pid: signaled.append(pid),
        )

        with pytest.raises(SystemExit) as exc:
            ensure_ssh_isolated_home_lock(
                ssh_session_token="token123",
                hermes_home=tmp_path,
            )
        assert "refusing a second writer" in str(exc.value)
        assert signaled == []
    finally:
        os.close(incumbent_fd)


def test_ensure_lock_cooperative_handover_refuses_when_turn_in_flight(
    tmp_path, monkeypatch
):
    from hermes_cli.ssh_isolated_liveness import (
        ensure_ssh_isolated_home_lock,
        write_ssh_isolated_state,
    )

    incumbent_fd = acquire_ssh_isolated_home_lock(tmp_path)
    assert incumbent_fd is not None

    try:
        write_ssh_isolated_state(
            tmp_path,
            pid=99999,
            state="idle",
            active_clients=0,
            turn_active=True,
        )

        signaled = []
        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness._signal_retirement",
            lambda pid: signaled.append(pid),
        )

        with pytest.raises(SystemExit) as exc:
            ensure_ssh_isolated_home_lock(
                ssh_session_token="token123",
                hermes_home=tmp_path,
            )
        assert "refusing a second writer" in str(exc.value)
        assert signaled == []
    finally:
        os.close(incumbent_fd)


def test_e2e_cooperative_handover_with_real_watchdog_state_transitions(
    tmp_path, monkeypatch
):
    """End-to-end test proving that:
    1. start_ssh_isolated_idle_watchdog plumbs hermes_home
    2. track_ssh_isolated_ws transitions disk state from active to idle
    3. ensure_ssh_isolated_home_lock successfully performs cooperative handover
    """
    from hermes_cli.ssh_isolated_liveness import (
        acquire_ssh_isolated_home_lock,
        ensure_ssh_isolated_home_lock,
        read_ssh_isolated_state,
        start_ssh_isolated_idle_watchdog,
        track_ssh_isolated_ws,
    )

    incumbent_fd = acquire_ssh_isolated_home_lock(tmp_path)
    assert incumbent_fd is not None

    try:
        # Start watchdog with hermes_home
        start_ssh_isolated_idle_watchdog(
            has_ssh_token=True,
            hermes_home=tmp_path,
            turn_probe=lambda: False,
        )

        # Client connects and disconnects
        with track_ssh_isolated_ws():
            mid_state = read_ssh_isolated_state(tmp_path)
            assert mid_state is not None
            assert mid_state["state"] == "active"
            assert mid_state["active_clients"] == 1

        # After client disconnect with no turn running, disk state MUST transition to idle
        post_state = read_ssh_isolated_state(tmp_path)
        assert post_state is not None
        assert post_state["state"] == "idle"
        assert post_state["active_clients"] == 0
        assert post_state["turn_active"] is False

        # Simulate cooperative handover: newcomer signals incumbent, incumbent yields
        signaled = []

        def fake_signal(pid):
            signaled.append(pid)
            os.close(incumbent_fd)

        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness._pid_is_alive", lambda pid: True
        )
        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness._signal_retirement", fake_signal
        )
        monkeypatch.setattr("os.getpid", lambda: 88888)

        new_fd = ensure_ssh_isolated_home_lock(
            ssh_session_token="newcomer_token",
            hermes_home=tmp_path,
        )
        assert new_fd is not None
        assert len(signaled) == 1
        os.close(new_fd)
    finally:
        try:
            os.close(incumbent_fd)
        except OSError:
            pass


def test_e2e_turn_in_flight_preserves_active_state_on_disconnect(
    tmp_path, monkeypatch
):
    """End-to-end test proving that when a turn is in flight:
    1. Client disconnect records state='active', turn_active=True on disk
    2. Newcomer is refused (SystemExit) and NEVER signals the running turn
    """
    from hermes_cli.ssh_isolated_liveness import (
        acquire_ssh_isolated_home_lock,
        ensure_ssh_isolated_home_lock,
        read_ssh_isolated_state,
        start_ssh_isolated_idle_watchdog,
        track_ssh_isolated_ws,
    )

    incumbent_fd = acquire_ssh_isolated_home_lock(tmp_path)
    assert incumbent_fd is not None

    try:
        # Start watchdog with active turn probe
        start_ssh_isolated_idle_watchdog(
            has_ssh_token=True,
            hermes_home=tmp_path,
            turn_probe=lambda: True,
        )

        # Client disconnects while turn is active
        with track_ssh_isolated_ws():
            pass

        # State on disk MUST be active and turn_active=True
        state = read_ssh_isolated_state(tmp_path)
        assert state is not None
        assert state["state"] == "active"
        assert state["turn_active"] is True

        signaled = []
        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness._signal_retirement",
            lambda pid: signaled.append(pid),
        )

        # Newcomer MUST be refused and MUST NOT signal
        with pytest.raises(SystemExit) as exc:
            ensure_ssh_isolated_home_lock(
                ssh_session_token="newcomer_token",
                hermes_home=tmp_path,
            )
        assert "refusing a second writer" in str(exc.value)
        assert signaled == []
    finally:
        os.close(incumbent_fd)


@pytest.mark.live_system_guard_bypass
def test_handover_refuses_when_reused_pid_or_different_flock_holder(
    tmp_path, monkeypatch
):
    """Pin control 1: separate process for reused-PID and different-holder control

    so that _is_positive_flock_holder actually executes rather than short-circuiting on os.getpid().
    """
    import subprocess
    import sys
    from hermes_cli.ssh_isolated_liveness import (
        acquire_ssh_isolated_home_lock,
        ensure_ssh_isolated_home_lock,
        write_ssh_isolated_state,
    )

    # Spawn an actual separate helper process that exits cleanly when stdin closes
    helper = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
    )
    try:
        incumbent_fd = acquire_ssh_isolated_home_lock(tmp_path)
        assert incumbent_fd is not None

        try:
            # Case A: Sidecar points to separate process PID, but create_time is stale (reused PID)
            write_ssh_isolated_state(
                tmp_path,
                pid=helper.pid,
                state="idle",
                active_clients=0,
                turn_active=False,
                create_time=1.0,  # Deliberately mismatched create_time
            )

            signaled = []
            monkeypatch.setattr(
                "hermes_cli.ssh_isolated_liveness._signal_retirement",
                lambda pid: signaled.append(pid),
            )

            with pytest.raises(SystemExit) as exc:
                ensure_ssh_isolated_home_lock(
                    ssh_session_token="newcomer_token",
                    hermes_home=tmp_path,
                )
            assert "refusing a second writer" in str(exc.value)
            assert signaled == []

            # Case B: Sidecar points to separate process PID with correct create_time,
            # but helper process does NOT hold the flock file (different flock holder)
            actual_ctime = 1000.0
            try:
                import psutil
                actual_ctime = psutil.Process(helper.pid).create_time()
            except Exception:
                if sys.platform.startswith("linux") and os.path.isdir(f"/proc/{helper.pid}"):
                    actual_ctime = os.stat(f"/proc/{helper.pid}").st_mtime

            write_ssh_isolated_state(
                tmp_path,
                pid=helper.pid,
                state="idle",
                active_clients=0,
                turn_active=False,
                create_time=actual_ctime,
            )

            with pytest.raises(SystemExit) as exc:
                ensure_ssh_isolated_home_lock(
                    ssh_session_token="newcomer_token",
                    hermes_home=tmp_path,
                )
            assert "refusing a second writer" in str(exc.value)
            assert signaled == []
        finally:
            os.close(incumbent_fd)
    finally:
        try:
            if helper.stdin:
                helper.stdin.close()
            helper.wait(timeout=2.0)
        except Exception:
            helper.kill()


def test_sidecar_publication_reordering_cannot_finish_idle_while_live_clients(
    tmp_path, monkeypatch
):
    """Pin control 3: forced on_close() / on_open() publication reordering cannot

    leave sidecar in 'idle' while live_count() == 1.
    """
    import threading
    from hermes_cli.ssh_isolated_liveness import (
        SshIsolatedIdleTracker,
        read_ssh_isolated_state,
    )

    tracker = SshIsolatedIdleTracker(hermes_home=tmp_path)
    # Start with 1 client
    tracker.on_open()
    st = read_ssh_isolated_state(tmp_path)
    assert st is not None and st["state"] == "active"

    # Simulate slow evaluate_turn_probe during on_close, while concurrent on_open runs
    in_close_probe = threading.Event()
    release_close_probe = threading.Event()

    def slow_probe():
        in_close_probe.set()
        release_close_probe.wait(timeout=2.0)
        return False

    tracker._turn_probe = slow_probe

    # Thread 1: on_close starts and gets paused inside probe
    close_thread = threading.Thread(target=tracker.on_close)
    close_thread.start()
    assert in_close_probe.wait(timeout=2.0)

    # Thread 2: on_open occurs while on_close is stalled
    tracker.on_open()
    mid_st = read_ssh_isolated_state(tmp_path)
    assert mid_st is not None and mid_st["state"] == "active"

    # Now release on_close so it finishes publishing
    release_close_probe.set()
    close_thread.join(timeout=2.0)

    # In-memory count is 1; sidecar MUST remain active, NEVER idle!
    assert tracker.live_count() == 1
    final_st = read_ssh_isolated_state(tmp_path)
    assert final_st is not None
    assert final_st["state"] == "active", f"Sidecar regressed to idle: {final_st}"
    assert final_st["active_clients"] == 1


def test_newcomer_paused_after_idle_observation_aborts_when_incumbent_becomes_active(
    tmp_path, monkeypatch
):
    """Deterministic barrier: pause newcomer after it observes idle, make incumbent

    open a client, then resume; assert no signal/exit and no lock transfer.
    """
    import threading
    from hermes_cli.ssh_isolated_liveness import (
        acquire_ssh_isolated_home_lock,
        ensure_ssh_isolated_home_lock,
        read_ssh_isolated_state,
        SshIsolatedIdleTracker,
    )

    incumbent_fd = acquire_ssh_isolated_home_lock(tmp_path)
    assert incumbent_fd is not None

    try:
        tracker = SshIsolatedIdleTracker(hermes_home=tmp_path, turn_probe=lambda: False)
        # Seed idle state
        tracker.on_open()
        tracker.on_close()
        st = read_ssh_isolated_state(tmp_path)
        assert st is not None and st["state"] == "idle"

        signaled = []
        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness._signal_retirement",
            lambda pid: signaled.append(pid),
        )
        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness._pid_is_alive", lambda pid: True
        )
        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness._is_positive_flock_holder",
            lambda pid, home, state: True,
        )

        # Hook read_ssh_isolated_state so the first read returns idle,
        # then pauses newcomer, incumbent opens a client, and newcomer resumes.
        orig_read = read_ssh_isolated_state
        read_count = 0

        def hooked_read(home):
            nonlocal read_count
            read_count += 1
            res = orig_read(home)
            if read_count == 1:
                # Incumbent becomes active while newcomer is in-flight!
                tracker.on_open()
            return res

        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness.read_ssh_isolated_state",
            hooked_read,
        )
        monkeypatch.setattr("os.getpid", lambda: 88888)

        # Newcomer must detect that incumbent is active, send NO signal, and refuse second writer
        with pytest.raises(SystemExit) as exc:
            ensure_ssh_isolated_home_lock(
                ssh_session_token="newcomer_token",
                hermes_home=tmp_path,
            )
        assert "refusing a second writer" in str(exc.value)
        assert signaled == [], f"Newcomer signaled an active incumbent: {signaled}"
    finally:
        os.close(incumbent_fd)


def test_newcomer_paused_after_idle_observation_aborts_when_turn_starts(
    tmp_path, monkeypatch
):
    """Deterministic barrier: pause newcomer after observing idle, make incumbent

    start an agent turn (or indeterminate probe), resume; assert no signal and no lock transfer.
    """
    from hermes_cli.ssh_isolated_liveness import (
        acquire_ssh_isolated_home_lock,
        ensure_ssh_isolated_home_lock,
        read_ssh_isolated_state,
        SshIsolatedIdleTracker,
        write_ssh_isolated_state,
    )

    incumbent_fd = acquire_ssh_isolated_home_lock(tmp_path)
    assert incumbent_fd is not None

    try:
        tracker = SshIsolatedIdleTracker(hermes_home=tmp_path, turn_probe=lambda: False)
        tracker.on_open()
        tracker.on_close()
        st = read_ssh_isolated_state(tmp_path)
        assert st is not None and st["state"] == "idle"

        signaled = []
        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness._signal_retirement",
            lambda pid: signaled.append(pid),
        )
        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness._pid_is_alive", lambda pid: True
        )
        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness._is_positive_flock_holder",
            lambda pid, home, state: True,
        )

        orig_read = read_ssh_isolated_state
        read_count = 0

        def hooked_read(home):
            nonlocal read_count
            read_count += 1
            res = orig_read(home)
            if read_count == 1:
                # Incumbent starts a turn while newcomer is in-flight
                write_ssh_isolated_state(
                    tmp_path,
                    pid=os.getpid(),
                    state="active",
                    active_clients=0,
                    turn_active=True,
                )
            return res

        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness.read_ssh_isolated_state",
            hooked_read,
        )
        monkeypatch.setattr("os.getpid", lambda: 88888)

        with pytest.raises(SystemExit) as exc:
            ensure_ssh_isolated_home_lock(
                ssh_session_token="newcomer_token",
                hermes_home=tmp_path,
            )
        assert "refusing a second writer" in str(exc.value)
        assert signaled == []
    finally:
        os.close(incumbent_fd)


def test_failed_sidecar_refresh_unlinks_prior_state_and_refuses_handover(
    tmp_path, monkeypatch
):
    """Pin control 2: a failed sidecar refresh must not leave the prior generation

    reclaim-authoritative; the stale sidecar is unlinked and subsequent contention fails closed.
    """
    from hermes_cli.ssh_isolated_liveness import (
        SSH_ISOLATED_STATE_NAME,
        acquire_ssh_isolated_home_lock,
        ensure_ssh_isolated_home_lock,
        read_ssh_isolated_state,
        write_ssh_isolated_state,
    )

    # Prior generation left an idle state file
    write_ssh_isolated_state(
        tmp_path,
        pid=os.getpid(),
        state="idle",
        active_clients=0,
        turn_active=False,
    )
    assert read_ssh_isolated_state(tmp_path) is not None

    # New generation acquires lock, but write_ssh_isolated_state fails (e.g. disk error)
    def fail_write(*args, **kwargs):
        return False

    monkeypatch.setattr(
        "hermes_cli.ssh_isolated_liveness.write_ssh_isolated_state", fail_write
    )

    # First process acquires lock
    first_fd = ensure_ssh_isolated_home_lock(
        ssh_session_token="token_gen2",
        hermes_home=tmp_path,
    )
    assert first_fd is not None

    try:
        # State file must have been unlinked so prior generation does not remain authoritative
        assert not (tmp_path / SSH_ISOLATED_STATE_NAME).exists()
        assert read_ssh_isolated_state(tmp_path) is None

        # Subsequent newcomer encounters lock contention: no sidecar -> cannot reclaim
        signaled = []
        monkeypatch.setattr(
            "hermes_cli.ssh_isolated_liveness._signal_retirement",
            lambda pid: signaled.append(pid),
        )
        with pytest.raises(SystemExit) as exc:
            ensure_ssh_isolated_home_lock(
                ssh_session_token="newcomer_token",
                hermes_home=tmp_path,
            )
        assert "refusing a second writer" in str(exc.value)
        assert signaled == []
    finally:
        os.close(first_fd)


def test_consume_handover_retirement_boundary_validation(tmp_path):
    """Pin control: incumbent consume_handover_retirement performs authoritative

    in-memory client count and fail-closed turn probe rechecks at the shutdown boundary.
    """
    from hermes_cli.ssh_isolated_liveness import (
        SshIsolatedIdleTracker,
        consume_handover_retirement,
    )

    class DummyServer:
        should_exit = False

    server = DummyServer()
    turn_active = False

    def probe():
        return turn_active

    tracker = SshIsolatedIdleTracker(hermes_home=tmp_path, turn_probe=probe)

    # 1. Client active in memory -> refused
    tracker.on_open()
    assert consume_handover_retirement(tracker, server=server) is False
    assert server.should_exit is False

    # 2. Client disconnected, but turn active -> refused
    tracker.on_close()
    turn_active = True
    assert consume_handover_retirement(tracker, server=server) is False
    assert server.should_exit is False

    # 3. Turn probe raises exception (indeterminate) -> fails closed, refused
    def failing_probe():
        raise RuntimeError("probe boom")

    tracker._turn_probe = failing_probe
    assert consume_handover_retirement(tracker, server=server) is False
    assert server.should_exit is False

    # 4. Genuinely idle (no clients, turn probe False) -> accepted, triggers exit
    tracker._turn_probe = lambda: False
    assert consume_handover_retirement(tracker, server=server) is True
    assert server.should_exit is True


def test_handover_signal_when_tracker_lock_held_on_main_thread_does_not_deadlock(
    tmp_path,
):
    """Hold tracker lock on main thread, deliver handover signal,

    and prove no deadlock occurs and handover is rejected when live client is present.
    """
    import signal
    from hermes_cli.ssh_isolated_liveness import (
        start_ssh_isolated_idle_watchdog,
        write_ssh_isolated_handover_request,
    )

    class DummyServer:
        should_exit = False

    server = DummyServer()
    tracker = start_ssh_isolated_idle_watchdog(
        has_ssh_token=True,
        hermes_home=tmp_path,
        server=server,
        turn_probe=lambda: False,
    )
    assert tracker is not None
    tracker.on_open()

    write_ssh_isolated_handover_request(tmp_path, nonce="test-nonce", seq=tracker._seq)

    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler)

    # In main thread, hold tracker._lock and invoke signal handler
    with tracker._lock:
        handler(signal.SIGTERM, None)

    # Prove background watchdog safely consumed and adjudicated the request without deadlock
    assert tracker._handover_done.wait(timeout=2.0) is True
    assert server.should_exit is False


def test_ordinary_sigterm_without_handover_request_shuts_down_even_when_active(
    tmp_path,
):
    """Deliver an ordinary SIGTERM with no handover request while active

    and prove normal uvicorn shutdown still occurs.
    """
    import signal
    from hermes_cli.ssh_isolated_liveness import (
        start_ssh_isolated_idle_watchdog,
    )

    orig_called = []

    def fake_uvicorn_handler(signum, frame):
        orig_called.append(signum)

    old_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, fake_uvicorn_handler)
    try:
        class DummyServer:
            should_exit = False

        server = DummyServer()
        tracker = start_ssh_isolated_idle_watchdog(
            has_ssh_token=True,
            hermes_home=tmp_path,
            server=server,
            turn_probe=lambda: False,
        )
        assert tracker is not None
        tracker.on_open()  # active client

        current_handler = signal.getsignal(signal.SIGTERM)
        current_handler(signal.SIGTERM, None)

        assert orig_called == [signal.SIGTERM]
    finally:
        signal.signal(signal.SIGTERM, old_handler)







