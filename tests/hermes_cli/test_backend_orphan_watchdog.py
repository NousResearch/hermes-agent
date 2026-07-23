import psutil

from hermes_cli import main as hermes_main


def test_is_orphaned_true_when_ppid_changes():
    # Parent went away and we were reparented to a subreaper/init.
    assert hermes_main._backend_is_orphaned(1234, 1.0, getppid=lambda: 999999) is True


def test_is_orphaned_true_when_parent_create_time_mismatch():
    # Same ppid but a different create_time means the PID was reused.
    me = psutil.Process()
    assert hermes_main._backend_is_orphaned(me.pid, 0.0, getppid=lambda: me.pid) is True


def test_is_orphaned_false_when_parent_alive_and_matches():
    me = psutil.Process()
    assert (
        hermes_main._backend_is_orphaned(me.pid, me.create_time(), getppid=lambda: me.pid)
        is False
    )


class _CapturingThread:
    """Captures the watchdog's loop target so the test can drive it directly."""

    captured = None

    def __init__(self, target=None, **kwargs):
        type(self).captured = target
        self.kwargs = kwargs

    def start(self):
        pass


def test_watchdog_noop_without_desktop_env(monkeypatch):
    # Standalone `hermes serve` (no HERMES_DESKTOP) must NOT self-terminate when
    # its launching shell exits — arming here would break nohup/systemd/`&`.
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    _CapturingThread.captured = None
    monkeypatch.setattr(hermes_main.threading, "Thread", _CapturingThread)
    hermes_main._arm_desktop_orphan_watchdog()
    assert _CapturingThread.captured is None


def test_watchdog_arms_daemon_thread_under_desktop(monkeypatch):
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    _CapturingThread.captured = None
    monkeypatch.setattr(hermes_main.threading, "Thread", _CapturingThread)
    hermes_main._arm_desktop_orphan_watchdog()
    assert callable(_CapturingThread.captured)


def test_watchdog_loop_exits_process_once_orphaned(monkeypatch):
    # Drive the real loop target: poll twice while the parent is alive, then
    # report orphaned and assert the loop reaches os._exit(0).
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    _CapturingThread.captured = None
    monkeypatch.setattr(hermes_main.threading, "Thread", _CapturingThread)

    orphaned = iter([False, False, True])
    monkeypatch.setattr(hermes_main, "_backend_is_orphaned", lambda *_a, **_k: next(orphaned))

    sleeps = []
    monkeypatch.setattr(hermes_main._time, "sleep", lambda s: sleeps.append(s))

    exited = []

    def _fake_exit(code):
        exited.append(code)
        raise SystemExit(code)  # stop the loop the way os._exit would

    monkeypatch.setattr(hermes_main.os, "_exit", _fake_exit)

    hermes_main._arm_desktop_orphan_watchdog(poll_s=0.01)
    loop = _CapturingThread.captured
    assert callable(loop)

    try:
        loop()
    except SystemExit:
        pass

    assert exited == [0]  # exit branch reached
    # Polled twice while the parent was still alive — it never exits early.
    assert sleeps == [0.01, 0.01]
