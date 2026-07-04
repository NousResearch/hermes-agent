import asyncio
import threading
import time

import pytest

from gateway import run as gateway_run


class _FakeParent:
    def mkdir(self, *args, **kwargs):
        return None


class _FakeHeartbeatFile:
    parent = _FakeParent()

    def __init__(self):
        self.touches = 0

    def touch(self, *args, **kwargs):
        self.touches += 1


def _wait_until(predicate, timeout=0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class _ThreadProbe:
    def __init__(self):
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.thread = None

    def run(self, stop_event, *args, **kwargs):
        self.thread = threading.current_thread()
        self.started.set()
        stop_event.wait()
        self.stopped.set()


class _RecordingCronProvider:
    def __init__(self):
        self.stop_called = threading.Event()
        self.probe = _ThreadProbe()

    def start(self, stop_event, *, adapters=None, loop=None, interval=60):
        self.probe.run(stop_event)

    def stop(self):
        self.stop_called.set()


def _patch_start_gateway_basics(
    monkeypatch, tmp_path, cron_provider, housekeeping_probe, planned_stop_probe
):
    cleanup_calls = {"mcp_shutdown": 0}

    def shutdown_mcp_servers():
        cleanup_calls["mcp_shutdown"] += 1

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr("gateway.code_skew.record_boot_fingerprint", lambda: None)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("gateway.status.acquire_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr("gateway.status.write_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.release_gateway_runtime_lock", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: None)
    monkeypatch.setattr(gateway_run, "_ensure_windows_gateway_venv_imports", lambda: None)
    monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", lambda: None)
    monkeypatch.setattr("tools.mcp_tool.shutdown_mcp_servers", shutdown_mcp_servers)
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: cron_provider)
    monkeypatch.setattr(gateway_run, "_start_gateway_housekeeping", housekeeping_probe.run)
    monkeypatch.setattr(gateway_run, "_run_planned_stop_watcher", planned_stop_probe.run)
    return cleanup_calls


def _capture_heartbeat_threads(monkeypatch):
    original = gateway_run._start_heartbeat_bumper
    threads = []

    def wrapped(stop_event, hb_file, interval=30, loop_alive=None):
        threads.append(threading.current_thread())
        return original(stop_event, hb_file, interval=0.01, loop_alive=loop_alive)

    monkeypatch.setattr(gateway_run, "_start_heartbeat_bumper", wrapped)
    return threads


def _heartbeat_path(tmp_path):
    return tmp_path / "cron" / gateway_run._GATEWAY_HEARTBEAT_FILENAME


def _assert_heartbeat_stopped(threads, heartbeat_path):
    assert threads
    assert all(not thread.is_alive() for thread in threads)
    assert heartbeat_path.exists()
    modified_at = heartbeat_path.stat().st_mtime_ns
    time.sleep(0.03)
    assert heartbeat_path.stat().st_mtime_ns == modified_at


def _assert_backgrounds_stopped(
    cron_provider, housekeeping_probe, planned_stop_probe, cleanup_calls
):
    assert cron_provider.stop_called.is_set()
    for probe in (cron_provider.probe, housekeeping_probe, planned_stop_probe):
        assert probe.started.is_set()
        assert probe.stopped.is_set()
        assert probe.thread is not None
        assert not probe.thread.is_alive()
    assert cleanup_calls["mcp_shutdown"] == 1


def test_heartbeat_bumper_skips_touches_when_loop_is_not_alive():
    stop = threading.Event()
    hb_file = _FakeHeartbeatFile()

    thread = threading.Thread(
        target=gateway_run._start_heartbeat_bumper,
        args=(stop, hb_file),
        kwargs={"interval": 0.01, "loop_alive": lambda: False},
        daemon=True,
    )
    thread.start()

    assert _wait_until(lambda: hb_file.touches == 1)
    time.sleep(0.05)

    stop.set()
    thread.join(timeout=0.2)

    assert not thread.is_alive()
    assert hb_file.touches == 1


def test_heartbeat_bumper_resumes_touches_when_loop_is_alive():
    stop = threading.Event()
    hb_file = _FakeHeartbeatFile()
    alive = threading.Event()

    def loop_alive():
        return alive.is_set()

    thread = threading.Thread(
        target=gateway_run._start_heartbeat_bumper,
        args=(stop, hb_file),
        kwargs={"interval": 0.01, "loop_alive": loop_alive},
        daemon=True,
    )
    thread.start()

    assert _wait_until(lambda: hb_file.touches == 1)
    time.sleep(0.03)
    assert hb_file.touches == 1

    alive.set()
    assert _wait_until(lambda: hb_file.touches >= 2)

    stop.set()
    thread.join(timeout=0.2)
    assert not thread.is_alive()


@pytest.mark.asyncio
async def test_start_gateway_stops_backgrounds_after_clean_shutdown(monkeypatch, tmp_path):
    cron_provider = _RecordingCronProvider()
    housekeeping_probe = _ThreadProbe()
    planned_stop_probe = _ThreadProbe()
    cleanup_calls = _patch_start_gateway_basics(
        monkeypatch, tmp_path, cron_provider, housekeeping_probe, planned_stop_probe
    )
    threads = _capture_heartbeat_threads(monkeypatch)

    class CleanRunner:
        def __init__(self, config):
            self.config = config
            self.adapters = {}
            self._running = True
            self.should_exit_cleanly = False
            self.should_exit_with_failure = False
            self.exit_reason = None
            self.exit_code = None
            self._restart_requested = False
            self._restart_via_service = False

        async def start(self):
            return True

        async def wait_for_shutdown(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(gateway_run, "GatewayRunner", CleanRunner)

    assert await gateway_run.start_gateway(config=None, replace=False, verbosity=None) is True

    _assert_heartbeat_stopped(threads, _heartbeat_path(tmp_path))
    _assert_backgrounds_stopped(
        cron_provider, housekeeping_probe, planned_stop_probe, cleanup_calls
    )


@pytest.mark.asyncio
async def test_start_gateway_stops_backgrounds_after_failed_shutdown(monkeypatch, tmp_path):
    cron_provider = _RecordingCronProvider()
    housekeeping_probe = _ThreadProbe()
    planned_stop_probe = _ThreadProbe()
    cleanup_calls = _patch_start_gateway_basics(
        monkeypatch, tmp_path, cron_provider, housekeeping_probe, planned_stop_probe
    )
    threads = _capture_heartbeat_threads(monkeypatch)

    class FailedRunner:
        def __init__(self, config):
            self.config = config
            self.adapters = {}
            self._running = True
            self.should_exit_cleanly = False
            self.should_exit_with_failure = False
            self.exit_reason = None
            self.exit_code = None
            self._restart_requested = False
            self._restart_via_service = False

        async def start(self):
            return True

        async def wait_for_shutdown(self):
            self.should_exit_with_failure = True
            self.exit_reason = "shutdown failed"

        async def stop(self):
            return None

    monkeypatch.setattr(gateway_run, "GatewayRunner", FailedRunner)

    assert await gateway_run.start_gateway(config=None, replace=False, verbosity=None) is False

    _assert_heartbeat_stopped(threads, _heartbeat_path(tmp_path))
    _assert_backgrounds_stopped(
        cron_provider, housekeeping_probe, planned_stop_probe, cleanup_calls
    )


@pytest.mark.asyncio
async def test_start_gateway_stops_backgrounds_when_shutdown_wait_is_cancelled(
    monkeypatch, tmp_path
):
    cron_provider = _RecordingCronProvider()
    housekeeping_probe = _ThreadProbe()
    planned_stop_probe = _ThreadProbe()
    cleanup_calls = _patch_start_gateway_basics(
        monkeypatch, tmp_path, cron_provider, housekeeping_probe, planned_stop_probe
    )
    threads = _capture_heartbeat_threads(monkeypatch)
    wait_started = asyncio.Event()

    class BlockingRunner:
        def __init__(self, config):
            self.config = config
            self.adapters = {}
            self._running = True
            self.should_exit_cleanly = False
            self.should_exit_with_failure = False
            self.exit_reason = None
            self.exit_code = None
            self._restart_requested = False
            self._restart_via_service = False

        async def start(self):
            return True

        async def wait_for_shutdown(self):
            wait_started.set()
            await asyncio.Event().wait()

        async def stop(self):
            return None

    monkeypatch.setattr(gateway_run, "GatewayRunner", BlockingRunner)

    task = asyncio.create_task(gateway_run.start_gateway(config=None, replace=False, verbosity=None))
    await asyncio.wait_for(wait_started.wait(), timeout=1)
    assert _wait_until(lambda: threads)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    _assert_heartbeat_stopped(threads, _heartbeat_path(tmp_path))
    _assert_backgrounds_stopped(
        cron_provider, housekeeping_probe, planned_stop_probe, cleanup_calls
    )
