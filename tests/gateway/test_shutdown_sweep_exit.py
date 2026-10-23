"""Real asyncio.run/executor exit regressions for the tool sweep (#140162)."""

import os
import subprocess
import sys
import textwrap

import pytest


_CHILD = textwrap.dedent('''
    import asyncio
    import contextlib
    import os
    from pathlib import Path
    import threading
    import time
    from unittest.mock import AsyncMock, patch

    from gateway.run import GatewayRunner
    import gateway.run as run
    from gateway.run_shutdown import GatewayShutdownMixin
    from tests.gateway.restart_test_helpers import make_restart_runner

    os.environ.pop("PYTEST_CURRENT_TEST", None)
    mode = os.environ["SWEEP_TEST_MODE"]
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    run._hermes_home = home

    async def main():
        runner, _ = make_restart_runner()
        started = asyncio.Event()
        loop = asyncio.get_running_loop()
        calls = []
        release_worker = threading.Event()
        worker_done = threading.Event()

        def sweep(phase):
            calls.append(phase)
            print("sweep_started", phase, flush=True)
            loop.call_soon_threadsafe(started.set)
            if mode != "normal" and phase == "post-drain":
                try:
                    release_worker.wait()
                    if mode == "late-exception":
                        raise RuntimeError("late sweep failure")
                finally:
                    worker_done.set()
            return []

        async def begin(self, ctx):
            ctx.started_at = time.monotonic()
            self._running = False

        async def finalize(self, ctx):
            if mode.startswith("late"):
                release_worker.set()
                assert await asyncio.to_thread(worker_done.wait, 5)

        async def release(self, ctx):
            await self._stop_kill_tool_subprocesses_off_loop("final-cleanup")

        with (
            patch("gateway.run_shutdown._effective_watchdog_leash", return_value=2.0),
            patch.object(GatewayShutdownMixin, "_TOOL_SUBPROCESS_SWEEP_TIMEOUT_S", 0.05),
            patch.object(GatewayShutdownMixin, "_stop_kill_tool_subprocesses", side_effect=sweep),
            patch.object(GatewayRunner, "_stop_begin_teardown", begin),
            patch.object(GatewayRunner, "_stop_drain_active_work", AsyncMock()),
            patch.object(GatewayRunner, "_stop_finalize_agents_and_adapters", finalize),
            patch.object(GatewayRunner, "_stop_release_runtime_state", release),
            patch.object(GatewayRunner, "_stop_quiesce_and_close_session_dbs"),
            patch("gateway.status.flush_runtime_status_async", AsyncMock(return_value=True)),
            patch("gateway.run._shutdown_gateway_health_export"),
        ):
            if mode == "cancelled":
                request = asyncio.create_task(runner.stop())
                await asyncio.wait_for(started.wait(), timeout=5)
                runner._stop_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await request
            else:
                await asyncio.gather(runner.stop(), runner.stop())
            print("stop_returned", runner._shutdown_watchdog_done.is_set(), flush=True)
            print("clean_marker", (home / ".clean_shutdown").exists(), flush=True)
            print("sweep_calls", len(calls), flush=True)

    asyncio.run(main())
    print("asyncio_run_returned", flush=True)
''')


@pytest.mark.parametrize("mode", ["normal", "blocked", "cancelled", "late", "late-exception"])
def test_shutdown_sweep_has_bounded_process_exit(tmp_path, mode):
    """The real watchdog must survive stop() while a real executor worker is stuck."""
    env = dict(os.environ, HERMES_HOME=str(tmp_path), SWEEP_TEST_MODE=mode)
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", _CHILD],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        stdout, stderr = child.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        child.kill()
        stdout, stderr = child.communicate(timeout=5)
        pytest.fail(f"asyncio.run hung beyond the watchdog deadline:\n{stdout}\n{stderr}")
    assert "sweep_started post-drain" in stdout, (stdout, stderr)
    if mode == "normal":
        assert child.returncode == 0, (stdout, stderr)
        assert "stop_returned True" in stdout
        assert "clean_marker True" in stdout
        assert "sweep_calls 2" in stdout
        assert "asyncio_run_returned" in stdout
    elif mode.startswith("late"):
        assert child.returncode == 0, (stdout, stderr)
        assert "stop_returned True" in stdout
        assert "clean_marker False" in stdout
        assert "sweep_calls 2" in stdout
        assert "asyncio_run_returned" in stdout
        if mode == "late-exception":
            assert "late sweep failure" in stderr
    else:
        assert child.returncode == 1, (stdout, stderr)
        assert "stop_returned False" in stdout
        assert "clean_marker False" in stdout
        assert "sweep_calls 1" in stdout
        assert "asyncio_run_returned" not in stdout
        assert (tmp_path / "logs" / "gateway-shutdown-watchdog.log").is_file()
