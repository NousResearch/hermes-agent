"""A dead session listener must never leave a ready gateway behind."""
import asyncio

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('listener_failure', [False, True])
async def test_runtime_wait_observes_listener_exit(tmp_path, listener_failure):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway import run_runtime
    from gateway.runtime_ownership import process_ownership
    from hermes_constants import get_hermes_home

    process_ownership.reserve([get_hermes_home()])
    runner = GatewayRunner(GatewayConfig())
    waiter = None
    try:
        await run_runtime.initialize_gateway_runtime(runner)
        from gateway.control_socket import GatewayControlServer
        runner.session_control_server = GatewayControlServer()
        assert await runner.session_control_server.start()
        await run_runtime.start_gateway_runtime_api(runner)
        assert await runner.start()
        run_runtime.publish_gateway_runtime_ready(runner)
        waiter = asyncio.create_task(run_runtime.wait_gateway_runtime(runner))
        if listener_failure:
            # Stop the actual listener independently of the runner lifecycle.
            runner.session_api.server.should_exit = True
            with pytest.raises(RuntimeError, match='session API stopped unexpectedly'):
                await asyncio.wait_for(waiter, 10)
            assert runner.session_runtime_descriptor['state'] != 'ready'
            assert not runner.session_runtime_descriptor['capabilities']
        else:
            await runner.stop()
            await asyncio.wait_for(waiter, 10)
    finally:
        if waiter is not None and not waiter.done():
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        await runner.stop()
        await runner.session_control_server.stop()
        process_ownership.close()


@pytest.mark.asyncio
async def test_listener_failure_stops_bootstrap_background_threads(monkeypatch):
    import threading
    from gateway import run_bootstrap, run_runtime
    from gateway.config import GatewayConfig
    from hermes_constants import get_hermes_home

    get_hermes_home().chmod(0o700)
    stopped = threading.Event()
    threads = []

    def start_owned_threads(runner):
        for _ in range(2):
            thread = threading.Thread(target=stopped.wait, daemon=True)
            thread.start()
            threads.append(thread)
        return stopped, None, *threads

    publish = run_runtime.publish_gateway_runtime_ready

    def fail_after_ready(runner):
        publish(runner)
        runner.session_api.server.should_exit = True

    monkeypatch.setattr(run_bootstrap, '_start_gateway_start_cron_and_housekeeping', start_owned_threads)
    monkeypatch.setattr(run_runtime, 'publish_gateway_runtime_ready', fail_after_ready)
    try:
        with pytest.raises(RuntimeError, match='session API stopped unexpectedly'):
            await asyncio.wait_for(run_bootstrap.start_gateway(GatewayConfig()), 20)
        assert threads
        assert stopped.is_set(), 'listener failure skipped background-writer shutdown'
        assert all(not thread.is_alive() for thread in threads)
    finally:
        stopped.set()
        for thread in threads:
            thread.join(timeout=2)
