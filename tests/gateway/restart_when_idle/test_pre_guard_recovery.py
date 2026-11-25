"""A supervised restart whose stop fails before the shutdown watchdog starts (#160095)."""
import asyncio
import os
import logging
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from gateway import run_shutdown, shutdown_watchdog
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_failure_at_stop_entry_does_not_replace_another_active_stop(monkeypatch, caplog):
    runner, _ = make_restart_runner()
    existing = asyncio.create_task(asyncio.Event().wait())
    runner._stop_task = existing
    runner._stop_loop_liveness_guards = Mock(side_effect=RuntimeError('entry failed'))
    arm = Mock()
    monkeypatch.setattr(shutdown_watchdog, 'arm_shutdown_watchdog', arm)
    try:
        with caplog.at_level(logging.ERROR):
            assert runner.request_restart(detached=False, via_service=True)
            await runner._restart_task
        arm.assert_not_called()
        assert runner._stop_task is existing
        assert any(r.exc_info and str(r.exc_info[1]) == 'entry failed' for r in caplog.records)
    finally:
        existing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await existing


def enable_watchdogs(monkeypatch, tmp_path):
    monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
    monkeypatch.setenv('HERMES_MANAGED_DIR', str(tmp_path / 'managed'))


@pytest.fixture(autouse=True)
def no_process_exit(monkeypatch):
    exit_process = Mock(side_effect=AssertionError('unexpected process exit'))
    monkeypatch.setattr(shutdown_watchdog, '_hard_exit', exit_process)
    return exit_process


@pytest.mark.asyncio
async def test_failure_before_the_stop_guard_is_logged_and_covered(monkeypatch, tmp_path, caplog):
    enable_watchdogs(monkeypatch, tmp_path)
    runner, _ = make_restart_runner()
    fault = RuntimeError('deadline setup failed')
    monkeypatch.setattr(run_shutdown, '_effective_watchdog_leash', Mock(side_effect=fault))
    arm = Mock(return_value=threading.Event())
    monkeypatch.setattr(shutdown_watchdog, 'arm_shutdown_watchdog', arm)
    with caplog.at_level(logging.ERROR):
        assert runner.request_restart(detached=False, via_service=True)
        await runner._restart_task
    assert runner._draining
    assert arm.call_count == 1
    assert sum(bool(r.exc_info and r.exc_info[1] is fault) for r in caplog.records) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('replacement_starts', [True, False])
async def test_stop_takes_over_only_after_starting_its_watchdog(monkeypatch, tmp_path, replacement_starts):
    enable_watchdogs(monkeypatch, tmp_path)
    runner, _ = make_restart_runner()
    runner._restart_drain_timeout = 180
    old = threading.Event()
    monkeypatch.setattr(shutdown_watchdog, 'arm_shutdown_watchdog', Mock(return_value=old))
    normal_leash = run_shutdown._effective_watchdog_leash
    monkeypatch.setattr(run_shutdown, '_effective_watchdog_leash', Mock(side_effect=RuntimeError('stop setup failed')))
    assert runner.request_restart(detached=False, via_service=True)
    await runner._restart_task
    monkeypatch.setattr(run_shutdown, '_effective_watchdog_leash', normal_leash)

    original_arm = run_shutdown.arm_shutdown_watchdog
    start_thread = threading.Thread.start

    def refuse_replacement(thread):
        if thread.name == 'gateway-shutdown-watchdog':
            raise RuntimeError('replacement did not start')
        return start_thread(thread)

    def normal_arm(delay_s, *, done_event, **kwargs):
        assert not old.is_set(), 'retiring the only timer before starting its replacement loses recovery'
        assert delay_s > 60
        if not replacement_starts:
            # Exercise the real arm contract: its best-effort default swallows Thread.start() refusal.
            return original_arm(delay_s, done_event=done_event, **kwargs)
        return done_event

    if not replacement_starts:
        monkeypatch.setattr(threading.Thread, 'start', refuse_replacement)

    monkeypatch.setattr(run_shutdown, 'arm_shutdown_watchdog', normal_arm)
    inside = asyncio.Event()

    async def pause(_runner, _ctx):
        inside.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(type(runner), '_stop_begin_teardown', pause)
    if not replacement_starts:
        with pytest.raises(RuntimeError, match='replacement did not start'):
            await asyncio.wait_for(runner.stop(), 2)
        assert not old.is_set()
        return
    stopping = asyncio.create_task(runner.stop())
    try:
        await asyncio.wait_for(inside.wait(), 5)
        assert old.is_set(), 'the old deadline can still interrupt this stop'
        assert not runner._shutdown_watchdog_done.is_set()
    finally:
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping


@pytest.mark.asyncio
async def test_thread_start_refusal_exits_without_claiming_to_be_armed(monkeypatch, tmp_path, caplog, no_process_exit):
    enable_watchdogs(monkeypatch, tmp_path)
    runner, _ = make_restart_runner()
    monkeypatch.setattr(run_shutdown, '_effective_watchdog_leash', Mock(side_effect=RuntimeError('stop setup failed')))
    start = threading.Thread.start
    refused = []

    def refuse_watchdog(thread):
        if thread.name == 'gateway-restart-failure-exit':
            refused.append(thread.name)
            raise RuntimeError('thread start refused')
        return start(thread)

    monkeypatch.setattr(threading.Thread, 'start', refuse_watchdog)
    no_process_exit.side_effect = None  # capture the terminal call instead of exiting the pytest worker
    with caplog.at_level(logging.ERROR):
        assert runner.request_restart(detached=False, via_service=True)
        await runner._restart_task
    assert refused == ['gateway-restart-failure-exit']
    no_process_exit.assert_called_once_with(1)
    assert not any('watchdog is armed' in r.getMessage() for r in caplog.records)
    assert any(r.exc_info and str(r.exc_info[1]) == 'stop setup failed' for r in caplog.records)



_RESTART_CHILD = '''
import asyncio, logging, os, sys, threading
deadline = threading.Timer(30.0, os._exit, args=(97,))
deadline.daemon = True
deadline.start()
repo_root = sys.argv[1]
sys.path.insert(0, repo_root)

import hermes_cli.env_loader as env_loader
env_loader.load_hermes_dotenv = lambda **_kwargs: []

from gateway import run_shutdown, shutdown_watchdog
from tests.gateway.restart_test_helpers import make_restart_runner

logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)
shutdown_watchdog.DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S = 0.5  # a short leash for the test


async def main():
    runner, _adapter = make_restart_runner()

    def boom():
        raise ImportError("cannot import name 'x' from 'gateway.run_startup_liveness'")
    run_shutdown._effective_watchdog_leash = lambda _runner: boom()
    if sys.argv[2] == "thread_refused":
        start = threading.Thread.start
        def refuse(thread):
            if thread.name == "gateway-restart-failure-exit":
                raise RuntimeError("test refused the recovery thread")
            return start(thread)
        threading.Thread.start = refuse
    runner.request_restart(detached=False, via_service=True)
    try:
        await runner._restart_task
    except Exception as exc:
        print("restart ended:", type(exc).__name__, flush=True)
    await asyncio.sleep(10)
    print("still running", flush=True)


asyncio.run(main())
'''


@pytest.mark.parametrize("ending", ["timer_fires", "thread_refused"])
def test_a_restart_that_fails_after_admission_closed_still_exits(tmp_path, ending):
    child = tmp_path / "restart_child.py"
    child.write_text(_RESTART_CHILD, encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("terminal:\n  backend: local\n", encoding="utf-8")
    managed = tmp_path / "managed"
    managed.mkdir()
    env = {key: value for key, value in os.environ.items() if key != "PYTEST_CURRENT_TEST"}  # arms it
    env.update(HERMES_HOME=str(home), HERMES_MANAGED_DIR=str(managed), HERMES_GATEWAY_LOCK_DIR=str(tmp_path / "locks"),
               PYTHONIOENCODING="utf-8")
    repo_root = str(Path(__file__).resolve().parents[3])
    proc = subprocess.run([sys.executable, str(child), repo_root, ending], cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, encoding="utf-8", errors="replace", timeout=120, check=False)

    output = f"{proc.stdout}\n{proc.stderr[-3000:]}"
    assert "still running" not in proc.stdout, output
    assert proc.returncode == 1, output
    assert "Restart stop failed before its teardown began" in proc.stderr, output
    assert "ImportError: cannot import name 'x'" in proc.stderr, output
    if ending == "thread_refused":
        assert "Cannot arm restart recovery watchdog; exiting now" in proc.stderr, output
        assert "test refused the recovery thread" in proc.stderr, output
        assert "watchdog is armed" not in proc.stderr, output
    else:
        assert "Shutdown watchdog fired" in proc.stderr, output  # the watchdog's exit, not a crash
