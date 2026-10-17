"""A stopping ``hermes serve`` must say what it is still draining on (#140411).

Reported shape: ``systemctl stop hermes-serve`` sat out the whole ``TimeoutStopSec`` (180 s) and was
SIGKILLed, in-process only ``ERROR: Cancel 2 running task(s), timeout graceful shutdown exceeded``,
and the journal named nothing else — a multi-minute 502 outage for a remote Desktop backend with no
way to see what held the drain without py-spy on the dying process. These tests pin the reporter
that fills that gap: one line per interval with the pending counts/names, and one all-threads stack
dump once the drain outlives the threshold.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_cli import serve_drain_report as drain


class _FakeServerState:
    def __init__(self, tasks=(), connections=()):
        self.tasks = list(tasks)
        self.connections = list(connections)


class _FakeServer:
    """The two attributes the reporter reads, plus uvicorn's ``handle_exit`` seam."""

    def __init__(self, tasks=(), connections=()):
        self.server_state = _FakeServerState(tasks, connections)
        self.exit_calls: list[int] = []

    def handle_exit(self, signum, frame=None):  # uvicorn's signature
        self.exit_calls.append(int(signum))


class _FakeLoop:
    def __init__(self, tasks):
        self._tasks = tasks


@pytest.fixture
def fake_loop_tasks(monkeypatch):
    """Install N pending loop tasks with recognisable coroutine names (no running loop needed)."""
    holder: dict[str, list] = {}

    def _install(count: int) -> None:
        tasks = [
            SimpleNamespace(
                done=lambda: False,  # asyncio.all_tasks only calls done()
                get_name=lambda i=index: f"Task-{i}",
                get_coro=lambda i=index: SimpleNamespace(__qualname__=f"endpoint_{i}"),
            )
            for index in range(count)
        ]
        holder["tasks"] = tasks
        monkeypatch.setattr(asyncio, "all_tasks", lambda loop=None: list(holder["tasks"]))

    yield _install
    monkeypatch.undo()


def test_pending_work_counts_asgi_tasks_connections_and_loop_tasks(fake_loop_tasks):
    fake_loop_tasks(2)
    server = _FakeServer(tasks=[object(), object(), object()], connections=[object()])

    work = drain.pending_work(server, _FakeLoop([]))

    assert work["asgi_tasks"] == 3
    assert work["connections"] == 1
    assert work["loop_tasks"] == 2
    # uvicorn's per-request tasks are unnamed: the coroutine is the part that identifies them.
    assert work["loop_task_names"] == ["Task-0:endpoint_0", "Task-1:endpoint_1"]


def test_pending_work_survives_a_missing_server_and_loop():
    work = drain.pending_work(None, None)

    assert work["asgi_tasks"] == 0 and work["connections"] == 0 and work["loop_tasks"] == 0
    assert work["running_sessions"] == 0


def test_format_drain_progress_names_the_counts_and_truncates_the_task_list():
    work = {
        "asgi_tasks": 2,
        "connections": 1,
        "running_sessions": 1,
        "loop_tasks": 9,
        "loop_task_names": [f"Task-{index}:endpoint" for index in range(9)],
    }

    line = drain.format_drain_progress(137.6, work)

    assert "still draining after 137s" in line
    assert "2 ASGI task(s)" in line and "1 open connection(s)" in line
    assert "1 running turn(s)" in line
    assert "9 loop task(s): Task-0:endpoint" in line
    assert "+3 more" in line, "a long task list must be summarised, not dumped into the journal"


def test_reporter_logs_each_interval_and_dumps_stacks_once_past_the_threshold(monkeypatch, caplog):
    """The reported 180 s drain: several lines, then exactly one stack dump — never a per-tick dump."""
    clock = {"now": 0.0}
    dumps: list[int] = []
    reporter = drain.DrainReporter(
        server=_FakeServer(tasks=[object()]), loop=None,
        interval=0.01, stack_dump_after=5.0,
        dump_stacks=lambda: dumps.append(1), mono=lambda: clock["now"],
    )

    with caplog.at_level(logging.WARNING, logger=drain.__name__):
        reporter.start()
        # Idempotent: uvicorn re-raises the signal after its graceful shutdown.
        reporter.start()
        clock["now"] = 2.0
        reporter.report_once()
        assert dumps == [], "a drain inside the threshold must not dump stacks"
        clock["now"] = 6.0
        reporter.report_once()
        clock["now"] = 7.0
        reporter.report_once()
        reporter.stop()

    lines = [record.getMessage() for record in caplog.records]
    assert any("still draining after 6s" in line for line in lines)
    assert any("outlived 5s" in line and "thread's stack" in line for line in lines)
    assert dumps == [1], "one dump per stop: the frames do not change while the drain is stuck"


def test_reporter_thread_reports_without_being_asked_then_stops(caplog):
    seen = threading.Event()

    class _RecordingLineHandler(logging.Handler):
        def emit(self, record):
            if "still draining" in record.getMessage():
                seen.set()

    handler = _RecordingLineHandler()
    logging.getLogger(drain.__name__).addHandler(handler)
    reporter = drain.DrainReporter(server=_FakeServer(), loop=None, interval=0.01, stack_dump_after=60.0)
    try:
        reporter.start()
        assert seen.wait(2.0), "the drain line must come from the reporter's own thread"
    finally:
        reporter.stop()
        logging.getLogger(drain.__name__).removeHandler(handler)

    assert reporter.elapsed_s() > 0


def test_report_serve_drain_chains_uvicorn_handle_exit_and_starts_reporting(caplog):
    """Installed inside ``capture_signals()``: uvicorn's handler still runs, reporting comes first."""
    server = _FakeServer(tasks=[object()])
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    reporter = None
    try:
        with caplog.at_level(logging.WARNING, logger=drain.__name__):
            reporter = drain.report_serve_drain(server, interval=0.01, stack_dump_after=60.0)
            assert reporter is not None
            os.kill(os.getpid(), signal.SIGTERM)
            deadline = time.monotonic() + 5.0
            while not server.exit_calls and time.monotonic() < deadline:
                time.sleep(0.01)
            while not any("still draining" in record.getMessage() for record in caplog.records):
                time.sleep(0.01)
                if time.monotonic() > deadline:
                    break
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if reporter is not None:
            reporter.stop()

    assert server.exit_calls == [int(signal.SIGTERM)], "the stop itself stays uvicorn's decision"
    assert any("still draining" in record.getMessage() for record in caplog.records)


def test_serve_boot_step_installs_the_reporter_from_a_running_loop():
    """``_serve`` installs it INSIDE ``capture_signals()``, where a loop is running (pre-capture
    handlers only run after the shutdown these lines describe)."""
    from hermes_cli import web_server

    server = _FakeServer()
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    reporter = None
    try:
        async def _install():
            return web_server._install_serve_drain_reporting(server)

        reporter = asyncio.run(_install())
        assert isinstance(reporter, drain.DrainReporter), "the serve path must install the reporter"
        os.kill(os.getpid(), signal.SIGTERM)
        deadline = time.monotonic() + 5.0
        while not server.exit_calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.exit_calls == [int(signal.SIGTERM)]
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if reporter is not None:
            reporter.stop()
