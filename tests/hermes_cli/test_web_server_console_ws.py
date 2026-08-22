"""Dashboard Hermes Console websocket tests."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from urllib.parse import urlencode

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli import web_server


@pytest.fixture
def console_client(monkeypatch, _isolate_hermes_home):
    previous_auth_required = getattr(web_server.app.state, "auth_required", None)
    previous_bound_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None
    monkeypatch.setattr(web_server, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

    client = TestClient(web_server.app)
    try:
        yield client
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
        if previous_auth_required is None:
            if hasattr(web_server.app.state, "auth_required"):
                delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous_auth_required
        if previous_bound_host is None:
            if hasattr(web_server.app.state, "bound_host"):
                delattr(web_server.app.state, "bound_host")
        else:
            web_server.app.state.bound_host = previous_bound_host


def _url(token: str | None = None, **params: str) -> str:
    query = {"token": web_server._SESSION_TOKEN, **params}
    if token is not None:
        query["token"] = token
    return f"/api/console?{urlencode(query)}"


def _recv_until(conn, frame_type: str, *, status: str | None = None) -> dict:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        frame = conn.receive_json()
        if frame.get("type") != frame_type:
            continue
        if status is not None and frame.get("status") != status:
            continue
        return frame
    raise AssertionError(f"Timed out waiting for {frame_type} frame")


def test_console_ws_rejects_missing_or_bad_token(console_client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with console_client.websocket_connect("/api/console"):
            pass
    assert exc.value.code == 4401

    with pytest.raises(WebSocketDisconnect) as exc:
        with console_client.websocket_connect(_url(token="wrong")):
            pass
    assert exc.value.code == 4401


def test_console_ws_cancel_returns_to_prompt(console_client, monkeypatch):
    from hermes_cli.console_engine import ConsoleResult, HermesConsoleEngine

    def slow_execute(self, line: str, *, confirmed: bool = False):
        time.sleep(0.2)
        return ConsoleResult("ok", output="late", command=line)

    monkeypatch.setattr(HermesConsoleEngine, "execute", slow_execute)

    with console_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "ready"
        conn.send_json({"type": "input", "line": "status"})
        conn.send_json({"type": "cancel"})

        complete = _recv_until(conn, "complete", status="cancelled")
        assert complete["prompt"] == "hermes> "


# ---------------------------------------------------------------------------
# Console worker pool: timeout bookkeeping + recycling.
#
# A console command that outlives the timeout keeps its (unkillable) worker
# thread forever, so the bounded pool must notice when every worker is wedged
# and replace itself. These drive the pool helpers directly: the websocket path
# would need a real 60s timeout, which is neither fast nor deterministic.
# ---------------------------------------------------------------------------

_MAX_WORKERS = web_server._CONSOLE_EXECUTOR_MAX_WORKERS


@pytest.fixture
def console_pool():
    """Give each test a pristine console pool and tear its threads down."""
    saved = (
        web_server._console_executor,
        web_server._console_executor_generation,
        dict(web_server._console_stuck_futures),
    )
    web_server._console_executor = None
    web_server._console_executor_generation = 0
    web_server._console_stuck_futures.clear()
    created = []
    try:
        yield created
    finally:
        for executor in {id(e): e for e in created}.values():
            executor.shutdown(wait=False, cancel_futures=True)
        current = web_server._console_executor
        if current is not None and current is not saved[0]:
            current.shutdown(wait=False, cancel_futures=True)
        (
            web_server._console_executor,
            web_server._console_executor_generation,
            _stuck,
        ) = saved
        web_server._console_stuck_futures.clear()
        web_server._console_stuck_futures.update(_stuck)


class _Wedge:
    """A console command that blocks until the test explicitly releases it."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self) -> str:
        self.started.set()
        assert self.release.wait(30), "wedged console command was never released"
        return "late"


def _submit_wedged(created):
    """Submit a blocking command and wait until its worker really holds a slot."""
    wedge = _Wedge()
    future, generation = web_server._submit_console_command(wedge)
    created.append(web_server._console_executor)
    assert wedge.started.wait(10), "console worker never started"
    return wedge, future, generation


def _run_to_completed_callback(created, value="ok"):
    """Run a normal command and wait until its done-callbacks have all fired.

    ``Future.result()`` can return before the callbacks run, so register a probe
    afterwards: callbacks fire in registration order, so once the probe has run
    the production callback has already finished.
    """
    future, generation = web_server._submit_console_command(lambda: value)
    created.append(web_server._console_executor)
    callbacks_done = threading.Event()
    future.add_done_callback(lambda _f: callbacks_done.set())
    assert future.result(timeout=10) == value
    assert callbacks_done.wait(10), "console done-callback never ran"
    return future, generation


def test_console_pool_recycles_only_when_every_worker_is_stuck(console_pool):
    """Four wedged workers replace the pool; a normal command never counts."""
    original = None
    wedges = []
    try:
        first_wedge, first_future, generation = _submit_wedged(console_pool)
        wedges.append(first_wedge)
        original = web_server._console_executor
        assert original is not None
        assert generation == 0

        web_server._note_console_command_stuck(first_future, generation)
        assert web_server._console_stuck_futures[0] == {first_future}

        # Ordinary commands complete on the three free workers. None of them
        # timed out, so none may clear the wedged worker's mark -- if they did,
        # the pool would never reach the recycle threshold below.
        for _ in range(3):
            _run_to_completed_callback(console_pool)
            assert web_server._console_stuck_futures[0] == {first_future}
            assert web_server._console_executor is original

        # Wedge the remaining workers. The pool is only replaced on the last one.
        for expected in range(2, _MAX_WORKERS + 1):
            wedge, future, gen = _submit_wedged(console_pool)
            wedges.append(wedge)
            assert gen == 0
            web_server._note_console_command_stuck(future, gen)
            if expected < _MAX_WORKERS:
                assert web_server._console_executor is original
                assert len(web_server._console_stuck_futures[0]) == expected

        assert web_server._console_executor is not original
        assert web_server._console_executor_generation == 1
        assert web_server._console_stuck_futures == {1: set()}

        # The console works again: new commands land on the replacement pool.
        healthy, healthy_gen = web_server._submit_console_command(lambda: "alive")
        assert healthy_gen == 1
        assert healthy.result(timeout=10) == "alive"
    finally:
        for wedge in wedges:
            wedge.release.set()


def test_completed_console_command_is_never_marked_stuck(console_pool):
    """A command that returns as we give up on the await is slow, not wedged."""
    future, generation = _run_to_completed_callback(console_pool, value="fast")

    web_server._note_console_command_stuck(future, generation)

    assert web_server._console_stuck_futures[generation] == set()
    assert web_server._console_executor_generation == 0


def test_late_completion_from_retired_pool_leaves_replacement_untouched(console_pool):
    """A retired generation's futures must not touch the replacement's state."""
    wedges = []
    retired_futures = []
    try:
        for _ in range(_MAX_WORKERS):
            wedge, future, generation = _submit_wedged(console_pool)
            wedges.append(wedge)
            retired_futures.append(future)
            assert generation == 0
            web_server._note_console_command_stuck(future, generation)

        replacement = web_server._console_executor
        assert web_server._console_executor_generation == 1

        # Wedge one worker on the replacement pool so it has state to corrupt.
        new_wedge, new_future, new_generation = _submit_wedged(console_pool)
        wedges.append(new_wedge)
        assert new_generation == 1
        web_server._note_console_command_stuck(new_future, new_generation)
        assert web_server._console_stuck_futures == {1: {new_future}}

        # A timeout reported against the retired generation -- whose workers are
        # still wedged, so the "already finished" shortcut cannot absorb it --
        # must be dropped on generation ownership alone. Otherwise these four
        # would re-trip the threshold and retire the *live* pool.
        for future in retired_futures:
            assert not future.done()
            web_server._note_console_command_stuck(future, 0)
        assert web_server._console_stuck_futures == {1: {new_future}}
        assert web_server._console_executor is replacement
        assert web_server._console_executor_generation == 1

        # Now let the retired pool's wedged workers finally return.
        probes = []
        for future in retired_futures:
            done = threading.Event()
            future.add_done_callback(lambda _f, ev=done: ev.set())
            probes.append(done)
        for wedge in wedges[:_MAX_WORKERS]:
            wedge.release.set()
        for future, done in zip(retired_futures, probes):
            assert future.result(timeout=10) == "late"
            assert done.wait(10), "retired console done-callback never ran"

        # Generation 1's bookkeeping is exactly as it was.
        assert web_server._console_stuck_futures == {1: {new_future}}
        assert web_server._console_executor is replacement
        assert web_server._console_executor_generation == 1

        # A late timeout report from the retired generation is ignored too, so
        # it can neither re-arm nor re-trip the replacement's threshold.
        web_server._note_console_command_stuck(retired_futures[0], 0)
        assert web_server._console_stuck_futures == {1: {new_future}}
        assert web_server._console_executor is replacement

        # The consequence that actually matters: the replacement still counts
        # its own wedged worker, so it recycles on its own fourth one -- not on
        # a fifth, which is what losing that mark to the old pool would cost.
        for expected in range(2, _MAX_WORKERS + 1):
            wedge, future, gen = _submit_wedged(console_pool)
            wedges.append(wedge)
            assert gen == 1
            web_server._note_console_command_stuck(future, gen)
            if expected < _MAX_WORKERS:
                assert web_server._console_executor is replacement
                assert len(web_server._console_stuck_futures[1]) == expected
        assert web_server._console_executor_generation == 2
        assert web_server._console_executor is not replacement
    finally:
        for wedge in wedges:
            wedge.release.set()


def test_console_executor_atexit_hook_shuts_down_its_own_pool(console_pool, monkeypatch):
    """Each pool's exit hook targets that pool, not whatever the global holds."""
    hooks = []
    monkeypatch.setattr(
        web_server.atexit, "register", lambda fn: (hooks.append(fn), fn)[1]
    )

    first = web_server._new_console_executor()
    console_pool.append(first)
    second = web_server._new_console_executor()
    console_pool.append(second)
    assert len(hooks) == 2
    assert isinstance(first, concurrent.futures.ThreadPoolExecutor)

    # Point the module global at the second pool, the way a recycle does.
    web_server._console_executor = second

    # The first pool's hook must tear down the first pool even though the
    # global now names the replacement.
    hooks[0]()
    with pytest.raises(RuntimeError):
        first.submit(lambda: "nope")
    assert second.submit(lambda: "alive").result(timeout=10) == "alive"
