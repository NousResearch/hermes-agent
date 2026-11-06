"""Slow-dispatch attribution must work while real synchronous SQLite holds the loop."""

import asyncio
from contextlib import contextmanager
import logging
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def fresh_diagnostics(monkeypatch):
    from gateway import session_dispatch_diagnostics as diagnostics

    watchdog = diagnostics._DispatchWatchdog()
    monkeypatch.setattr(diagnostics, "_watchdog", watchdog)
    yield diagnostics, watchdog
    assert watchdog.close(), "diagnostic thread did not settle"
    assert watchdog._active == {}
    if watchdog._thread is not None:
        assert not watchdog._thread.is_alive()


@pytest.fixture
def connection(tmp_path):
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("private-chat", source="test")
    authority = SessionAuthority(
        SimpleNamespace(_draining=False, _cached_agent_for=lambda route: None),
        profile_id="test",
        instance_id="test",
        db=db,
        epoch=begin_runtime_epoch(db, instance_id="test"),
    )
    authority.sessions["private-chat"] = LiveSession(None, "private-route")
    viewer = AuthorityConnection(authority, object(), {"user_id": "private-human"})
    yield viewer
    db.close()


@pytest.mark.asyncio
async def test_reports_actual_dispatch_while_sqlite_still_holds_loop(
    connection, monkeypatch
):
    """The release controller is independent of the blocked event loop and diagnostic.

    SQLite's native authorizer calls a fixture Event gate during the real sessions
    SELECT. This is a held-query test, not a claim of native lock contention.
    """
    await connection.dispatch({"id": 0, "method": "session.list", "params": {}})
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    emitted, heartbeat = threading.Event(), threading.Event()
    observed, records = [], []
    loop = asyncio.get_running_loop()
    db = connection.authority.db
    original = db._read_ctx
    armed = True

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)
            emitted.set()

    handler = Capture()
    logger = logging.getLogger("gateway.session_dispatch_diagnostics")
    logger.addHandler(handler)
    prior_level = logger.level
    logger.setLevel(logging.WARNING)

    @contextmanager
    def read_ctx():
        nonlocal armed
        with original() as sql:

            def held_sqlite_read(action, table, column, database, trigger):
                nonlocal armed
                import sqlite3

                if armed and action == sqlite3.SQLITE_READ and table == "sessions":
                    armed = False
                    entered.set()
                    loop.call_soon_threadsafe(heartbeat.set)
                    release.wait(8)  # failure-only escape; never the PASS oracle
                    returned.set()
                return sqlite3.SQLITE_OK

            sql.set_authorizer(held_sqlite_read)
            try:
                yield sql
            finally:
                sql.set_authorizer(None)

    def controller():
        if entered.wait(5):
            found = emitted.wait(5)
            observed.append(
                found
                and not release.is_set()
                and not returned.is_set()
                and not heartbeat.is_set()
            )
        release.set()

    monkeypatch.setattr(db, "_read_ctx", read_ctx)
    worker = threading.Thread(target=controller, name="sqlite-test-controller")
    worker.start()
    try:
        reply = await connection.dispatch(
            {"id": 1, "method": "session.list", "params": {}}
        )
    finally:
        release.set()
        worker.join(6)
        logger.removeHandler(handler)
        logger.setLevel(prior_level)
    assert not worker.is_alive(), "fixture controller leaked"
    assert observed == [True], (
        "no diagnostic arrived while synchronous SQLite still blocked the loop"
    )
    assert "result" in reply
    assert len(records) == 1
    text = records[0].getMessage()
    assert "method=session.list" in text
    assert "session_controls.py:list_sessions:" in text
    assert "session_authority.py:resolve:" in text
    assert "session_authority.py:_handle:" in text
    assert "hermes_state_sessions.py:get_session:" in text
    assert "private-chat" not in text


@pytest.fixture
def diagnostic_log():
    records, emitted = [], threading.Event()

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)
            emitted.set()

    logger = logging.getLogger("gateway.session_dispatch_diagnostics")
    handler, previous = Capture(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    yield SimpleNamespace(records=records, emitted=emitted)
    logger.removeHandler(handler)
    logger.setLevel(previous)


@pytest.mark.asyncio
async def test_request_payload_and_sqlite_exception_never_enter_diagnostic(
    connection, fresh_diagnostics, diagnostic_log, monkeypatch
):
    import sqlite3

    secret = "SENSITIVE-" + "payload-session-prompt-token-credential"
    entered, release = threading.Event(), threading.Event()
    observed = []

    async def failing_submit(ref, params):
        private_value = params["text"]
        entered.set()
        release.wait(8)
        raise sqlite3.OperationalError(private_value)

    def controller():
        if entered.wait(5):
            observed.append(diagnostic_log.emitted.wait(5) and not release.is_set())
        release.set()

    # Warm only module/handler registration, never an inference or a submitted turn.
    await connection.dispatch({"id": 0, "method": "ping"})
    monkeypatch.setattr(connection, "submit", failing_submit)
    worker = threading.Thread(target=controller)
    worker.start()
    try:
        reply = await connection.dispatch(
            {
                "id": secret,
                "method": "prompt.submit",
                "params": {
                    "session_id": secret,
                    "text": secret,
                    "submission_id": secret,
                },
            }
        )
    finally:
        release.set()
        worker.join(6)
    assert not worker.is_alive()
    assert observed == [True]
    assert reply["id"] == secret  # RPC semantics unchanged, not part of the diagnostic.
    assert reply["error"]["data"]["reason"] == "storage_unavailable"
    assert len(diagnostic_log.records) == 1
    record = diagnostic_log.records[0]
    assert "method=prompt.submit" in record.getMessage()
    assert "failing_submit" in record.getMessage()
    assert secret not in repr(vars(record))
    assert record.exc_info is None and record.stack_info is None


def test_global_rate_and_probe_queue_are_bounded(
    fresh_diagnostics, diagnostic_log, monkeypatch
):
    import sys

    diagnostics, watchdog = fresh_diagnostics
    now, callbacks = [0.0], []
    monkeypatch.setattr(diagnostics, "time", SimpleNamespace(monotonic=lambda: now[0]))

    class Loop:
        def is_closed(self):
            return False

        def call_soon_threadsafe(self, callback):
            callbacks.append(callback)

    loop = Loop()
    record = ("session.list", id(sys._getframe()), threading.get_ident(), 0.0, loop)
    key = object()
    watchdog._active[key] = record
    watchdog._last_log = float("-inf")
    probes = {}
    watchdog._sample(dict(watchdog._active), probes)
    assert len(callbacks) == 1
    for value in (0.999, 1.0, 1.001, 2.0, 30.999):
        now[0] = value
        watchdog._sample(dict(watchdog._active), probes)
    assert len(diagnostic_log.records) == 1
    assert "dispatch_elapsed_ms=1000" in diagnostic_log.records[0].getMessage()
    # Another dispatch/method cannot evade the process-wide rate cap.
    watchdog._active.clear()
    key = object()
    watchdog._active[key] = ("cron.status", record[1], record[2], 1.0, loop)
    now[0] = 31.0
    watchdog._sample(dict(watchdog._active), probes)
    assert len(diagnostic_log.records) == 2
    assert "method=cron.status" in diagnostic_log.records[1].getMessage()
    assert len(callbacks) == 1, "a blocked loop accumulated watchdog callbacks"
    for _ in range(2000):
        watchdog._sample(dict(watchdog._active), probes)
    assert len(diagnostic_log.records) == 2


def test_stack_has_explicit_frame_and_text_bounds(
    fresh_diagnostics, diagnostic_log, monkeypatch
):
    import sys

    diagnostics, watchdog = fresh_diagnostics
    now = [0.0]
    monkeypatch.setattr(diagnostics, "time", SimpleNamespace(monotonic=lambda: now[0]))

    class Loop:
        def is_closed(self):
            return False

        def call_soon_threadsafe(self, callback):
            pass

    loop, probes, key = Loop(), {}, object()
    watchdog._active[key] = (
        "session.list",
        id(sys._getframe()),
        threading.get_ident(),
        0.0,
        loop,
    )
    watchdog._sample(dict(watchdog._active), probes)
    now[0] = 1.0

    def recurse(depth):
        if depth:
            return recurse(depth - 1)
        watchdog._sample(dict(watchdog._active), probes)

    recurse(48)
    assert len(diagnostic_log.records) == 1
    text = diagnostic_log.records[0].getMessage()
    assert len(text.split("stack=", 1)[1].split(" <- ")) <= diagnostics._MAX_FRAMES
    assert len(text) < 4400


@pytest.mark.asyncio
async def test_parked_rpc_is_not_blamed_for_another_current_handler(
    connection, fresh_diagnostics, diagnostic_log, monkeypatch
):
    started, resume = asyncio.Event(), asyncio.Event()
    entered, release = threading.Event(), threading.Event()
    observed = []

    async def parked_ping(ref, params):
        started.set()
        await resume.wait()
        return {"pong": True}

    async def current_list(ref, params):
        entered.set()
        release.wait(8)
        return {"sessions": [], "scope": "live"}

    def controller():
        if entered.wait(5):
            observed.append(diagnostic_log.emitted.wait(5) and not release.is_set())
        release.set()

    await connection.dispatch({"id": 0, "method": "ping"})
    monkeypatch.setattr(connection, "ping", parked_ping)
    monkeypatch.setattr(connection, "list_sessions", current_list)
    parked = asyncio.create_task(connection.dispatch({"id": 1, "method": "ping"}))
    await asyncio.wait_for(started.wait(), 5)
    worker = threading.Thread(target=controller)
    worker.start()
    try:
        current = await connection.dispatch({"id": 2, "method": "session.list"})
        resume.set()
        first = await asyncio.wait_for(parked, 5)
    finally:
        release.set()
        resume.set()
        await asyncio.gather(parked, return_exceptions=True)
        worker.join(6)
    assert not worker.is_alive()
    assert observed == [True]
    assert "result" in current and first["result"] == {"pong": True}
    assert len(diagnostic_log.records) == 1
    text = diagnostic_log.records[0].getMessage()
    assert "method=session.list" in text and "current_list" in text
    assert "method=ping" not in text and "parked_ping" not in text


@pytest.mark.asyncio
async def test_fast_and_unknown_requests_are_silent(
    connection, fresh_diagnostics, diagnostic_log
):
    for index in range(40):
        assert (await connection.dispatch({"id": index, "method": "ping"}))[
            "result"
        ] == {"pong": True}
    secret = "UNREGISTERED-" + "private-method"
    reply = await connection.dispatch({"id": secret, "method": secret})
    assert reply["error"]["code"] == -32601
    assert diagnostic_log.records == []
    assert fresh_diagnostics[1]._active == {}


def test_one_pending_probe_survives_idle_gaps(fresh_diagnostics, monkeypatch):
    import sys

    diagnostics, watchdog = fresh_diagnostics
    monkeypatch.setattr(diagnostics, "time", SimpleNamespace(monotonic=lambda: 0.0))
    callbacks = []

    class Loop:
        def is_closed(self):
            return False

        def call_soon_threadsafe(self, callback):
            callbacks.append(callback)

    loop, probes = Loop(), {}
    key = object()
    active = {key: ("ping", id(sys._getframe()), threading.get_ident(), 0.0, loop)}
    for _ in range(100):
        watchdog._sample(active, probes)
        watchdog._sample({}, probes)
    assert len(callbacks) == 1, "idle gaps discarded an outstanding loop probe"


def test_wakeups_cannot_bypass_poll_budget(fresh_diagnostics, monkeypatch):
    import sys
    import time

    diagnostics, watchdog = fresh_diagnostics
    sampled, done = threading.Event(), threading.Event()
    moments = []

    class Loop:
        pass

    loop = Loop()

    def sample(active, probes):
        moments.append(time.monotonic())
        sampled.set()
        if len(moments) >= 2:
            done.set()

    monkeypatch.setattr(watchdog, "_sample", sample)
    key = watchdog.begin("ping", sys._getframe(), loop)
    try:
        assert sampled.wait(5)
        for _ in range(400):
            other = watchdog.begin("ping", sys._getframe(), loop)
            watchdog.end(other)
        assert done.wait(5)
        assert moments[1] - moments[0] >= diagnostics._POLL_SECONDS, (
            "request notifications caused oversampling"
        )
    finally:
        watchdog.end(key)


@pytest.mark.asyncio
async def test_logging_failure_does_not_recurse_or_change_rpc(
    connection, fresh_diagnostics, monkeypatch
):
    diagnostics, watchdog = fresh_diagnostics
    entered, release, attempted = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    observations, writes = [], []

    class Broken(logging.Handler):
        def emit(self, record):
            writes.append(record.getMessage())
            attempted.set()
            raise OSError("synthetic sink failure")

    handler = Broken()
    logger = diagnostics.logger
    previous = logger.level
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)

    async def held_ping(ref, params):
        entered.set()
        release.wait(8)
        return {"pong": True}

    def controller():
        if entered.wait(5):
            observations.append(attempted.wait(5) and not release.is_set())
        release.set()

    await connection.dispatch({"id": 0, "method": "ping"})
    monkeypatch.setattr(connection, "ping", held_ping)
    worker = threading.Thread(target=controller)
    worker.start()
    try:
        reply = await connection.dispatch({"id": 1, "method": "ping"})
    finally:
        release.set()
        worker.join(6)
        logger.removeHandler(handler)
        logger.setLevel(previous)
    assert not worker.is_alive()
    assert observations == [True]
    assert reply["result"] == {"pong": True}
    assert watchdog.close()
    assert len(writes) == 1 and watchdog._closed
    assert watchdog._active == {}


@pytest.mark.asyncio
async def test_cancellation_removes_dispatch_without_stopping_its_owner(
    connection, fresh_diagnostics, monkeypatch
):
    started, resume = asyncio.Event(), asyncio.Event()

    async def parked_ping(ref, params):
        started.set()
        await resume.wait()
        return {"pong": True}

    monkeypatch.setattr(connection, "ping", parked_ping)
    task = asyncio.create_task(connection.dispatch({"id": 1, "method": "ping"}))
    try:
        await asyncio.wait_for(started.wait(), 5)
        assert len(fresh_diagnostics[1]._active) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fresh_diagnostics[1]._active == {}
    finally:
        resume.set()
        await asyncio.gather(task, return_exceptions=True)


def test_registration_has_fixed_capacity_and_one_lazy_thread(fresh_diagnostics):
    import sys

    diagnostics, watchdog = fresh_diagnostics
    assert watchdog._thread is None
    loop = asyncio.new_event_loop()
    keys = []
    try:
        for _ in range(diagnostics._MAX_ACTIVE):
            keys.append(watchdog.begin("ping", sys._getframe(), loop))
        thread = watchdog._thread
        assert all(key is not None for key in keys)
        assert len(watchdog._active) == diagnostics._MAX_ACTIVE
        assert watchdog.begin("ping", sys._getframe(), loop) is None
        assert watchdog._thread is thread
    finally:
        for key in keys:
            watchdog.end(key)
        assert watchdog.close()
        loop.close()
    assert watchdog.begin("ping", sys._getframe(), loop) is None


def test_dynamic_external_code_metadata_is_not_echoed(fresh_diagnostics):
    diagnostics, _ = fresh_diagnostics
    marker = "PRIVATE-" + "dynamic-name"

    def placeholder():
        pass

    code = placeholder.__code__.replace(
        co_filename="/outside/" + marker, co_name=marker
    )
    frame = SimpleNamespace(f_code=code, f_lineno=9)
    assert marker not in diagnostics._location(frame)
    code = placeholder.__code__.replace(
        co_filename=str(diagnostics._SOURCE_ROOT / ("unsafe name-" + marker + ".py")),
        co_name=marker,
    )
    frame = SimpleNamespace(f_code=code, f_lineno=9)
    assert marker not in diagnostics._location(frame)


@pytest.mark.asyncio
async def test_disabled_logging_adds_no_thread(connection, fresh_diagnostics):
    diagnostics, watchdog = fresh_diagnostics
    previous = diagnostics.logger.level
    diagnostics.logger.setLevel(logging.ERROR)
    try:
        reply = await connection.dispatch({"id": 1, "method": "ping"})
    finally:
        diagnostics.logger.setLevel(previous)
    assert reply["result"] == {"pong": True}
    assert watchdog._thread is None and watchdog._active == {}


def test_closed_loop_drops_its_probe_without_scheduling(fresh_diagnostics, monkeypatch):
    import sys

    diagnostics, watchdog = fresh_diagnostics
    monkeypatch.setattr(diagnostics, "time", SimpleNamespace(monotonic=lambda: 0.0))
    calls = []

    class Loop:
        def is_closed(self):
            return True

        def call_soon_threadsafe(self, callback):
            calls.append(callback)

    loop, probes = Loop(), {}
    key = object()
    active = {key: ("ping", id(sys._getframe()), threading.get_ident(), 0.0, loop)}
    watchdog._sample(active, probes)
    watchdog._sample({}, probes)
    assert calls == [] and probes == {}


def test_probe_memory_has_fixed_capacity(fresh_diagnostics, monkeypatch):
    import sys

    diagnostics, watchdog = fresh_diagnostics
    monkeypatch.setattr(diagnostics, "time", SimpleNamespace(monotonic=lambda: 0.0))

    class Loop:
        def is_closed(self):
            return False

        def call_soon_threadsafe(self, callback):
            pass

    probes, loops = {}, [Loop() for _ in range(diagnostics._MAX_ACTIVE + 40)]
    for loop in loops:
        active = {
            object(): ("ping", id(sys._getframe()), threading.get_ident(), 0.0, loop)
        }
        watchdog._sample(active, probes)
    assert len(probes) <= diagnostics._MAX_ACTIVE
