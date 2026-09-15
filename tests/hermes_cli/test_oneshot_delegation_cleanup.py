"""One-shot async-delegation shutdown and crash-recovery invariants."""

import json
import os
import subprocess
import sys
import threading
import time

import pytest

from hermes_cli import main as cli_main
from tools import async_delegation as ad
from tools.process_registry import process_registry


_ONESHOT_UNKNOWN_REASON = (
    "One-shot shutdown grace period elapsed before the delegation recorded a "
    "terminal result; outcome unknown."
)


@pytest.fixture(autouse=True)
def _isolated_delegations(monkeypatch, tmp_path):
    real_sleep = time.sleep
    monotonic = 0.0

    def controlled_monotonic():
        return monotonic

    def controlled_sleep(seconds):
        nonlocal monotonic
        monotonic += seconds
        real_sleep(0.001)

    monkeypatch.setattr(time, "monotonic", controlled_monotonic)
    monkeypatch.setattr(time, "sleep", controlled_sleep)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    monkeypatch.setattr(cli_main, "_oneshot_cleanup_done", False)
    module, attr, kwargs, swallow = next(
        step for step in cli_main._ONESHOT_CLEANUPS if step[0] == "tools.async_delegation"
    )
    if attr == "finalize_for_oneshot_shutdown":
        finalizer = ad.finalize_for_oneshot_shutdown
        monkeypatch.setattr(ad, attr, lambda: finalizer(grace_seconds=0.1))
    async_cleanup = (module, attr, kwargs, swallow)
    monkeypatch.setattr(cli_main, "_ONESHOT_CLEANUPS", (async_cleanup,))
    yield
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _dispatch(*, goal, runner, interrupt_fn):
    started = time.monotonic()
    handle = ad.dispatch_async_delegation(
        goal=goal,
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="",
        runner=runner,
        interrupt_fn=interrupt_fn,
    )
    assert handle["status"] == "dispatched"
    assert time.monotonic() - started < 0.5
    return handle["delegation_id"]


def test_oneshot_cleanup_terminalizes_prompt_child_and_grace_survivor():
    """One-shot cleanup waits only for unwind, then closes every durable row."""
    prompt_interrupted = threading.Event()
    survivor_interrupted = threading.Event()
    release_survivor = threading.Event()

    def prompt_runner():
        assert prompt_interrupted.wait(timeout=10)
        return {"status": "interrupted", "summary": None, "error": "cancelled"}

    def survivor_runner():
        release_survivor.wait(timeout=10)
        return {"status": "completed", "summary": "late result"}

    def survivor_interrupt():
        survivor_interrupted.set()
        time.sleep(0.08)

    prompt_id = _dispatch(
        goal="prompt finalizer",
        runner=prompt_runner,
        interrupt_fn=prompt_interrupted.set,
    )
    survivor_id = _dispatch(
        goal="grace survivor",
        runner=survivor_runner,
        interrupt_fn=survivor_interrupt,
    )

    try:
        started = time.monotonic()
        cli_main._cleanup_oneshot_runtime()
        elapsed = time.monotonic() - started

        assert prompt_interrupted.is_set() and survivor_interrupted.is_set()
        assert elapsed == pytest.approx(0.1, abs=0.001)
        prompt_row = ad.get_durable_delegation(prompt_id)
        survivor_row = ad.get_durable_delegation(survivor_id)
        assert prompt_row["state"] == "interrupted"
        assert survivor_row["state"] == "unknown"
        assert survivor_row["result"] == {
            "status": "unknown",
            "summary": None,
            "error": _ONESHOT_UNKNOWN_REASON,
        }

        events = []
        while not process_registry.completion_queue.empty():
            events.append(process_registry.completion_queue.get_nowait())
        assert {event["delegation_id"]: event["status"] for event in events} == {
            prompt_id: "interrupted",
            survivor_id: "unknown",
        }
    finally:
        release_survivor.set()

    deadline = time.monotonic() + 2
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ad.get_durable_delegation(survivor_id)["state"] == "unknown"
    assert process_registry.completion_queue.empty()


def test_oneshot_cleanup_does_not_renew_grace_after_signal_overrun():
    """A synchronous signal overrun skips the wait instead of restarting it."""
    release = threading.Event()

    def runner():
        release.wait(timeout=10)
        return {"status": "completed", "summary": "late result"}

    def blocking_interrupt():
        time.sleep(0.12)

    delegation_id = _dispatch(goal="blocking signal", runner=runner, interrupt_fn=blocking_interrupt)
    try:
        started = time.monotonic()
        cli_main._cleanup_oneshot_runtime()
        assert time.monotonic() - started == pytest.approx(0.12, abs=0.001)
        assert ad.get_durable_delegation(delegation_id)["state"] == "unknown"
    finally:
        release.set()


def test_first_durable_read_preserves_live_owner_when_start_identity_unavailable(monkeypatch):
    """Unavailable start identity cannot turn a live owner's row terminal."""
    from gateway import status as gateway_status

    release = threading.Event()

    def live_runner():
        release.wait(timeout=10)
        return {"status": "completed", "summary": "done"}

    delegation_id = _dispatch(
        goal="live owner",
        runner=live_runner,
        interrupt_fn=lambda: None,
    )
    actual_started = gateway_status.get_process_start_time(os.getpid())
    assert actual_started is not None

    try:
        with ad._DB_LOCK, ad._transaction() as conn:
            conn.execute(
                "UPDATE async_delegations SET owner_started_at=NULL WHERE delegation_id=?",
                (delegation_id,),
            )
        assert ad.get_durable_delegation(delegation_id)["state"] == "running"

        with ad._DB_LOCK, ad._transaction() as conn:
            conn.execute(
                "UPDATE async_delegations SET owner_started_at=? WHERE delegation_id=?",
                (actual_started, delegation_id),
            )
        monkeypatch.setattr(gateway_status, "get_process_start_time", lambda _pid: None)
        assert ad.get_durable_delegation(delegation_id)["state"] == "running"
    finally:
        release.set()

    deadline = time.monotonic() + 2
    row = ad.get_durable_delegation(delegation_id)
    while row["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.02)
        row = ad.get_durable_delegation(delegation_id)
    assert row["state"] == "completed"
    assert row["result"]["summary"] == "done"


def test_first_durable_read_recovers_reused_owner_pid_as_unknown(tmp_path):
    """A live reused PID cannot keep a crashed owner's durable row running."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env = {**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": repo}
    producer = r'''
import os
import time
from tools import async_delegation as ad

handle = ad.dispatch_async_delegation(
    goal="crash before finalization", context=None, toolsets=None, role="leaf",
    model="m", session_key="", runner=lambda: time.sleep(600),
)
print(handle["delegation_id"], flush=True)
os._exit(17)
'''
    first = subprocess.run(
        [sys.executable, "-c", producer],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert first.returncode == 17
    delegation_id = first.stdout.strip().splitlines()[-1]

    consumer = r'''
import json
import logging
import os
import sqlite3
from pathlib import Path

from tools import async_delegation as ad

db = Path(os.environ["HERMES_HOME"]) / "state.db"
with sqlite3.connect(db) as conn:
    conn.execute(
        "UPDATE async_delegations SET owner_pid=?, owner_started_at=? WHERE delegation_id=?",
        (os.getpid(), 1, os.environ["DELEGATION_ID"]),
    )

records = []
class Capture(logging.Handler):
    def emit(self, record):
        if hasattr(record, "async_delegations_recovered_total"):
            records.append(record.async_delegations_recovered_total)

ad.logger.setLevel(logging.INFO)
ad.logger.addHandler(Capture())
row = ad.get_durable_delegation(os.environ["DELEGATION_ID"])
print(json.dumps({"row": row, "recovered": records}))
'''
    second = subprocess.run(
        [sys.executable, "-c", consumer],
        cwd=repo,
        env={**env, "DELEGATION_ID": delegation_id},
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    observed = json.loads(second.stdout.strip().splitlines()[-1])
    assert observed["row"]["state"] == "unknown"
    assert observed["row"]["result"]["status"] == "unknown"
    assert observed["recovered"] == [1]
