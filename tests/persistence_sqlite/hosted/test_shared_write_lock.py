"""The per-file lock a ``shared-state.db`` write takes in this process: a write waits for it at most the hosted-room
patience, then refuses with the contract's ``StoreContention``."""

from __future__ import annotations

import threading
import time

import pytest

from persistence import StoreContention
from persistence.sqlite.gateway import _hosted_store
from persistence.sqlite.gateway._hosted_store import HostedFiles
from tests.persistence_sqlite.hosted._task_fixtures import Hosted, make_hosted

PATIENCE_S = 0.2
# How long the case waits for the blocked write before calling it unbounded.
UNBOUNDED_S = 5.0


@pytest.fixture
def hosted() -> Hosted:
    hosted = make_hosted()
    yield hosted
    hosted.backend.close()


def test_a_shared_write_refuses_once_the_file_lock_stays_held_past_its_patience(hosted, monkeypatch):
    monkeypatch.setattr(_hosted_store, "HOSTED_PATIENCE_S", PATIENCE_S)
    files = HostedFiles(hosted.store)
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with _hosted_store._file_lock(files.path("shared")):
            held.set()
            release.wait()

    outcome: dict = {}

    def write() -> None:
        start = time.monotonic()
        try:
            files._hosted_write("shared", lambda conn: "written", immediate=True)
        except Exception as exc:  # health: allow BLE001 -- the case reads whatever the write raised
            outcome["error"] = exc
        outcome["elapsed"] = time.monotonic() - start

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait()
    writer = threading.Thread(target=write)
    writer.start()
    writer.join(UNBOUNDED_S)
    release.set()
    holder.join()
    writer.join()
    # CONTRACT: the write gives up after its patience with a retryable contention refusal, the lock still held.
    assert isinstance(outcome.get("error"), StoreContention), outcome
    assert PATIENCE_S <= outcome["elapsed"] < UNBOUNDED_S
