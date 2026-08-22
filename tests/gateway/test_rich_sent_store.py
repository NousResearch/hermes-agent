"""Behavioral tests for the best-effort rich-message text index."""

import json
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gateway import rich_sent_store
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def store_path(tmp_path, monkeypatch):
    path = tmp_path / "state" / "rich_sent_index.json"
    monkeypatch.setattr(rich_sent_store, "_store_path", lambda: str(path))
    return path


def _record_batch_in_subprocess(path, worker_id, start, count):
    from gateway import rich_sent_store as process_store

    process_store._store_path = lambda: path
    if not start.wait(timeout=10):
        raise RuntimeError("parent did not release concurrent writers")
    for index in range(count):
        process_store.record(
            "chat", f"{worker_id}-{index}", f"text-{worker_id}-{index}"
        )


def _record_once_in_subprocess(path):
    from gateway import rich_sent_store as process_store

    process_store._store_path = lambda: path
    process_store.record("chat", "child", "from-child")


def test_concurrent_distinct_records_preserve_both_updates(tmp_path, monkeypatch):
    store_path = tmp_path / "state" / "rich_sent_index.json"
    store_path.parent.mkdir()
    store_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rich_sent_store, "_store_path", lambda: str(store_path))

    both_read_initial_state = threading.Event()
    reads_lock = threading.Lock()
    read_count = 0
    original_load = json.load

    def synchronized_load(handle):
        nonlocal read_count
        data = original_load(handle)
        with reads_lock:
            read_count += 1
            if read_count == 2:
                both_read_initial_state.set()
        # Unlocked writers both arrive and are released together. A serialized
        # writer times out here, publishes, then lets the next writer read it.
        both_read_initial_state.wait(timeout=1)
        return data

    monkeypatch.setattr(rich_sent_store.json, "load", synchronized_load)
    # Isolate the lost update from the old PID-only temp-name collision.
    monkeypatch.setattr(rich_sent_store.os, "getpid", threading.get_ident)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(rich_sent_store.record, "chat", "one", "first"),
            pool.submit(rich_sent_store.record, "chat", "two", "second"),
        ]
        for future in futures:
            future.result(timeout=5)

    with store_path.open(encoding="utf-8") as handle:
        data = original_load(handle)
    assert set(data) == {"chat:one", "chat:two"}


@pytest.mark.skipif(
    os.name not in {"nt", "posix"},
    reason="rich sent store supports Windows and POSIX file locking",
)
def test_separate_process_writers_preserve_all_updates(tmp_path):
    path = tmp_path / "state" / "rich_sent_index.json"
    path.parent.mkdir()
    path.write_text("{}", encoding="utf-8")

    context = multiprocessing.get_context("spawn")
    start = context.Event()
    process_count = 3
    records_per_process = 12
    processes = [
        context.Process(
            target=_record_batch_in_subprocess,
            args=(str(path), worker_id, start, records_per_process),
        )
        for worker_id in range(process_count)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=20)

    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(timeout=5)
    assert not alive, "concurrent record processes deadlocked"
    assert [process.exitcode for process in processes] == [0] * process_count

    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == process_count * records_per_process
    for worker_id in range(process_count):
        for index in range(records_per_process):
            key = f"chat:{worker_id}-{index}"
            assert data[key]["t"] == f"text-{worker_id}-{index}"


@pytest.mark.linux_only
def test_forked_writer_does_not_inherit_held_path_mutex(tmp_path):
    path = tmp_path / "state" / "rich_sent_index.json"
    path.parent.mkdir()
    path.write_text("{}", encoding="utf-8")

    context = multiprocessing.get_context("fork")
    process = context.Process(target=_record_once_in_subprocess, args=(str(path),))
    path_mutex = rich_sent_store._thread_lock(str(path))
    path_mutex.acquire()
    try:
        process.start()
        process.join(timeout=5)
        alive = process.is_alive()
        if alive:
            process.terminate()
            process.join(timeout=5)
    finally:
        path_mutex.release()

    assert not alive, "forked writer inherited the parent's locked path mutex"
    assert process.exitcode == 0
    assert (
        json.loads(path.read_text(encoding="utf-8"))["chat:child"]["t"] == "from-child"
    )


def test_sequential_records_remain_lookupable(store_path):
    rich_sent_store.record("chat", "one", "first")
    rich_sent_store.record("chat", "two", "second")

    assert rich_sent_store.lookup("chat", "one") == "first"
    assert rich_sent_store.lookup("chat", "two") == "second"
    assert rich_sent_store.lookup("chat", "missing") is None


def test_same_key_record_overwrites_text(store_path):
    rich_sent_store.record("chat", "message", "before")
    rich_sent_store.record("chat", "message", "after")

    assert rich_sent_store.lookup("chat", "message") == "after"
    assert len(json.loads(store_path.read_text(encoding="utf-8"))) == 1


def test_malformed_file_is_recovered_on_record(store_path):
    store_path.parent.mkdir()
    store_path.write_text("{not valid json", encoding="utf-8")

    rich_sent_store.record("chat", "message", "recovered")

    assert rich_sent_store.lookup("chat", "message") == "recovered"


@pytest.mark.parametrize(
    "malformed_entry",
    [None, {"t": "bad", "ts": "invalid"}],
    ids=("non-dict", "non-numeric-timestamp"),
)
def test_retention_recovers_parseable_malformed_entry(
    store_path, monkeypatch, malformed_entry
):
    store_path.parent.mkdir()
    store_path.write_text(
        json.dumps({
            "chat:valid": {"t": "valid", "ts": 2},
            "chat:bad": malformed_entry,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(rich_sent_store, "_MAX_ENTRIES", 2)
    monkeypatch.setattr(rich_sent_store.time, "time", lambda: 3)

    rich_sent_store.record("chat", "new", "new")

    assert rich_sent_store.lookup("chat", "valid") == "valid"
    assert rich_sent_store.lookup("chat", "new") == "new"
    assert set(json.loads(store_path.read_text(encoding="utf-8"))) == {
        "chat:valid",
        "chat:new",
    }


def test_retention_keeps_newest_records_lookupable(store_path, monkeypatch):
    timestamps = iter((1, 2, 3))
    monkeypatch.setattr(rich_sent_store, "_MAX_ENTRIES", 2)
    monkeypatch.setattr(rich_sent_store.time, "time", lambda: next(timestamps))

    rich_sent_store.record("chat", "old", "old")
    rich_sent_store.record("chat", "middle", "middle")
    rich_sent_store.record("chat", "new", "new")

    assert rich_sent_store.lookup("chat", "old") is None
    assert rich_sent_store.lookup("chat", "middle") == "middle"
    assert rich_sent_store.lookup("chat", "new") == "new"


def test_profile_indexes_remain_isolated(tmp_path):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"

    token = set_hermes_home_override(profile_a)
    try:
        rich_sent_store.record("chat", "message", "from-a")
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(profile_b)
    try:
        rich_sent_store.record("chat", "message", "from-b")
        assert rich_sent_store.lookup("chat", "message") == "from-b"
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(profile_a)
    try:
        assert rich_sent_store.lookup("chat", "message") == "from-a"
    finally:
        reset_hermes_home_override(token)


def test_failed_replace_cleans_unique_temp_and_releases_lock(store_path, monkeypatch):
    real_replace = os.replace
    temp_paths: list[Path] = []

    def fail_first_replace(source, destination):
        temp_paths.append(Path(source))
        if len(temp_paths) == 1:
            raise OSError("injected replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(rich_sent_store.os, "replace", fail_first_replace)

    rich_sent_store.record("chat", "failed", "not published")
    assert len(temp_paths) == 1
    assert not temp_paths[0].exists()
    assert rich_sent_store.lookup("chat", "failed") is None

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rich_sent_store.record, "chat", "succeeded", "published")
        future.result(timeout=5)

    assert len(temp_paths) == 2
    assert temp_paths[0] != temp_paths[1]
    assert not temp_paths[1].exists()
    assert rich_sent_store.lookup("chat", "succeeded") == "published"
