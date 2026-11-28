"""A host record whose pid is now another process's thread id is stale, not a live owner."""

import dataclasses
import threading

import pytest

from gateway import host_rendezvous as hr


@pytest.fixture
def host_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    yield tmp_path
    hr.release_host_lock(hr.ROLE_GATEWAY)


@pytest.mark.platforms("linux")
def test_gateway_record_naming_a_thread_id_is_stale(host_dir):
    """Same boot epoch, start ticks that match: only the tid/tgid split tells them apart.

    This is what a recycled owner pid looks like when the kernel hands the number to a thread
    of some unrelated process: psutil.Process() and /proc/<tid>/stat both answer for it.
    """
    from gateway.status import get_process_start_time

    record = hr.publish_record(hr.ROLE_GATEWAY, profiles=("default",), home=str(host_dir))
    assert record is not None
    assert hr.record_is_stale(record) is False  # control: the real owner is live

    ready, release = threading.Event(), threading.Event()
    tid = []

    def run():
        tid.append(threading.get_native_id())
        ready.set()
        release.wait(10)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert ready.wait(10)
    try:
        start = get_process_start_time(tid[0])
        assert start is not None  # /proc/<tid>/stat resolves, so the fingerprint alone can't help
        as_thread = dataclasses.replace(record, pid=tid[0], start_time=start)

        assert hr.record_is_stale(as_thread) is True
        assert hr.liveness_is_proven(as_thread) is False
        assert hr.record_names_live_owner(as_thread) is False
    finally:
        release.set()
        worker.join(10)
