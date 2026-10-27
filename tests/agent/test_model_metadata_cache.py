"""Concurrent writers of ``context_length_cache.yaml`` (race audit lostupdate-7).

Every agent (CLI, gateway sessions, cron jobs, subagents) persists probed context windows with a
load -> mutate -> write cycle; an unserialized pair rewrites a stale snapshot and drops the other
writer's entry, so the next session re-probes or overflows its context on the first call.
"""

import threading

import pytest

import agent.model_metadata as mm
from tests.lock_contention import watch_file_lock_contention
from tests.waiting import wait_until


@pytest.mark.parametrize("first_writer", ["save", "invalidate"])
def test_concurrent_cache_writers_keep_each_others_entries(monkeypatch, first_writer):
    contended = watch_file_lock_contention(monkeypatch)
    mm.save_context_length("model-a", "https://a.example/v1", 200_000)
    real_load = mm._load_context_cache_document
    loaded = threading.Event()
    other_saved = threading.Event()

    def parked_load():
        document = real_load()
        if threading.current_thread().name == "first-writer":
            loaded.set()
            # Unserialized writers let the other agent save here; serialized ones are seen
            # blocking on the cache lock instead.
            wait_until(lambda: other_saved.is_set() or contended.is_set(),
                       what="the other agent saved or blocked on the cache lock")
        return document

    monkeypatch.setattr(mm, "_load_context_cache_document", parked_load)
    if first_writer == "save":
        action = lambda: mm.save_context_length("model-a", "https://a.example/v1", 128_000)
        expected_a = 128_000
    else:
        action = lambda: mm._invalidate_cached_context_length("model-a", "https://a.example/v1")
        expected_a = None
    worker = threading.Thread(target=action, name="first-writer")
    worker.start()
    assert loaded.wait(timeout=5)
    mm.save_context_length("model-b", "https://b.example/v1", 64_000)
    other_saved.set()
    worker.join(timeout=10)
    assert not worker.is_alive()

    assert mm.get_cached_context_length("model-b", "https://b.example/v1") == 64_000
    assert mm.get_cached_context_length("model-a", "https://a.example/v1") == expected_a


def test_invalidation_evicts_in_process_probe_caches_when_the_lock_is_unavailable(monkeypatch):
    """Only the file update needs the cross-process lock: a stale in-process probe result must be
    dropped even when that lock cannot be taken, or the next resolution reuses it for its TTL."""
    import contextlib

    import hermes_cli.auth as auth_module
    from hermes_constants import hermes_home_key

    base_url = "https://a.example/v1"
    mm._LOCAL_CTX_PROBE_CACHE[("model-a", base_url)] = (200_000, 0.0)
    mm._LOCAL_CTX_PROBE_CACHE[("ollama_show", "model-a", base_url)] = (200_000, 0.0)
    mm._BEDROCK_PROBE_FAILURE_CACHE[(hermes_home_key(), base_url, "model-a", "us-east-1")] = 0.0

    @contextlib.contextmanager
    def wedged_lock(*_args, **_kwargs):
        raise TimeoutError("Timed out waiting for the context length cache lock")
        yield

    monkeypatch.setattr(auth_module, "_file_lock", wedged_lock)

    mm._invalidate_cached_context_length("model-a", base_url)

    assert ("model-a", base_url) not in mm._LOCAL_CTX_PROBE_CACHE
    assert ("ollama_show", "model-a", base_url) not in mm._LOCAL_CTX_PROBE_CACHE
    assert not [k for k in mm._BEDROCK_PROBE_FAILURE_CACHE if k[1:3] == (base_url, "model-a")]
