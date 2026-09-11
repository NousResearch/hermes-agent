"""Deferred review dispatch must use the profile that admitted each item."""

import threading

import pytest

from agent.review_idle_queue import ReviewIdleQueue
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.memory_tool_store import MemoryStore


@pytest.mark.parametrize("ambient_enabled", [True, False])
def test_deferred_review_keeps_profile_config_and_memory(tmp_path, monkeypatch, ambient_enabled):
    ambient = tmp_path / "ambient"
    profiles = [tmp_path / "first", tmp_path / "second"]
    enabled = not ambient_enabled
    for home, gate in [(ambient, ambient_enabled), *[(p, enabled) for p in profiles]]:
        home.mkdir()
        (home / "config.yaml").write_text(
            f"auxiliary:\n  background_review:\n    enabled: {str(gate).lower()}\n",
            encoding="utf-8",
        )
    monkeypatch.setenv("HERMES_HOME", str(ambient))
    queue = ReviewIdleQueue()
    monkeypatch.setattr(queue, "_ensure_thread", lambda: None)
    clock = [0.0]
    queue._now = lambda: clock[0]
    writes = []

    class Parent:
        def _spawn_background_review_now(self, **kwargs):
            # Real profile-aware memory I/O at the spawn boundary; no model call.
            writes.append(MemoryStore().add("memory", kwargs["fact"]))

    parent = Parent()
    for index, home in enumerate(profiles):
        token = set_hermes_home_override(home)
        try:
            queue.enqueue(parent, str(index), {"fact": f"Project {index} uses Python", "task_cfg": {}})
        finally:
            reset_hermes_home_override(token)
    clock[0] = 4000.0  # age-out avoids contacting a model server

    class StopWorker(BaseException):
        pass

    class DrainWake(threading.Event):
        def wait(self, timeout=None):
            if queue.pending_count() == 0:
                raise StopWorker
            return True

    queue._wake = DrainWake()
    failures = []

    def drain():
        try:
            queue._run()
        except StopWorker:
            pass
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=drain, daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert not failures
    assert queue.pending_count() == 0
    assert len(writes) == (2 if enabled else 0)
    assert all(result["success"] for result in writes)
    assert not (ambient / "memories" / "MEMORY.md").exists()
    for index, home in enumerate(profiles):
        path = home / "memories" / "MEMORY.md"
        assert path.exists() == enabled
        if enabled:
            assert path.read_text(encoding="utf-8") == f"Project {index} uses Python"
