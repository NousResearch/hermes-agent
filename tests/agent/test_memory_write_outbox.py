"""Behavior tests for durable external-provider memory write recovery."""

import json
import threading
import time

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from agent.memory_write_outbox import MemoryWriteOutbox


class _Provider(MemoryProvider):
    def __init__(self, *, fail: bool) -> None:
        self.fail = fail
        self.calls = []

    @property
    def name(self) -> str:
        return "external-test"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def shutdown(self) -> None:
        pass

    def on_memory_write(self, action, target, content, metadata=None):
        if self.fail:
            raise RuntimeError("gateway unavailable")
        self.calls.append((action, target, content, dict(metadata or {})))


class _FailingThenOrderedProvider(_Provider):
    def __init__(self, failures: int) -> None:
        super().__init__(fail=False)
        self.failures = failures

    def on_memory_write(self, action, target, content, metadata=None):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("gateway unavailable")
        super().on_memory_write(action, target, content, metadata)


def test_outbox_deduplicates_and_bounds_pending_writes(tmp_path):
    outbox = MemoryWriteOutbox(tmp_path, max_entries_per_provider=2)

    first = outbox.enqueue("provider", "add", "memory", "one", {"session_id": "s1"})
    duplicate = outbox.enqueue(
        "provider",
        "add",
        "memory",
        "one",
        {"session_id": "s2", "tool_call_id": "retry"},
    )
    outbox.enqueue("provider", "add", "memory", "two", {})
    bounded = outbox.enqueue("provider", "add", "memory", "three", {})

    assert first["queued"] is True
    assert duplicate["deduplicated"] is True
    assert bounded["dropped"] == 1
    assert outbox.pending_count("provider") == 2


def test_failed_write_is_visible_and_replayed_after_restart(tmp_path, caplog):
    warnings = []
    failing = _Provider(fail=True)
    manager = MemoryManager(
        external_write_alerts=True,
        warning_callback=warnings.append,
        alert_cooldown_seconds=3600,
    )
    manager.add_provider(failing)
    manager.initialize_all("session-1", hermes_home=str(tmp_path))

    result = manager.notify_memory_tool_write(
        json.dumps({"success": True}),
        {"action": "add", "target": "user", "content": "likes tea"},
        build_metadata=lambda: {"session_id": "session-1"},
    )
    visible = json.loads(result)

    assert visible["success"] is True
    assert visible["external_provider_writes"] == [{
        "provider": "external-test",
        "success": False,
        "queued": True,
        "error": "gateway unavailable",
    }]
    assert "on_memory_write failed" in caplog.text
    assert len(warnings) == 1

    manager.notify_memory_tool_write(
        json.dumps({"success": True}),
        {"action": "add", "target": "user", "content": "likes coffee"},
    )
    assert len(warnings) == 1, "persistent cooldown must suppress repeated outage alerts"

    recovered = _Provider(fail=False)
    restarted = MemoryManager()
    restarted.add_provider(recovered)
    restarted.initialize_all("session-2", hermes_home=str(tmp_path))

    assert recovered.calls == [
        ("add", "user", "likes tea", {"session_id": "session-1"}),
        ("add", "user", "likes coffee", {}),
    ]
    assert MemoryWriteOutbox(tmp_path).pending_count("external-test") == 0


def test_new_write_queues_behind_failed_fifo_head(tmp_path):
    failing = _FailingThenOrderedProvider(failures=2)
    manager = MemoryManager()
    manager.add_provider(failing)
    manager.initialize_all("session-1", hermes_home=str(tmp_path))

    manager.notify_memory_tool_write(
        json.dumps({"success": True}),
        {"action": "remove", "target": "user", "old_text": "old"},
    )
    result = manager.notify_memory_tool_write(
        json.dumps({"success": True}),
        {"action": "add", "target": "user", "content": "new"},
    )

    assert failing.calls == []
    assert json.loads(result)["external_provider_writes"][0]["queued"] is True

    failing.failures = 0
    restarted = MemoryManager()
    restarted.add_provider(failing)
    restarted.initialize_all("session-2", hermes_home=str(tmp_path))

    assert [call[:3] for call in failing.calls] == [
        ("remove", "user", ""),
        ("add", "user", "new"),
    ]


def test_concurrent_replay_claims_each_row_once(tmp_path):
    first = MemoryWriteOutbox(tmp_path)
    second = MemoryWriteOutbox(tmp_path)
    first.enqueue("provider", "add", "memory", "one", {})
    delivery_started = threading.Barrier(2)
    release_delivery = threading.Event()
    calls = []

    def deliver(action, target, content, metadata):
        calls.append((action, target, content, metadata))
        delivery_started.wait(timeout=5)
        assert release_delivery.wait(timeout=5)

    worker = threading.Thread(target=lambda: first.replay("provider", deliver))
    worker.start()
    delivery_started.wait(timeout=5)

    blocked = second.replay("provider", deliver)
    release_delivery.set()
    worker.join(timeout=5)

    assert worker.is_alive() is False
    assert blocked["blocked"] is True
    assert blocked["remaining"] == 1
    assert len(calls) == 1
    assert first.pending_count("provider") == 0


def test_bound_never_evicts_claimed_head(tmp_path):
    first = MemoryWriteOutbox(tmp_path, max_entries_per_provider=1)
    second = MemoryWriteOutbox(tmp_path, max_entries_per_provider=1)
    first.enqueue("provider", "add", "memory", "old", {})
    started = threading.Event()
    release = threading.Event()

    def fail_after_release(action, target, content, metadata):
        started.set()
        assert release.wait(timeout=5)
        raise RuntimeError("still unavailable")

    result = {}
    worker = threading.Thread(
        target=lambda: result.update(first.replay("provider", fail_after_release))
    )
    worker.start()
    assert started.wait(timeout=5)

    overflow = second.enqueue("provider", "add", "memory", "new", {})
    assert overflow["queued"] is False
    assert overflow["dropped"] == 1

    release.set()
    worker.join(timeout=5)
    assert worker.is_alive() is False
    assert result["error"] == "still unavailable"
    assert first.pending_count("provider") == 1

    delivered = []
    replayed = second.replay(
        "provider",
        lambda action, target, content, metadata: delivered.append(content),
    )
    assert replayed["replayed"] == 1
    assert delivered == ["old"]


def test_claim_is_renewed_while_delivery_exceeds_lease(tmp_path):
    first = MemoryWriteOutbox(tmp_path, claim_lease_seconds=1)
    second = MemoryWriteOutbox(tmp_path, claim_lease_seconds=1)
    first.enqueue("provider", "add", "memory", "one", {})
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_deliver(action, target, content, metadata):
        calls.append(content)
        started.set()
        assert release.wait(timeout=5)

    first_result = {}
    worker = threading.Thread(
        target=lambda: first_result.update(first.replay("provider", slow_deliver))
    )
    worker.start()
    assert started.wait(timeout=5)
    time.sleep(1.25)

    blocked = second.replay("provider", slow_deliver)
    assert blocked["blocked"] is True
    assert calls == ["one"]

    release.set()
    worker.join(timeout=5)
    assert worker.is_alive() is False
    assert first_result["replayed"] == 1
    assert first.pending_count("provider") == 0
