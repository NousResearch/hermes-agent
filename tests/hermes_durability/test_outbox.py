
import pytest


from hermes_durability import DurableRuntime, RetryPolicy
from hermes_durability.guardrail import Envelope


class MemoryReceiver:
    """Simulates an external service that honours idempotency keys."""

    def __init__(self, fail_first_n: int = 0):
        self.by_key: dict[str, dict] = {}
        self.send_calls = 0
        self.fail_first_n = fail_first_n

    def send(self, envelope: Envelope, idempotency_key: str) -> dict:
        self.send_calls += 1
        if self.send_calls <= self.fail_first_n:
            raise ConnectionError("transient")
        # idempotent: duplicate keys do not create duplicate messages
        if idempotency_key not in self.by_key:
            self.by_key[idempotency_key] = dict(envelope.payload)
        return {"id": idempotency_key}


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "o.db")


def test_enqueue_commit_deliver(db):
    recv = MemoryReceiver()
    rt = DurableRuntime(db, adapters={"mem": recv}, start_worker=False)
    with rt.transaction("s1") as txn:
        oid = txn.enqueue_outbound("mem", {"text": "hello world"})
    assert rt.worker.drain_once() == 1
    assert list(recv.by_key) == [oid]
    rt.close()


def test_rollback_never_sends(db):
    recv = MemoryReceiver()
    rt = DurableRuntime(db, adapters={"mem": recv}, start_worker=False)
    txn = rt.transaction("s1")
    txn.enqueue_outbound("mem", {"text": "should not go"})
    txn.rollback()
    assert rt.worker.drain_once() == 0
    assert recv.by_key == {}
    rt.close()


def test_retry_then_success_exactly_once(db):
    recv = MemoryReceiver(fail_first_n=2)
    rt = DurableRuntime(db, adapters={"mem": recv},
                        retry=RetryPolicy(base_delay=0, max_attempts=5),
                        start_worker=False)
    with rt.transaction("s1") as txn:
        txn.enqueue_outbound("mem", {"text": "retry me"})
    total = 0
    for _ in range(5):
        total += rt.worker.drain_once()
    assert total == 1
    assert recv.send_calls == 3
    assert len(recv.by_key) == 1
    rt.close()


def test_dead_letter_and_replay(db):
    recv = MemoryReceiver(fail_first_n=100)
    rt = DurableRuntime(db, adapters={"mem": recv},
                        retry=RetryPolicy(base_delay=0, max_attempts=3),
                        start_worker=False)
    with rt.transaction("s1") as txn:
        oid = txn.enqueue_outbound("mem", {"text": "doomed"})
    for _ in range(5):
        rt.worker.drain_once()
    row = rt.journal._conn.execute(
        "SELECT status FROM outbox WHERE outbox_id = ?", (oid,)).fetchone()
    assert row[0] == "deadletter"
    assert rt.journal._conn.execute(
        "SELECT COUNT(*) FROM dlq").fetchone()[0] == 1
    # manual replay after the receiver recovers
    recv.fail_first_n = 0
    assert rt.worker.replay_dead_letter(oid)
    assert rt.worker.drain_once() == 1
    assert oid in recv.by_key
    rt.close()


def test_per_session_ordering(db):
    order = []

    class OrderedRecv(MemoryReceiver):
        def send(self, envelope, idempotency_key):
            order.append(envelope.payload["n"])
            return super().send(envelope, idempotency_key)

    recv = OrderedRecv()
    rt = DurableRuntime(db, adapters={"mem": recv}, start_worker=False)
    for n in range(10):
        with rt.transaction("s1") as txn:
            txn.enqueue_outbound("mem", {"text": f"msg {n}", "n": n})
    rt.worker.drain_once()
    assert order == list(range(10))
    rt.close()


def test_crash_after_send_before_mark_is_deduped_by_key(db):
    """Simulate: send succeeded, process died before OutboxDelivered."""
    recv = MemoryReceiver()
    rt = DurableRuntime(db, adapters={"mem": recv}, start_worker=False)
    with rt.transaction("s1") as txn:
        oid = txn.enqueue_outbound("mem", {"text": "once only"})
    # send manually, but do NOT finalize (crash window)
    recv.send(Envelope("s1", "mem", {"text": "once only"}, oid), oid)
    rt.close()
    # restart: recovery finds the row still pending and retries
    rt2 = DurableRuntime(db, adapters={"mem": recv}, start_worker=False)
    report = rt2.recover()
    assert report["pending_outbox"] == 1
    assert report["delivered_on_recovery"] == 1
    assert recv.send_calls == 2          # at-least-once attempts...
    assert len(recv.by_key) == 1         # ...exactly-once effect
    rt2.close()
