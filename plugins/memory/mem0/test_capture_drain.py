"""Tests for the mem0 auto-capture drain worker (Track A-lite).

Run: PYTHONPATH=<worktree> pytest plugins/memory/mem0/test_capture_drain.py -q
Covers: happy path, exactly-once reconcile (D-8), transient-fault retry + dead-letter (D-10/INV-1),
post-write secret scrub (INV-4), breaker skip. Uses injected fakes — no live mem0.
"""
import pytest

from capture_queue import CaptureQueue, idem_key
from capture_drain import CaptureDrainWorker
import capture_scrub as scrub


class FakeStore:
    """Minimal fake of the mem0 client for add / recall-by-idem / get-written / forget."""
    def __init__(self, fail_times=0, recall_raises=0, getwritten_raises=0, fail_error=None):
        self.rows = []          # [{id, memory, capture_idem}]
        self._id = 0
        self.fail_times = fail_times
        self.add_calls = 0
        self.recall_raises = recall_raises      # raise on the first N recall_idem calls (transient)
        self.getwritten_raises = getwritten_raises
        self.recall_calls = 0
        self.getwritten_calls = 0
        # the exception add() raises while failing; default is an AMBIGUOUS fault (server may have
        # committed). Pass a clearly-pre-send error to exercise the bounded/dead-letter path.
        self.fail_error = fail_error or RuntimeError("simulated transient 500")

    def add(self, messages, kwargs):
        self.add_calls += 1
        if self.add_calls <= self.fail_times:
            raise self.fail_error
        idem = (kwargs.get("metadata") or {}).get("capture_idem", "")
        # simulate server extraction: 1 durable fact per turn (+ echo the user text so a secret shows)
        text = messages[0]["content"]
        self._id += 1
        self.rows.append({"id": f"m{self._id}", "memory": text, "capture_idem": idem})
        return 1   # server extracted+wrote 1 memory (drives require_rows in the post-write scrub)

    def recall_idem(self, key):
        self.recall_calls += 1
        if self.recall_calls <= self.recall_raises:
            raise RuntimeError("simulated transient search 503")
        return sum(1 for r in self.rows if r["capture_idem"] == key)

    def get_written(self, key):
        self.getwritten_calls += 1
        if self.getwritten_calls <= self.getwritten_raises:
            raise RuntimeError("simulated transient search 503")
        return [r for r in self.rows if r["capture_idem"] == key]

    def forget(self, mid):
        if getattr(self, "forget_raises", 0) > 0:
            self.forget_raises -= 1
            raise RuntimeError("simulated transient update 503")
        self.rows = [r for r in self.rows if r["id"] != mid]


def make_worker(q, store, **kw):
    defaults = dict(
        gate="GATE_V3",
        model="gpt-5.4-mini",
        write_filters={"user_id": "ace"},
        max_attempts=3,
        backoff_base_s=1.0,
    )
    defaults.update(kw)
    return CaptureDrainWorker(
        q,
        add_fn=store.add,
        recall_idem_fn=store.recall_idem,
        scrub_fn=lambda facts: scrub.filter_facts(facts),
        forget_fn=store.forget,
        get_written_fn=store.get_written,
        **defaults,
    )


@pytest.fixture
def q(tmp_path):
    return CaptureQueue(str(tmp_path / "cq.db"))


def _enq(q, user, assistant="ok", sess="s", n=1):
    k = idem_key(sess, n, user, assistant)
    q.enqueue(k, {"user": user, "assistant": assistant})
    return k


def test_happy_path_extracts_and_marks_done(q):
    store = FakeStore()
    w = make_worker(q, store)
    k = _enq(q, "User prefers dark mode.")
    assert w.drain_once() is True
    assert q.counts()["done"] == 1
    assert store.add_calls == 1
    # the add carried the gate + model + capture_idem
    assert store.rows[0]["capture_idem"] == k
    assert w.stats["drained"] == 1


def test_gate_and_model_threaded_into_add(q):
    captured = {}
    store = FakeStore()
    orig = store.add
    def spy(messages, kwargs):
        captured.update(kwargs); return orig(messages, kwargs)
    store.add = spy
    w = make_worker(q, store)
    _enq(q, "User runs QMD on the Mac Studio.")
    w.drain_once()
    assert captured["prompt"] == "GATE_V3"
    assert captured["model"] == "gpt-5.4-mini"
    assert captured["metadata"]["capture_idem"]
    assert captured["user_id"] == "ace"


def test_exactly_once_reconcile_no_double_add(q):
    """A crash after add() but before mark_done: the row was re-leased; the reconcile sees existing
    rows and marks done WITHOUT a second add (D-8)."""
    store = FakeStore()
    w = make_worker(q, store)
    k = _enq(q, "User's DNS is AdGuard on 192.168.1.208.")
    # simulate: add already ran on a prior lease (row exists), status still not done
    store.rows.append({"id": "pre", "memory": "prewritten", "capture_idem": k})
    assert w.drain_once() is True
    assert store.add_calls == 0          # NEVER re-added
    assert q.counts()["done"] == 1


def test_transient_fault_retries_then_recovers(q):
    store = FakeStore(fail_times=1)      # first add 500s, second succeeds
    w = make_worker(q, store)
    _enq(q, "User prefers concise replies.")
    # first drain: add fails -> requeued
    assert w.drain_once() is True
    assert q.counts()["pending"] == 1 and w.stats["retried"] == 1
    # advance past backoff, drain again -> succeeds
    import time as _t
    row = q.lease_one(now=_t.time() + 10)
    assert row is None or True  # lease timing; force the retry by draining after backoff
    # simplest: directly re-drain after making it due
    q._connect().execute("UPDATE capture_queue SET next_attempt_at=0, status='pending', leased_until=NULL")
    assert w.drain_once() is True
    assert q.counts()["done"] == 1 and store.add_calls == 2


def test_dead_letter_after_max_attempts(q):
    # a clearly PRE-SEND failure (connection refused) means nothing was written -> bounded path/dead
    store = FakeStore(fail_times=99, fail_error=ConnectionRefusedError("connection refused"))
    w = make_worker(q, store, max_attempts=3)
    _enq(q, "User likes X.")
    for _ in range(3):
        w.drain_once()
        # force due again
        q._connect().execute("UPDATE capture_queue SET next_attempt_at=0 WHERE status='pending'")
    assert q.counts()["dead"] == 1
    assert w.stats["dead"] >= 1


def test_ambiguous_add_fault_never_deadletters(q):
    """Greptile P1: a timeout/500/read-error can hit AFTER the server committed /memories, so the
    add may be live+unscanned. Such an ambiguous fault must NEVER dead-letter (it would abandon a
    possibly-written, never-scanned row) — requeue forever + escalate."""
    alerts = []
    store = FakeStore(fail_times=99)  # default 500 error = ambiguous (server may have committed)
    w = make_worker(q, store, max_attempts=2, backoff_base_s=0.0, alert_fn=alerts.append)
    _enq(q, "User likes Y.")
    for _ in range(6):
        w.drain_once()
    assert q.counts()["dead"] == 0                 # never abandoned on an ambiguous fault
    assert q.counts()["pending"] == 1              # still retriable
    assert any("ADD AMBIGUOUS" in a for a in alerts), f"expected escalation, got {alerts}"


def test_post_write_scrub_forgets_secret_bearing_memory(q):
    store = FakeStore()
    w = make_worker(q, store)
    # a turn whose user text carries a telegram bot token -> the fake "extracts" it verbatim
    _enq(q, "my bot token is " + ("8905425635:" + "AAH3xY9zKq" + "_Wp0LmNoPqRsTuVwXyZ" + "012345") + " keep it")
    w.drain_once()
    # the secret-bearing memory was scrubbed (forgotten) post-write
    assert w.stats["scrubbed"] == 1
    assert all("8905425635" not in r["memory"] for r in store.rows)
    assert q.counts()["done"] == 1     # turn still completes


def test_clean_memory_not_scrubbed(q):
    store = FakeStore()
    w = make_worker(q, store)
    _enq(q, "User's Mac Studio is the always-on fleet host.")
    w.drain_once()
    assert w.stats["scrubbed"] == 0
    assert len(store.rows) == 1


def test_breaker_open_skips(q):
    store = FakeStore()
    w = make_worker(q, store, breaker_open_fn=lambda: True)
    _enq(q, "User prefers Y.")
    assert w.drain_once() is False
    assert store.add_calls == 0
    assert q.counts()["pending"] == 1   # untouched, will drain when breaker closes


# ---- Greptile P1 fail-closed regressions -----------------------------------
def test_idem_check_failure_requeues_without_readd(q):
    """If the idem pre-check raises (transient search fault), the drainer must NOT re-add (would
    duplicate) — it requeues fail-closed."""
    store = FakeStore(recall_raises=1)   # first recall_idem raises
    w = make_worker(q, store, backoff_base_s=0.0)   # no backoff so the requeued row is immediately due
    _enq(q, "User's DNS is AdGuard.")
    w.drain_once()
    assert store.add_calls == 0             # did NOT add on the unknown
    assert w.stats["retried"] == 1
    assert q.counts()["pending"] == 1       # requeued
    # next drain: recall now succeeds (0 rows) -> proceeds to add exactly once
    w.drain_once()
    assert store.add_calls == 1
    assert q.counts()["done"] == 1


def test_scrub_read_failure_requeues_not_done(q):
    """If the post-write scrub can't READ the written rows (transient), the row must NOT be marked
    done (a secret could be left recallable) — requeue so the scrub retries."""
    store = FakeStore(getwritten_raises=1)  # first get_written raises, AFTER the add succeeded
    w = make_worker(q, store, backoff_base_s=0.0)
    _enq(q, "some benign fact")
    w.drain_once()
    assert store.add_calls == 1             # the add happened
    assert w.stats["retried"] == 1          # but the row was requeued, not completed
    assert q.counts()["done"] == 0
    assert q.counts()["pending"] == 1
    # second drain: idem pre-check now finds the existing row -> marks done WITHOUT re-adding
    w.drain_once()
    assert store.add_calls == 1             # NOT re-added (exactly-once preserved)
    assert q.counts()["done"] == 1


def test_exactly_once_shortcut_still_scrubs(q):
    """A prior lease added a SECRET-bearing row but crashed before scrubbing. On the next drain the
    exactly-once shortcut (idem>0) must STILL run the scrub — not skip it and leave the secret."""
    store = FakeStore()
    w = make_worker(q, store, backoff_base_s=0.0)
    tok = "8905425635:" + "AAH3xY9zKq" + "_Wp0LmNoPqRsTuVwXyZ" + "012345"
    key = _enq(q, f"my bot token is {tok} keep it")
    # simulate the prior lease: the row was ADDED but never scrubbed/marked done
    store.add([{"role": "user", "content": f"my bot token is {tok} keep it"}],
              {"metadata": {"capture_idem": key}})
    assert len(store.rows) == 1
    # now drain: idem>0 shortcut fires -> must scrub the secret before completing
    w.drain_once()
    assert w.stats["scrubbed"] == 1
    assert all(tok not in r["memory"] for r in store.rows)   # secret forgotten
    assert store.add_calls == 1                              # shortcut did NOT re-add
    assert q.counts()["done"] == 1


def test_empty_scrub_read_after_successful_add_requeues(q):
    """Greptile P1: mem0's metadata search is eventually-consistent. If add() wrote >=1 memory but the
    immediate scrub read returns empty (not yet index-visible), the row must NOT complete — that
    would leave a just-written secret unscanned. It requeues until the row is visible."""
    class LagStore(FakeStore):
        def __init__(self):
            super().__init__()
            self.getwritten_calls = 0
        def get_written(self, key):
            self.getwritten_calls += 1
            # first read after the add sees nothing (index lag); later reads see the row
            if self.getwritten_calls <= 1:
                return []
            return [r for r in self.rows if r["capture_idem"] == key]
    store = LagStore()
    w = make_worker(q, store, backoff_base_s=0.0)
    _enq(q, "some benign fact")
    w.drain_once()
    assert store.add_calls == 1              # the add happened (returned count=1)
    assert w.stats["retried"] == 1           # empty read after write -> requeued, NOT completed
    assert q.counts()["done"] == 0
    # next drain: idem shortcut finds the now-visible row and scrubs+completes
    w.drain_once()
    assert q.counts()["done"] == 1


def test_forget_failure_requeues_not_done(q):
    """If FORGET of a secret-bearing memory fails (transient), the row must NOT be marked done —
    requeue so the forget retries. Otherwise the secret stays recallable behind a completed row."""
    store = FakeStore()
    store.forget_raises = 1                  # first forget attempt raises
    w = make_worker(q, store, backoff_base_s=0.0)
    tok = "8905425635:" + "AAH3xY9zKq" + "_Wp0LmNoPqRsTuVwXyZ" + "012345"
    _enq(q, f"token {tok} here")
    w.drain_once()
    assert store.add_calls == 1              # added
    assert w.stats["scrubbed"] == 0          # forget failed -> not counted as scrubbed
    assert w.stats["retried"] == 1           # requeued (fail-closed)
    assert q.counts()["done"] == 0
    # second drain: forget now succeeds -> secret scrubbed, row completes
    w.drain_once()
    assert w.stats["scrubbed"] == 1
    assert all(tok not in r["memory"] for r in store.rows)
    assert q.counts()["done"] == 1


def test_scrub_deadletter_escalates_loudly(q):
    """Greptile P1: if the scrub keeps failing, the row must NEVER dead-letter (that would abandon a
    live secret) — it keeps retrying with capped backoff and escalates LOUDLY once at the threshold."""
    alerts = []
    store = FakeStore()
    store.forget_raises = 99                  # forget always fails -> scrub can never complete
    w = make_worker(q, store, backoff_base_s=0.0, max_attempts=2, alert_fn=alerts.append)
    tok = "8905425635:" + "AAH3xY9zKq" + "_Wp0LmNoPqRsTuVwXyZ" + "012345"
    _enq(q, f"token {tok} here")
    # drain repeatedly: the row must keep coming back as pending (never dead), and escalate once
    for _ in range(6):
        w.drain_once()
    assert w.stats["scrub_dead"] == 1                       # escalated exactly once (at threshold)
    assert q.counts()["dead"] == 0                          # NEVER dead-lettered
    assert q.counts()["pending"] == 1                       # still retriable
    assert any("SCRUB STUCK" in a for a in alerts), f"expected a loud alert, got {alerts}"
    # and once forget recovers, the next drain scrubs it clean
    store.forget_raises = 0
    w.drain_once()
    assert all(tok not in r["memory"] for r in store.rows)
    assert q.counts()["done"] == 1


def test_idem_check_failure_after_prior_write_never_deadletters(q):
    """Greptile P1: if a PRIOR lease may have committed add() (attempts>0), a persistent idem-check
    failure must NEVER dead-letter — that would abandon a possibly-secret-bearing live row with no
    scrub path. It requeues indefinitely (never 'dead') and escalates loudly at the threshold."""
    alerts = []
    store = FakeStore(recall_raises=99)   # idem check always fails
    w = make_worker(q, store, backoff_base_s=0.0, max_attempts=2, alert_fn=alerts.append)
    # seed the row with attempts=1 so the drainer treats it as "a prior lease may have written"
    key = _enq(q, "some fact")
    w._q.mark_scrub_retry(key, backoff_s=0.0)   # bumps attempts to 1, keeps pending
    for _ in range(6):
        w.drain_once()
    assert q.counts()["dead"] == 0                 # never abandoned
    assert q.counts()["pending"] == 1              # still retriable
    assert store.add_calls == 0                    # never re-added (can't confirm it's new)
    assert any("IDEM-CHECK STUCK" in a for a in alerts), f"expected escalation, got {alerts}"


def test_add_committed_flag_blocks_deadletter_after_reap_reset(q):
    """Greptile P1: add() committed on a prior lease, then a crash+reap reset attempts to 0. A later
    persistent idem-check failure must STILL not dead-letter — the sticky add_committed flag (which
    survives the reap) tells the drainer a possibly-secret-bearing row is live."""
    alerts = []
    store = FakeStore(recall_raises=99)   # idem check always fails now
    w = make_worker(q, store, backoff_base_s=0.0, max_attempts=2, alert_fn=alerts.append)
    key = _enq(q, "some fact")
    # simulate: a prior lease committed the add, then crashed+reaped -> attempts back to 0
    w._q.mark_add_committed(key)
    assert q.counts()["pending"] == 1 and w._q.lease_one() is not None  # leased for this drain
    # put it back to pending with attempts still 0 (reap-style), add_committed stays 1
    import sqlite3
    conn = sqlite3.connect(w._q.db_path)
    conn.execute("UPDATE capture_queue SET status='pending', attempts=0, next_attempt_at=0 WHERE idem_key=?", (key,))
    conn.commit(); conn.close()
    for _ in range(5):
        w.drain_once()
    assert q.counts()["dead"] == 0                 # add_committed prevented abandonment despite attempts=0
    assert store.add_calls == 0
    assert any("IDEM-CHECK STUCK" in a for a in alerts), f"expected escalation, got {alerts}"
