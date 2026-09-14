"""Wave 55 battery B14 race tests: gateway/rich_sent_store.py concurrent-writer invariants.

W54-F036 (lost-update): `_update` is an unlocked load->merge->dump->replace RMW.
Threads updating distinct keys load the same base snapshot; each later write
publishes a dict missing the other writers' keys.
W54-F043 (torn-write): every thread in one process shares the same
``{path}.tmp.{pid}`` temp inode, so concurrent dumps truncate/replace each other
mid-write and corrupt bytes can be published as the live index.

Design: start-barrier hammer tests. All coordination lives OUTSIDE the code
under test (a rendezvous inside the RMW body would deadlock the mutex fix), and
the race window is widened with pure CPU burn — zero sleeps. Each test asserts
the behaviour contract: after N concurrent updates, the on-disk store still
parses and contains every key that was written.
"""

from __future__ import annotations

import json
import threading

import gateway.rich_sent_store as rss


def _burn() -> None:
    """Pure-CPU stall that widens the race window without sleeping."""
    n = 0
    for i in range(4000):
        n += i * i
    assert n > 0  # keep the loop honest


def test_concurrent_updates_never_lose_distinct_keys(tmp_path, monkeypatch):
    """W54-F036: synced threads updating distinct keys must all survive.

    Pre-fix, the unlocked load->merge->replace interleaves: writers that load
    the same base snapshot each publish a dict lacking the others' keys (lost
    update). Red on the unfixed code: the final store misses most keys.
    """
    store = tmp_path / "index.json"
    monkeypatch.setattr(rss, "_store_path", lambda: str(store))

    real_load = rss._load

    def widened_load(path: str) -> dict:
        _burn()  # hold the RMW open so every writer loads the same snapshot
        return real_load(path)

    monkeypatch.setattr(rss, "_load", widened_load)

    threads_n, rounds = 16, 6
    start = threading.Barrier(threads_n)

    def worker(tid: int) -> None:
        for r in range(rounds):
            start.wait()
            rss._update("chat", f"msg-{r}-{tid}", {"t": f"v-{r}-{tid}"})

    workers = [threading.Thread(target=worker, args=(t,)) for t in range(threads_n)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=120)
        assert not w.is_alive(), "worker thread hung"

    data = real_load(str(store))
    missing = [
        f"chat:msg-{r}-{t}"
        for r in range(rounds)
        for t in range(threads_n)
        if f"chat:msg-{r}-{t}" not in data
    ]
    assert not missing, f"{len(missing)}/{threads_n * rounds} keys lost to the race"


def test_concurrent_dumps_never_publish_torn_tmp(tmp_path, monkeypatch):
    """W54-F043: two threads sharing one ``{path}.tmp.{pid}`` must not corrupt the store.

    Pre-fix both writers open the SAME temp inode ``'w'`` at once; whose bytes
    survive is scheduling luck, so one dump truncates/replaces the other
    mid-write and the published index is torn or missing keys. The two workers
    use different payload lengths so a prefix-overwrite cannot accidentally
    yield valid JSON. Red on the unfixed code.
    """
    store = tmp_path / "index.json"
    monkeypatch.setattr(rss, "_store_path", lambda: str(store))

    real_dump = json.dump

    def widened_dump(obj, fp, **kw) -> None:
        _burn()  # keep both writers inside the same dump window
        real_dump(obj, fp, **kw)

    monkeypatch.setattr(json, "dump", widened_dump)  # module-level ``json.dump`` binding

    rounds = 10
    start = threading.Barrier(2)

    def worker(tid: int) -> None:
        for r in range(rounds):
            start.wait()
            rss._update(f"chat{tid}", f"msg-{r}", {"t": "x" * (1024 + tid * 4096) + f"-{tid}-{r}"})

    workers = [threading.Thread(target=worker, args=(t,)) for t in (0, 1)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=120)
        assert not w.is_alive(), "worker thread hung"

    data = rss._load(str(store))
    missing = [
        f"chat{tid}:msg-{r}"
        for r in range(rounds)
        for tid in (0, 1)
        if f"chat{tid}:msg-{r}" not in data
    ]
    assert not missing, f"{len(missing)}/{rounds * 2} keys torn or lost"