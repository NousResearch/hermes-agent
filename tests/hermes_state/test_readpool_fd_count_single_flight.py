"""Concurrent readers share one fd-table listing per cache window instead of each listing it.

With a few hundred hosted-room threads and thousands of descriptors, every thread that saw
a stale stamp listed /proc/self/fd at once, which was most of the gateway's CPU.
"""
import threading

import hermes_state_readpool as rp


def test_concurrent_headroom_checks_list_fd_table_once(monkeypatch):
    calls = []
    gate = threading.Barrier(16)

    hold = threading.Event()

    def counting_count():
        # A real listing of thousands of descriptors takes time; hold the window open so
        # every thread that checked freshness unlocked would also reach the listing.
        calls.append(1)
        hold.wait(0.3)
        return 10

    monkeypatch.setattr(rp, '_fd_soft_limit', lambda: 4096)
    monkeypatch.setattr(rp, '_open_fd_count', counting_count)
    monkeypatch.setattr(rp, '_FD_USAGE_CACHE_SECONDS', 60.0)
    monkeypatch.setattr(rp, '_fd_usage_cache', (0.0, None))
    results = []

    def worker():
        gate.wait()
        results.append(rp._fd_headroom_ok())

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [True] * 16
    assert len(calls) == 1
