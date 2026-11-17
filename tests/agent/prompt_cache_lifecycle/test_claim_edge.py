"""``PromptCacheWarmer.claim_edge`` is the one cache-edge claim: it validates and latches under the warmer
lock, so two owners racing on the same idle chat (ticker and reaper, sweep and eviction) get one action
between them, and a release claim ends the live run so no keep-alive goes out after it."""

import threading
from types import SimpleNamespace

import agent.prompt_cache_lifecycle as pcl
import agent.prompt_cache_lifecycle.edge as edge


def _due_warmer(monkeypatch):
    warmer = pcl.PromptCacheWarmer(SimpleNamespace(compression_enabled=True))
    warmer._spent = edge._Spent(due_at=0.0, tier_s=3600.0, witness=())
    barrier = threading.Barrier(2, timeout=1)

    def slow_gate(agent):
        try:
            barrier.wait()  # both claimers are inside the decision at once (only possible unlocked)
        except threading.BrokenBarrierError:
            pass

    monkeypatch.setattr(edge, "_session_gate", slow_gate)
    monkeypatch.setattr(edge, "_window_gate", lambda agent, spent, now: None)
    monkeypatch.setattr(edge, "_grew_enough", lambda agent: True)
    return warmer, barrier


def test_two_racing_owners_get_one_compaction(monkeypatch):
    warmer, barrier = _due_warmer(monkeypatch)
    results = []
    threads = [threading.Thread(target=lambda: results.append(warmer.claim_edge(now=1.0))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    barrier.abort()
    assert results.count("compact") == 1 and results.count(None) == 1
    assert warmer._compacted and warmer._spent is None
