"""``PromptCacheWarmer.claim_edge`` is the one cache-edge claim: it validates and latches under the warmer
lock, so two owners racing on the same idle chat (ticker and reaper, sweep and eviction) get one action
between them, and a release claim ends the live run so no keep-alive goes out after it."""

import threading
from types import SimpleNamespace

import agent.prompt_cache_lifecycle as pcl
from agent.prompt_cache_lifecycle import edge


def test_two_racing_owners_get_one_compaction(monkeypatch):
    """Both owners pause after the decision returns: only the latch under the lock keeps it to one."""
    warmer = pcl.PromptCacheWarmer(SimpleNamespace(compression_enabled=True))
    warmer._spent = edge._Spent(due_at=0.0, tier_s=3600.0, witness=())
    monkeypatch.setattr(edge, "_session_gate", lambda agent: None)
    monkeypatch.setattr(edge, "_window_gate", lambda agent, spent, now: None)
    monkeypatch.setattr(edge, "_grew_enough", lambda agent: True)
    barrier = threading.Barrier(2, timeout=5)
    decide = pcl.PromptCacheWarmer.edge_decision

    def decide_then_pause(self, **kw):
        result = decide(self, **kw)
        try:
            barrier.wait()  # both owners hold a "compact" decision at once (only possible unlatched)
        except threading.BrokenBarrierError:
            pass
        return result

    monkeypatch.setattr(pcl.PromptCacheWarmer, "edge_decision", decide_then_pause)
    results = []
    threads = [threading.Thread(target=lambda: results.append(warmer.claim_edge(now=1.0))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    barrier.abort()
    assert results.count("compact") == 1 and results.count(None) == 1
    assert warmer._compacted and warmer._spent is None
