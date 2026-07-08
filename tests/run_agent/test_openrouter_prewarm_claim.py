import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


def test_openrouter_prewarm_claim_is_process_singleton(monkeypatch):
    worker_count = 16
    start = threading.Barrier(worker_count)

    monkeypatch.setattr(run_agent, "_openrouter_prewarm_done", False)
    monkeypatch.setattr(run_agent, "_openrouter_prewarm_lock", threading.Lock())

    def claim_once():
        start.wait(timeout=2)
        return run_agent._claim_openrouter_prewarm()

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        claims = list(pool.map(lambda _: claim_once(), range(worker_count)))

    assert claims.count(True) == 1
    assert claims.count(False) == worker_count - 1
