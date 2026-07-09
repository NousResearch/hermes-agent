"""LIVE integration test for the gbrain document leg (Phase 2b).

Hits the REAL warm gbrain serve (launchd ai.gbrain.serve, 127.0.0.1:8199) with real
OAuth creds. Skips cleanly when the serve is down or creds are absent, so CI and
machines without a brain never fail. Run from the repo root:
  venv/bin/python -m pytest plugins/memory/mem0/test_gbrain_live_integration.py -v -o addopts=""
"""
import json
import os
import time
import urllib.request

import pytest

from plugins.memory.mem0 import gbrain_recall

_URL = "http://127.0.0.1:8199"
_CREDS = os.path.expanduser("~/gbrain/.gbrain/rail-client.env")
_DEADLINE_S = 4.0  # the production per-turn budget (mem0_gbrain.total_deadline_s)


def _serve_up() -> bool:
    try:
        with urllib.request.urlopen(f"{_URL}/health", timeout=3) as r:
            return b'"status":"ok"' in r.read()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (_serve_up() and os.path.isfile(_CREDS)),
    reason="warm gbrain serve not reachable on 127.0.0.1:8199 (or rail creds absent)",
)

# 5 real queries spanning distinct corpus areas (fleet infra, smart home,
# hardware, media stack, agent architecture).
_QUERIES = [
    "reolink doorbell voice pipeline",
    "wake on lan dual boot windows ubuntu",
    "hermes agent fleet backup restore",
    "plex jellyfin media server admin",
    "mac studio local llm serving",
]


@pytest.mark.parametrize("query", _QUERIES)
def test_live_query_returns_sane_pointers_within_deadline(query):
    t0 = time.monotonic()
    hits = gbrain_recall.gbrain_search(
        query, limit=3, min_score=0.5, deadline_s=_DEADLINE_S,
        url=_URL, creds_path=_CREDS,
    )
    elapsed = time.monotonic() - t0
    # deadline honored with margin for the watchdog to fire + unwind
    assert elapsed < _DEADLINE_S + 1.0, f"took {elapsed:.2f}s"
    assert isinstance(hits, list)
    assert hits, f"no results for {query!r} (corpus should cover this)"
    for h in hits:
        assert set(h.keys()) == {"file", "title", "score", "line", "docid"}
        assert h["file"] and isinstance(h["file"], str)
        assert isinstance(h["score"], float) and 0.5 <= h["score"] <= 1.0
        assert h["docid"].startswith("gbrain:")
        assert isinstance(h["line"], int)
    # record actual output for the phase report (harmless side-channel)
    _log = os.environ.get("GBRAIN_ITEST_LOG")
    if _log:
        with open(_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"query": query, "elapsed_s": round(elapsed, 2),
                                "hits": hits}) + "\n")


def test_live_token_is_cached_across_calls():
    """Second call must not re-mint (one auth per process)."""
    gbrain_recall.gbrain_search("token cache probe one", limit=1, min_score=0.5,
                                deadline_s=_DEADLINE_S, url=_URL, creds_path=_CREDS)
    key = (_URL, _CREDS)
    with gbrain_recall._token_lock:
        ent = dict(gbrain_recall._token_cache.get(key) or {})
    assert ent.get("token")
    tok_before = ent["token"]
    gbrain_recall.gbrain_search("token cache probe two", limit=1, min_score=0.5,
                                deadline_s=_DEADLINE_S, url=_URL, creds_path=_CREDS)
    with gbrain_recall._token_lock:
        ent2 = dict(gbrain_recall._token_cache.get(key) or {})
    assert ent2.get("token") == tok_before
