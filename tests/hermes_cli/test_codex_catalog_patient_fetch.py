"""Patient Codex catalog discovery (port of zed-industries/zed#64925, clean-room).

The authenticated ``/backend-api/codex/models`` call can take ~30 s for some accounts. Callers wait a
bounded foreground window and fall back; the request keeps running and its answer is written to the
caches both discovery sites read, so the next read is live instead of re-stalling.
"""
from __future__ import annotations

import base64
import json
import sys
import threading
import time
from types import SimpleNamespace

from agent import codex_catalog_fetch


def _codex_jwt(subject: str) -> str:
    def enc(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    header = enc(json.dumps({"alg": "RS256"}).encode())
    payload = enc(json.dumps({"sub": subject}).encode())
    return f"{header}.{payload}.sig"


class _SlowCatalog:
    """A catalog endpoint that answers only once ``release`` is set."""

    def __init__(self, models):
        self.release = threading.Event()
        self.models = models
        self.calls = 0

    def get(self, url, headers=None, timeout=None, verify=None):
        self.calls += 1
        assert self.release.wait(5), "test catalog never released"
        return SimpleNamespace(status_code=200, json=lambda: {"models": self.models})


def _fresh_inflight(monkeypatch):
    monkeypatch.setattr(codex_catalog_fetch, "_inflight", {})


def test_picker_discovery_returns_within_the_foreground_window_and_caches_the_late_answer(monkeypatch):
    """Slow account: the picker call falls back at once instead of blocking for the HTTP timeout,
    and the catalog that lands later is persisted under the provider cache so the next open is live."""
    from hermes_cli import codex_models, models as models_mod

    _fresh_inflight(monkeypatch)
    slow = _SlowCatalog([{"slug": "gpt-6-astra", "priority": 0}, {"slug": "gpt-5.5", "priority": 1}])
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=slow.get))
    monkeypatch.setattr(codex_models, "CODEX_CATALOG_FOREGROUND_TIMEOUT", 0.2)
    stored = {}
    monkeypatch.setattr(models_mod, "update_provider_cache_entry", lambda provider, models: stored.update({provider: list(models)}))

    started = time.monotonic()
    assert codex_models._fetch_models_from_api(_codex_jwt("slow-acct")) == []
    assert time.monotonic() - started < 2, "the caller must not wait for the slow catalog"
    assert stored == {}

    slow.release.set()
    deadline = time.monotonic() + 5
    while not stored and time.monotonic() < deadline:
        time.sleep(0.02)
    cached = stored["openai-codex"]
    assert cached[0] == "gpt-6-astra" and "gpt-5.5" in cached  # account-gated row survives, priority order kept
    assert slow.calls == 1


def test_context_probe_does_not_memoize_a_miss_while_the_slow_answer_is_pending(monkeypatch):
    """The context-window probe shares the catalog request. A slow answer is not a "no catalog"
    verdict: the negative memo stays unset, and the late catalog fills the in-process caches
    (context_window AND max_context_window) without another request."""
    from agent import model_metadata as mm

    _fresh_inflight(monkeypatch)
    token = _codex_jwt("slow-acct")
    slow = _SlowCatalog([{"slug": "gpt-6-astra", "context_window": 272_000, "max_context_window": 400_000}])
    monkeypatch.setattr(mm.model_metadata_http, "get", slow.get)
    monkeypatch.setattr(mm, "_codex_oauth_context_cache", {})
    monkeypatch.setattr(mm, "_codex_oauth_max_context_cache", {})
    monkeypatch.setattr("hermes_cli.codex_models.CODEX_CATALOG_FOREGROUND_TIMEOUT", 0.2)

    assert mm._fetch_codex_oauth_context_lengths_with_source(token) == ({}, False)
    key = mm._codex_oauth_token_fingerprint(token, "")
    assert key not in mm._codex_oauth_context_cache, "a pending request must not be memoized as a miss"

    slow.release.set()
    deadline = time.monotonic() + 5
    while key not in mm._codex_oauth_context_cache and time.monotonic() < deadline:
        time.sleep(0.02)
    live, fresh = mm._fetch_codex_oauth_context_lengths_with_source(token)
    assert live == {"gpt-6-astra": 272_000} and fresh is False  # served from the cache the late answer filled
    assert mm._cached_codex_catalog_max(token, "", "gpt-6-astra") == 400_000
    assert slow.calls == 1


def test_fast_answer_is_returned_inline_and_concurrent_callers_share_one_request(monkeypatch):
    """Control: an endpoint answering inside the window behaves as before (entries returned in the
    caller's thread), and two callers racing on the same credential make one HTTP request."""
    _fresh_inflight(monkeypatch)
    slow = _SlowCatalog([{"slug": "gpt-5.5"}])
    results = []

    def _call():
        results.append(codex_catalog_fetch.fetch_catalog_patiently(
            "same-key", lambda: ([slow.get("u")], 200), foreground_timeout=5))

    threads = [threading.Thread(target=_call) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    slow.release.set()
    for t in threads:
        t.join(5)
    assert slow.calls == 1
    assert [r[1] for r in results] == [200, 200]
    assert codex_catalog_fetch._inflight == {}
