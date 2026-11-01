"""Account-scoped Browserbase paid-feature 402 cache (issue #106759 slice 3).

``create_session`` already drops ``keepAlive`` then ``proxies`` on HTTP 402.
Those drops must be remembered per ``(base_url, project_id)`` so a later
create for the same account omits the paid keys on the *first* POST
(no repeat 402 renegotiation). Cache is fail-open: miss, 5xx, and
connection errors must not disable paid features.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from plugins.browser.browserbase.provider import BrowserbaseBrowserProvider


@pytest.fixture(autouse=True)
def _isolate_paid_feature_cache() -> None:
    """Process-lifetime cache must not leak across tests."""
    from plugins.browser.browserbase import provider as bb

    cache = getattr(bb, "_PAID_FEATURE_DROP_CACHE", None)
    if cache is not None:
        cache.clear()
    yield
    if cache is not None:
        cache.clear()


@pytest.fixture
def bb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROWSERBASE_API_KEY", "test-key")
    monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "proj-a")
    monkeypatch.setenv("BROWSERBASE_BASE_URL", "https://bb.test")


def _ok(session_id: str = "sess-1") -> Mock:
    resp = Mock()
    resp.status_code = 200
    resp.ok = True
    resp.text = ""
    resp.json.return_value = {"id": session_id, "connectUrl": "ws://cdp.test/connect"}
    return resp


def _http(status: int, text: str = "") -> Mock:
    resp = Mock()
    resp.status_code = status
    resp.ok = 200 <= status < 300
    resp.text = text
    return resp


def _patch_post(monkeypatch: pytest.MonkeyPatch, handler):
    posts: List[Dict[str, Any]] = []

    def fake_post(self, url, headers, payload, *, wrap_errors=True):
        posts.append(dict(payload))
        return handler(payload, wrap_errors=wrap_errors)

    monkeypatch.setattr(BrowserbaseBrowserProvider, "_post_create", fake_post)
    return posts


def _402_keepalive_then_ok(payload, *, wrap_errors=True):
    if payload.get("keepAlive"):
        return _http(402, "Payment Required")
    return _ok()


class TestPaidFeatureCacheAcrossCreates:
    def test_second_create_omits_learned_keepalive_without_renegotiation(
        self, monkeypatch: pytest.MonkeyPatch, bb_env: None
    ) -> None:
        posts = _patch_post(monkeypatch, _402_keepalive_then_ok)

        first = BrowserbaseBrowserProvider().create_session("task-1")
        assert first["features"]["keep_alive"] is False
        assert any(p.get("keepAlive") for p in posts), "cache-miss must still probe keepAlive"
        first_post_count = len(posts)

        BrowserbaseBrowserProvider().create_session("task-2")
        second_posts = posts[first_post_count:]

        assert len(second_posts) == 1, (
            "second create for the same project must POST once without "
            "repeating the 402 keepAlive renegotiation"
        )
        assert "keepAlive" not in second_posts[0]

    def test_cache_not_shared_across_project_ids(
        self, monkeypatch: pytest.MonkeyPatch, bb_env: None
    ) -> None:
        posts = _patch_post(monkeypatch, _402_keepalive_then_ok)

        BrowserbaseBrowserProvider().create_session("task-a")
        after_first = len(posts)
        assert any(p.get("keepAlive") for p in posts[:after_first])

        monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "proj-b")
        BrowserbaseBrowserProvider().create_session("task-b")
        other_project_posts = posts[after_first:]

        assert other_project_posts, "different project_id must still attempt create"
        assert other_project_posts[0].get("keepAlive") is True

    def test_500_does_not_poison_cache(
        self, monkeypatch: pytest.MonkeyPatch, bb_env: None
    ) -> None:
        calls = {"n": 0}

        def handler(payload, *, wrap_errors=True):
            calls["n"] += 1
            if calls["n"] == 1:
                return _http(500, "Internal Server Error")
            if payload.get("keepAlive"):
                return _http(402, "Payment Required")
            return _ok()

        posts = _patch_post(monkeypatch, handler)

        with pytest.raises(RuntimeError, match="500"):
            BrowserbaseBrowserProvider().create_session("task-fail")

        BrowserbaseBrowserProvider().create_session("task-retry")
        assert posts[0].get("keepAlive") is True
        assert posts[1].get("keepAlive") is True

    def test_connection_error_does_not_poison_cache(
        self, monkeypatch: pytest.MonkeyPatch, bb_env: None
    ) -> None:
        calls = {"n": 0}

        def handler(payload, *, wrap_errors=True):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Browserbase API connection failed: connection reset")
            if payload.get("keepAlive"):
                return _http(402, "Payment Required")
            return _ok()

        posts = _patch_post(monkeypatch, handler)

        with pytest.raises(RuntimeError, match="connection"):
            BrowserbaseBrowserProvider().create_session("task-fail")

        BrowserbaseBrowserProvider().create_session("task-retry")
        assert posts[0].get("keepAlive") is True
        assert posts[1].get("keepAlive") is True


class TestPaidFeatureCacheFailOpenControls:
    def test_cache_not_shared_across_base_url(
        self, monkeypatch: pytest.MonkeyPatch, bb_env: None
    ) -> None:
        posts = _patch_post(monkeypatch, _402_keepalive_then_ok)

        BrowserbaseBrowserProvider().create_session("task-a")
        after_first = len(posts)

        monkeypatch.setenv("BROWSERBASE_BASE_URL", "https://bb-other.test")
        BrowserbaseBrowserProvider().create_session("task-b")
        other_host_posts = posts[after_first:]

        assert other_host_posts[0].get("keepAlive") is True

    def test_loop_still_renegotiates_on_cache_miss(
        self, monkeypatch: pytest.MonkeyPatch, bb_env: None
    ) -> None:
        """First encounter must keep the existing keepAlive-then-proxies 402 loop."""
        def handler(payload, *, wrap_errors=True):
            if payload.get("keepAlive") or payload.get("proxies"):
                return _http(402, "Payment Required")
            return _ok()

        posts = _patch_post(monkeypatch, handler)
        first = BrowserbaseBrowserProvider().create_session("task-1")
        assert posts == [
            {"projectId": "proj-a", "keepAlive": True, "proxies": True},
            {"projectId": "proj-a", "proxies": True},
            {"projectId": "proj-a"},
        ]
        second = BrowserbaseBrowserProvider().create_session("task-2")
        assert posts[3:] == [{"projectId": "proj-a"}]
        for result in (first, second):
            assert result["features"]["keep_alive"] is False
            assert result["features"]["proxies"] is False


def test_cached_drops_remain_consistent_during_concurrent_growth(
    monkeypatch: pytest.MonkeyPatch, bb_env: None
) -> None:
    from plugins.browser.browserbase import provider as bb

    scope = ("https://bb.test", "proj-a")
    bb._record_paid_drops(scope, {"keepAlive"})
    reading = Event()
    recorded = Event()
    apply_cached = bb._apply_cached_paid_drops
    observed_drops = []

    class InterleavedPayload(dict):
        def __contains__(self, key):
            if key == "keepAlive":
                # Pause with the old implementation's shared-set iterator live.
                reading.set()
                assert recorded.wait(5), "writer did not update the cache"
            return super().__contains__(key)

    def apply_with_interleaving(payload, account_scope):
        controlled = InterleavedPayload(payload)
        before = set(controlled)
        dropped = apply_cached(controlled, account_scope)
        assert dropped == before - set(controlled)
        observed_drops.append(dropped)
        payload.clear()
        payload.update(controlled)
        return dropped

    def record_proxies():
        assert reading.wait(5), "reader did not reach the cached key"
        try:
            bb._record_paid_drops(scope, {"proxies"})
        finally:
            recorded.set()

    monkeypatch.setattr(bb, "_apply_cached_paid_drops", apply_with_interleaving)
    posts = _patch_post(monkeypatch, lambda payload, **kwargs: _ok())
    with ThreadPoolExecutor(max_workers=1) as executor:
        writer = executor.submit(record_proxies)
        result = BrowserbaseBrowserProvider().create_session("task-concurrent")
        writer.result(timeout=5)

    assert observed_drops == [{"keepAlive", "proxies"}]
    assert len(posts) == 1
    assert "keepAlive" not in posts[0]
    assert "proxies" not in posts[0]
    assert result["features"]["keep_alive"] is False
    assert result["features"]["proxies"] is False
