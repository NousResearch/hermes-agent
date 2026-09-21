"""Tests for blog.news_verify — web-grounding for AI stream named events.

Tests mock _web_search so they never hit the live network.
"""

import blog.news_verify as nv


def test_unverified_when_no_evidence(monkeypatch):
    """When search returns nothing, verify_event returns unverified."""
    monkeypatch.setattr(nv, "_web_search", lambda q: [])
    r = nv.verify_event("Grok acquired Cursor")
    assert r["verified"] is False
    assert r["snippets"] == []


def test_verified_with_evidence(monkeypatch):
    """When search returns hits, verify_event returns verified with snippets."""
    monkeypatch.setattr(
        nv, "_web_search",
        lambda q: [{"title": "X acquires Cursor", "snippet": "confirmed deal", "url": "https://example.com"}],
    )
    r = nv.verify_event("Grok acquired Cursor")
    assert r["verified"] is True
    assert len(r["snippets"]) > 0
    assert "confirmed" in r["snippets"][0]["snippet"].lower()


def test_verify_event_returns_query_in_result(monkeypatch):
    """Result dict carries the original query for traceability."""
    monkeypatch.setattr(nv, "_web_search", lambda q: [{"title": "test", "snippet": "test", "url": "u"}])
    r = nv.verify_event("something happened")
    assert r["query"] == "something happened"


def test_unverified_when_network_fails(monkeypatch):
    """Network failure degrades to unverified, never fabricates."""
    def failing_search(q):
        raise ConnectionError("network down")
    monkeypatch.setattr(nv, "_web_search", failing_search)
    r = nv.verify_event("anything")
    assert r["verified"] is False
    assert r["snippets"] == []
