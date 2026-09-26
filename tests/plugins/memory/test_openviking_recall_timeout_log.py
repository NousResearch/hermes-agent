"""Recall timeouts must be visible: a silent timeout injects no memories and shows no error."""

import logging

import httpx
import pytest

from plugins.memory.openviking import _is_timeout_error


@pytest.mark.parametrize("error, expected", [
    (httpx.ReadTimeout("read"), True),
    (httpx.ConnectTimeout("connect"), True),
    (TimeoutError("socket"), True),
    (httpx.HTTPError("boom"), False),
    (ValueError("boom"), False),
])
def test_timeout_classification(error, expected):
    assert _is_timeout_error(error) is expected


def test_recall_timeout_logs_warning(monkeypatch, caplog):
    from plugins.memory import openviking as mod

    provider = mod.OpenVikingMemoryProvider.__new__(mod.OpenVikingMemoryProvider)
    monkeypatch.setattr(provider, "_user_space", lambda *a, **k: "u1", raising=False)
    monkeypatch.setattr(provider, "_recall_config", lambda: {
        "timeout_seconds": 4.0, "request_timeout_seconds": 3.0, "limit": 6,
        "score_threshold": 0.15, "max_injected_chars": 4000, "full_read_limit": 2,
        "prefer_abstract": False, "resources": False}, raising=False)

    def boom(*a, **k):
        raise httpx.ReadTimeout("upstream slow")

    monkeypatch.setattr(provider, "_post_prefetch_search", boom, raising=False)
    with caplog.at_level(logging.WARNING):
        assert provider._search_prefetch_context("what did we decide about billing",
                                                     session_id="sess", client=object()) == ""
    assert any("recall timed out" in r.message for r in caplog.records)
