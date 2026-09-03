"""Regression tests for community plugin index fetch diagnostics."""

import logging

from hermes_cli import plugin_index


def test_remote_fetch_failure_warns_before_fallback(monkeypatch, caplog):
    """A failed community index fetch must not be silent to operators."""
    import httpx

    def fail_fetch(*args, **kwargs):
        raise httpx.ConnectError("index unavailable")

    monkeypatch.setattr(httpx, "get", fail_fetch)

    with caplog.at_level(logging.WARNING, logger=plugin_index.__name__):
        entries = plugin_index._fetch_remote()

    assert entries is None
    assert "remote fetch failed" in caplog.text
    assert "will fall back to cached or bundled index" in caplog.text
