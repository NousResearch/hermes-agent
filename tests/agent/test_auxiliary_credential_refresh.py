"""A refreshed subscription credential must not be served from cache.

Claude Code refreshes its OAuth token in place: `~/.claude/.credentials.json` is
rewritten while the pool entry keeps its id. A long-lived process (the desktop
`serve` backend, a gateway) that keys its auxiliary-client cache on the entry id
alone keeps handing out a client built around the revoked token, and every
auxiliary call — context compression above all — fails 401 until restart.

These are behaviour contracts: they assert the relationship between the live
secret and the cache discriminator, not any particular hash.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import agent.auxiliary_client as ax


@pytest.fixture
def pooled_entry(monkeypatch):
    """A pool entry whose secret can be swapped, as a refresh does."""
    entry = SimpleNamespace(id="entry-1")
    secret = {"value": "sk-ant-oat01-first"}
    monkeypatch.setattr(ax, "_peek_pool_entry", lambda provider: entry)
    monkeypatch.setattr(ax, "_pool_runtime_api_key", lambda e: secret["value"])
    return secret


def test_refreshed_secret_changes_the_cache_hint(pooled_entry):
    """Same entry id, new token: the discriminator must move.

    This is the exact regression — the CLI refreshed the token, the id stayed,
    and the cached client kept using the revoked one.
    """
    before = ax._pool_cache_hint("anthropic")
    pooled_entry["value"] = "sk-ant-oat01-refreshed"
    after = ax._pool_cache_hint("anthropic")

    assert before != after, "a refreshed credential must invalidate the cached client"


def test_an_unchanged_secret_keeps_the_hint_stable(pooled_entry):
    """The cache must still work: no rebuild when nothing changed."""
    assert ax._pool_cache_hint("anthropic") == ax._pool_cache_hint("anthropic")


def test_the_hint_never_carries_the_secret(pooled_entry):
    """The hint reaches logs and cache dumps; the token must not ride along."""
    pooled_entry["value"] = "sk-ant-oat01-super-secret-value"
    hint = ax._pool_cache_hint("anthropic")

    assert "super-secret-value" not in hint
    assert "sk-ant" not in hint


def test_the_entry_id_still_discriminates(monkeypatch):
    """Two entries holding the same secret remain distinct cache entries."""
    secret = "sk-ant-oat01-shared"
    monkeypatch.setattr(ax, "_pool_runtime_api_key", lambda e: secret)

    monkeypatch.setattr(ax, "_peek_pool_entry", lambda p: SimpleNamespace(id="entry-a"))
    first = ax._pool_cache_hint("anthropic")
    monkeypatch.setattr(ax, "_peek_pool_entry", lambda p: SimpleNamespace(id="entry-b"))
    second = ax._pool_cache_hint("anthropic")

    assert first != second


def test_an_unreadable_secret_degrades_to_the_id(monkeypatch):
    """No secret available: fall back to the old behaviour, stably.

    Returning something volatile here would miss the cache on every call and
    rebuild a client per request.
    """
    monkeypatch.setattr(ax, "_peek_pool_entry", lambda p: SimpleNamespace(id="entry-1"))
    monkeypatch.setattr(ax, "_pool_runtime_api_key", lambda e: None)

    hint = ax._pool_cache_hint("anthropic")
    assert hint == ax._pool_cache_hint("anthropic")
    assert "entry-1" in hint


def test_no_pool_entry_yields_no_hint(monkeypatch):
    monkeypatch.setattr(ax, "_peek_pool_entry", lambda p: None)
    assert ax._pool_cache_hint("anthropic") == ""
