"""Behavior contracts for HonchoMemoryProvider.journey_cards().

journey_cards() is the session-independent hook the learning-journey graph
calls to surface Honcho conclusions as memory nodes. Contract under test:

- reads conclusions from BOTH observer scopes (user self-conclusions and the
  AI peer's conclusions about the user), deduped by server id,
- normalizes to {body, timestamp} cards,
- is best-effort: unconfigured, SDK missing, or backend errors → [] (never
  raises) — the journey must render regardless of backend health.

The Honcho SDK is faked at the client boundary (get_honcho_client), the same
seam the real code resolves through; no network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho import client as client_mod


class _FakeScope:
    def __init__(self, items, page_size_cap=None):
        self._items = items
        self._page_size_cap = page_size_cap

    def list(self, size=100):
        """Mimic the SDK's SyncPage: iterating it walks ALL items across
        pages (auto-pagination), regardless of the per-page ``size``."""
        if self._page_size_cap is not None:
            assert size <= self._page_size_cap
        return iter(self._items)


class _FakePeer:
    def __init__(self, scopes):
        self._scopes = scopes

    def conclusions_of(self, target):
        return self._scopes.get(target, _FakeScope([]))


class _FakeClient:
    def __init__(self, peers):
        self._peers = peers

    def peer(self, peer_id):
        return self._peers.get(peer_id, _FakePeer({}))


def _conclusion(cid, content, created_at=None):
    return SimpleNamespace(
        id=cid,
        content=content,
        created_at=created_at or datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc),
    )


def _set_config(monkeypatch, enabled=True, peer_name: str | None = "alice", ai_peer="hermes"):
    cfg = SimpleNamespace(enabled=enabled, peer_name=peer_name, ai_peer=ai_peer)
    monkeypatch.setattr(
        client_mod.HonchoClientConfig,
        "from_global_config",
        classmethod(lambda cls, **kw: cfg),
    )
    return cfg


def _set_client(monkeypatch, client):
    monkeypatch.setattr(client_mod, "get_honcho_client", lambda cfg: client)


@pytest.fixture
def provider(monkeypatch):
    _set_config(monkeypatch)
    return HonchoMemoryProvider()


def test_reads_both_observer_scopes_and_dedupes(provider, monkeypatch):
    shared = _conclusion("dup-1", "seen by both observers")
    client = _FakeClient(
        {
            "alice": _FakePeer({"alice": _FakeScope([
                shared,
                _conclusion("a-1", "alice self-fact"),
            ])}),
            "hermes": _FakePeer({"alice": _FakeScope([
                shared,
                _conclusion("h-1", "hermes-observed fact"),
            ])}),
        }
    )
    _set_client(monkeypatch, client)

    cards = provider.journey_cards()

    bodies = [c["body"] for c in cards]
    assert bodies.count("seen by both observers") == 1  # deduped by id
    assert "alice self-fact" in bodies
    assert "hermes-observed fact" in bodies
    assert all(isinstance(c["timestamp"], datetime) for c in cards)


def test_one_scope_failing_does_not_hide_the_other(provider, monkeypatch):
    class _BoomPeer:
        def conclusions_of(self, target):
            raise RuntimeError("scope down")

    client = _FakeClient(
        {
            "alice": _BoomPeer(),
            "hermes": _FakePeer({"alice": _FakeScope([_conclusion("h-1", "still visible")])}),
        }
    )
    _set_client(monkeypatch, client)

    assert [c["body"] for c in provider.journey_cards()] == ["still visible"]


def test_respects_limit(provider, monkeypatch):
    many = [_conclusion(f"c-{i}", f"fact {i}") for i in range(50)]
    client = _FakeClient({"alice": _FakePeer({"alice": _FakeScope(many)})})
    _set_client(monkeypatch, client)

    assert len(provider.journey_cards(limit=7)) == 7


def test_unconfigured_or_broken_returns_empty(monkeypatch):
    provider = HonchoMemoryProvider()

    # Not enabled → [].
    _set_config(monkeypatch, enabled=False)
    assert provider.journey_cards() == []

    # No peer name → [].
    _set_config(monkeypatch, peer_name=None)
    assert provider.journey_cards() == []

    # Client construction blowing up (no key, SDK missing) → [], never raises.
    _set_config(monkeypatch)
    monkeypatch.setattr(
        client_mod, "get_honcho_client",
        lambda cfg: (_ for _ in ()).throw(ValueError("no api key")),
    )
    assert provider.journey_cards() == []


def test_pagination_reaches_beyond_first_page(provider, monkeypatch):
    """A bulk history import can leave many hundreds of conclusions; reading
    only .items of the first page would silently hide the older ones from the
    journey timeline. Iterating the page object must walk all of them."""
    many = [_conclusion(f"c-{i}", f"fact {i}") for i in range(350)]
    client = _FakeClient({"alice": _FakePeer({"alice": _FakeScope(many)})})
    _set_client(monkeypatch, client)

    cards = provider.journey_cards()

    assert len(cards) == 350
    assert cards[-1]["body"] == "fact 349"
