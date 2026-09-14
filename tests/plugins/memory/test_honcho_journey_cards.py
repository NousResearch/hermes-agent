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


def _conclusion(cid, content, created_at=None, session_id=None):
    return SimpleNamespace(
        id=cid,
        content=content,
        created_at=created_at or datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc),
        session_id=session_id,
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


def test_cards_carry_session_provenance(provider, monkeypatch):
    """Conclusions record the session they were derived from; the card must
    pass it through so journey surfaces can resolve the originating
    conversation. A missing session id stays None (never fabricated)."""
    client = _FakeClient(
        {
            "alice": _FakePeer({"alice": _FakeScope([
                _conclusion("c-1", "sourced fact", session_id="20260101_000000_abc123"),
                _conclusion("c-2", "orphan fact"),
            ])}),
        }
    )
    _set_client(monkeypatch, client)

    by_body = {c["body"]: c for c in provider.journey_cards()}
    assert by_body["sourced fact"]["session_id"] == "20260101_000000_abc123"
    assert by_body["orphan fact"]["session_id"] is None


# ── journey_session_messages() — the source corpus behind a conclusion ──────


class _FakeMessagePage:
    def __init__(self, items):
        self._items = items

    def __iter__(self):
        return iter(self._items)


class _FakeSession:
    def __init__(self, sid, items):
        self.id = sid
        self._items = items

    def messages(self, size=100):
        return _FakeMessagePage(self._items)


class _FakeSessionClient(_FakeClient):
    """Client whose session(id) is get-or-create — reaching it for an unknown
    id would CREATE a session (the trap the existence probe must avoid)."""

    def __init__(self, sessions):
        super().__init__({})
        self._sessions = sessions
        self.created = []

    def sessions(self, filters=None, size=50):
        if filters and "id" in filters:
            wanted = str(filters["id"])
            return iter([s for s in self._sessions.values() if s.id == wanted])
        return iter(self._sessions.values())

    def session(self, sid):
        if sid not in self._sessions:
            self.created.append(sid)  # the bug: get-or-create side effect
            self._sessions[sid] = _FakeSession(sid, [])
        return self._sessions[sid]


def _message(content, peer="alice", created_at=None):
    return SimpleNamespace(
        content=content,
        peer_id=peer,
        created_at=created_at or datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc),
    )


def test_session_messages_returns_corpus(provider, monkeypatch):
    sess = _FakeSession("sess-1", [
        _message("hello", peer="alice"),
        _message("hi back", peer="hermes"),
        _message("   "),  # blank content is dropped
    ])
    _set_client(monkeypatch, _FakeSessionClient({"sess-1": sess}))

    msgs = provider.journey_session_messages("sess-1")

    assert [(m["peer"], m["content"]) for m in msgs] == [("alice", "hello"), ("hermes", "hi back")]
    assert all(isinstance(m["timestamp"], datetime) for m in msgs)


def test_session_messages_carry_roles(provider, monkeypatch):
    """The configured user peer speaks as 'user'; every other peer (hermes,
    chatgpt, any future import source) speaks as 'assistant'. Session
    materialization relies on this instead of guessing from peer names."""
    sess = _FakeSession("sess-1", [
        _message("question", peer="alice"),
        _message("answer", peer="chatgpt"),
        _message("follow-up", peer="alice"),
        _message("more", peer="hermes"),
    ])
    _set_client(monkeypatch, _FakeSessionClient({"sess-1": sess}))

    roles = [m["role"] for m in provider.journey_session_messages("sess-1")]

    assert roles == ["user", "assistant", "user", "assistant"]


def test_session_messages_unknown_id_never_creates(provider, monkeypatch):
    """client.session() is get-or-create server-side; a read-only journey
    lookup for an unknown id must probe first and return [] without ever
    touching the creating accessor."""
    client = _FakeSessionClient({"sess-1": _FakeSession("sess-1", [_message("x")])})
    _set_client(monkeypatch, client)

    assert provider.journey_session_messages("no-such-session") == []
    assert client.created == []


def test_session_messages_best_effort(provider, monkeypatch):
    # Empty/blank id → [].
    assert provider.journey_session_messages("") == []

    # Backend down → [], never raises.
    monkeypatch.setattr(
        client_mod, "get_honcho_client",
        lambda cfg: (_ for _ in ()).throw(ConnectionError("backend down")),
    )
    assert provider.journey_session_messages("sess-1") == []


def test_session_messages_respects_limit(provider, monkeypatch):
    sess = _FakeSession("sess-1", [_message(f"m{i}") for i in range(40)])
    _set_client(monkeypatch, _FakeSessionClient({"sess-1": sess}))

    assert len(provider.journey_session_messages("sess-1", limit=5)) == 5
