import json
from types import SimpleNamespace

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


class FakeHonchoSession:
    def __init__(self, same_payload=False):
        self.calls = []
        self.same_payload = same_payload

    def context(self, **kwargs):
        self.calls.append(kwargs)
        query = kwargs.get("search_query")
        suffix = "same" if self.same_payload else (query or "baseline")
        return SimpleNamespace(
            summary=SimpleNamespace(content=f"summary {suffix}"),
            peer_representation=f"representation {suffix}",
            peer_card=[f"card {suffix}"],
            messages=[],
        )


def make_manager(fake_session, *, ai_observe_others=True):
    mgr = HonchoSessionManager(config=None)
    mgr._ai_observe_others = ai_observe_others
    mgr._context_tokens = 2000
    session = HonchoSession(
        key="hermes",
        user_peer_id="maoge",
        assistant_peer_id="hermes",
        honcho_session_id="hermes",
    )
    mgr._cache["hermes"] = session
    mgr._sessions_cache["hermes"] = fake_session
    return mgr


def test_honcho_context_tool_forwards_query_to_session_manager():
    class Manager:
        def __init__(self):
            self.calls = []

        def get_session_context(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"representation": "focused billing risk"}

    manager = Manager()
    provider = HonchoMemoryProvider()
    provider._manager = manager
    provider._session_key = "hermes"
    provider._session_initialized = True

    raw = provider.handle_tool_call(
        "honcho_context",
        {"peer": "user", "query": "billing risk"},
    )

    payload = json.loads(raw)
    assert "focused billing risk" in payload["result"]
    assert manager.calls == [((), {"peer": "user", "query": "billing risk"})] or manager.calls == [(("hermes",), {"peer": "user", "query": "billing risk"})]


def test_get_session_context_forwards_query_and_observer_target(monkeypatch):
    fake = FakeHonchoSession()
    mgr = make_manager(fake)
    monkeypatch.setattr(mgr, "_fetch_peer_context", lambda *a, **k: {"representation": "", "card": []})
    monkeypatch.setattr(mgr, "_fetch_conclusion_context", lambda *a, **k: {"representation": "", "card": []})

    result = mgr.get_session_context("hermes", peer="user", query="billing risk")

    focused_call = fake.calls[0]
    assert focused_call["peer_target"] == "maoge"
    assert focused_call["peer_perspective"] == "hermes"
    assert focused_call["search_query"] == "billing risk"
    assert focused_call["search_top_k"] == 8
    assert focused_call["max_conclusions"] == 8
    assert focused_call["tokens"] == 2000
    assert fake.calls[1].get("search_query") is None
    assert result["representation"] == "representation billing risk"


def test_get_session_context_uses_self_perspective_when_ai_does_not_observe_others(monkeypatch):
    fake = FakeHonchoSession()
    mgr = make_manager(fake, ai_observe_others=False)
    monkeypatch.setattr(mgr, "_fetch_peer_context", lambda *a, **k: {"representation": "", "card": []})
    monkeypatch.setattr(mgr, "_fetch_conclusion_context", lambda *a, **k: {"representation": "", "card": []})

    mgr.get_session_context("hermes", peer="user", query="billing risk")

    focused_call = fake.calls[0]
    assert focused_call["peer_target"] == "maoge"
    assert focused_call["peer_perspective"] == "maoge"


def test_query_ignored_by_session_context_falls_back_to_peer_search_context(monkeypatch):
    fake = FakeHonchoSession(same_payload=True)
    mgr = make_manager(fake)

    def fake_peer_context(peer_id, search_query=None, target=None):
        if search_query:
            return {"representation": f"focused peer {search_query}", "card": ["peer card"]}
        return {"representation": "baseline peer", "card": ["baseline card"]}

    monkeypatch.setattr(mgr, "_fetch_peer_context", fake_peer_context)
    monkeypatch.setattr(
        mgr,
        "_fetch_conclusion_context",
        lambda *a, **k: {"representation": "focused conclusion", "card": ["conclusion card"]},
    )

    result = mgr.get_session_context("hermes", peer="user", query="billing risk")

    assert result["representation"] == "focused peer billing risk"
    assert result["card"] == "peer card"


def test_truncate_with_suffix_keeps_suffix_inside_cap():
    text = "x" * 100
    truncated = HonchoSessionManager._truncate_with_suffix(text, 20, suffix=" …")
    assert len(truncated) <= 20
    assert truncated.endswith(" …")
