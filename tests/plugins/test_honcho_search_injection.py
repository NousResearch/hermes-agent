"""Regression tests for Honcho query-scoped search injection."""

from types import SimpleNamespace

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


class FakeManager:
    def __init__(self):
        self.search_queries = []

    def get_prefetch_context(self, session_key, user_message=None, *, scope="session"):
        assert scope == "session"
        return {"summary": "DM session summary only"}

    def pop_context_result(self, session_key):
        return None

    def search_context(self, session_key, query, max_tokens=800, peer="user", *, scope="session"):
        self.search_queries.append((session_key, query, max_tokens, peer, scope))
        if scope == "global":
            return "LAB_TOOL animal = tatu canastra"
        return ""


class FakeHoncho:
    def __init__(self, messages):
        self.messages = messages
        self.queries = []

    def search(self, query, filters=None, limit=10):
        self.queries.append(query)
        if query in {"LAB_TOOL animal", "TOPICO_A_CANARIO valor"}:
            return self.messages
        return []


class ExplodingSearchHoncho:
    def search(self, *args, **kwargs):
        raise AssertionError("auto-inject must not use global honcho.search")


def test_prefetch_uses_session_scoped_search_for_auto_injection():
    provider = HonchoMemoryProvider()
    manager = FakeManager()
    provider._manager = manager
    provider._session_key = "lab-session"
    provider._recall_mode = "context"
    provider._turn_count = 1
    provider._last_dialectic_turn = 0  # avoid first-turn dialectic thread in this unit test
    provider._dialectic_cadence = 999
    provider._config = SimpleNamespace(context_tokens=None, search_injection_tokens=800)

    result = provider.prefetch("qual é o animal do LAB_TOOL?")

    assert "## Session Summary" in result
    assert "DM session summary only" in result
    assert "## Relevant Memory Search Results" not in result
    assert "LAB_TOOL animal = tatu canastra" not in result
    assert manager.search_queries == [
        ("lab-session", "qual é o animal do LAB_TOOL?", 800, "user", "session")
    ]


def test_auto_prefetch_does_not_include_broad_peer_context(monkeypatch):
    manager = HonchoSessionManager(honcho=FakeHoncho([]))
    manager._cache["dm-session"] = HonchoSession(
        key="dm-session",
        user_peer_id="965870659",
        assistant_peer_id="iris",
        honcho_session_id="dm-session",
    )

    class FakeSummary:
        content = "DM session summary only"

    class FakeCtx:
        summary = FakeSummary()

    class FakeSession:
        def context(self, summary=True):
            return FakeCtx()

    manager._sessions_cache["dm-session"] = FakeSession()

    def forbidden_peer_context(*args, **kwargs):
        raise AssertionError("auto prefetch must not call broad peer.context")

    monkeypatch.setattr(manager, "_fetch_peer_context", forbidden_peer_context)

    ctx = manager.get_prefetch_context("dm-session", scope="session")

    assert ctx == {"summary": "DM session summary only"}


def test_auto_search_context_does_not_use_global_honcho_search():
    manager = HonchoSessionManager(honcho=ExplodingSearchHoncho())
    manager._cache["dm-session"] = HonchoSession(
        key="dm-session",
        user_peer_id="965870659",
        assistant_peer_id="iris",
        honcho_session_id="dm-session",
    )

    result = manager.search_context(
        "dm-session",
        "qual é o canário do tópico A?",
        max_tokens=100,
        peer="user",
        scope="session",
    )

    assert result == ""


def test_explicit_global_search_context_can_use_raw_honcho_search():
    manager = HonchoSessionManager(honcho=FakeHoncho([
        SimpleNamespace(peer_id="965870659", content="TOPICO_A_CANARIO valor EMA-264 cor azul"),
    ]))
    manager._cache["dm-session"] = HonchoSession(
        key="dm-session",
        user_peer_id="965870659",
        assistant_peer_id="iris",
        honcho_session_id="dm-session",
    )

    result = manager.search_context(
        "dm-session",
        "TOPICO_A_CANARIO valor",
        max_tokens=100,
        peer="user",
        scope="global",
    )

    assert "EMA-264" in result


def test_global_search_context_prefers_raw_honcho_search_excerpts_over_peer_context(monkeypatch):
    manager = HonchoSessionManager(honcho=FakeHoncho([
        SimpleNamespace(peer_id="philip_lab", content="LAB_TOOL animal = tatu canastra"),
    ]))
    manager._cache["lab-session"] = HonchoSession(
        key="lab-session",
        user_peer_id="philip_lab",
        assistant_peer_id="honchogatewaylab",
        honcho_session_id="lab-session",
    )

    monkeypatch.setattr(
        manager,
        "_fetch_peer_context",
        lambda *args, **kwargs: {"representation": "old broad representation", "card": []},
    )

    result = manager.search_context("lab-session", "LAB_TOOL animal", max_tokens=50, scope="global")

    assert "LAB_TOOL animal = tatu canastra" in result
    assert "old broad representation" not in result
