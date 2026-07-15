"""Tests for Honcho conclusion peer resolution.

Regression: create_conclusion / delete_conclusion must route through
_resolve_observer_target (the same path used by get_peer_card,
set_peer_card, search_context, etc.) so that the observer peer object
carries workspace context into the conclusions_of() call.

See issue #37759 and PR #35988 for the original bug report.
"""

from types import SimpleNamespace

from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


class _FakeConclusionsScope:
    """Records create/delete calls for assertions."""

    def __init__(self):
        self.created = []
        self.deleted = []

    def create(self, items):
        self.created.extend(items)

    def delete(self, conclusion_id):
        self.deleted.append(conclusion_id)


class _FakePeer:
    """Minimal peer stub that records conclusions_of calls."""

    def __init__(self, peer_id):
        self.peer_id = peer_id
        self.conclusions_scopes = {}

    def conclusions_of(self, target_peer_id):
        if target_peer_id not in self.conclusions_scopes:
            self.conclusions_scopes[target_peer_id] = _FakeConclusionsScope()
        return self.conclusions_scopes[target_peer_id]


def _make_manager(ai_observe_others=True):
    """Build a manager with a cached session and controllable peer factory."""
    cfg = SimpleNamespace(
        write_frequency="turn",
        dialectic_reasoning_level="low",
        dialectic_dynamic=True,
        dialectic_max_chars=600,
        observation_mode="directional",
        user_observe_me=True,
        user_observe_others=True,
        ai_observe_me=True,
        ai_observe_others=ai_observe_others,
        message_max_chars=25000,
        dialectic_max_input_chars=10000,
    )
    mgr = HonchoSessionManager(honcho=SimpleNamespace(), config=cfg)
    session = HonchoSession(
        key="test-session",
        user_peer_id="chris",
        assistant_peer_id="hermes",
        honcho_session_id="test-session",
    )
    mgr._cache[session.key] = session

    # Inject fake peers so _get_or_create_peer returns our stubs
    peers = {}

    def _fake_get_or_create(peer_id):
        if peer_id not in peers:
            peers[peer_id] = _FakePeer(peer_id)
        return peers[peer_id]

    mgr._get_or_create_peer = _fake_get_or_create
    return mgr, session, peers


# --- create_conclusion tests ---


def test_create_conclusion_user_alias_uses_assistant_observer():
    """When ai_observe_others=True, conclusions about 'user' should route
    through the assistant peer (observer) targeting the user peer."""
    mgr, session, peers = _make_manager(ai_observe_others=True)

    result = mgr.create_conclusion("test-session", "likes dark mode", peer="user")

    assert result is True
    hermes_peer = peers["hermes"]
    scope = hermes_peer.conclusions_scopes["chris"]
    assert len(scope.created) == 1
    assert scope.created[0]["content"] == "likes dark mode"


def test_create_conclusion_user_alias_uses_user_observer_when_ai_cannot_observe():
    """When ai_observe_others=False, conclusions about 'user' should route
    through the user peer (self-observation)."""
    mgr, session, peers = _make_manager(ai_observe_others=False)

    result = mgr.create_conclusion("test-session", "likes dark mode", peer="user")

    assert result is True
    chris_peer = peers["chris"]
    scope = chris_peer.conclusions_scopes["chris"]
    assert len(scope.created) == 1
    assert scope.created[0]["content"] == "likes dark mode"


def test_create_conclusion_assistant_self_observation():
    """Conclusions about the assistant itself should route through the
    assistant peer targeting itself."""
    mgr, session, peers = _make_manager(ai_observe_others=True)

    result = mgr.create_conclusion("test-session", "should be concise", peer="hermes")

    assert result is True
    hermes_peer = peers["hermes"]
    scope = hermes_peer.conclusions_scopes["hermes"]
    assert len(scope.created) == 1
    assert scope.created[0]["content"] == "should be concise"


def test_create_conclusion_empty_content_returns_false():
    mgr, _, _ = _make_manager()
    assert mgr.create_conclusion("test-session", "") is False
    assert mgr.create_conclusion("test-session", "   ") is False


def test_create_conclusion_missing_session_returns_false():
    mgr, _, _ = _make_manager()
    assert mgr.create_conclusion("nonexistent", "content") is False


# --- delete_conclusion tests ---


def test_delete_conclusion_user_alias_uses_assistant_observer():
    mgr, session, peers = _make_manager(ai_observe_others=True)

    result = mgr.delete_conclusion("test-session", "conc-123", peer="user")

    assert result is True
    hermes_peer = peers["hermes"]
    scope = hermes_peer.conclusions_scopes["chris"]
    assert scope.deleted == ["conc-123"]


def test_delete_conclusion_user_alias_uses_user_observer_when_ai_cannot_observe():
    mgr, session, peers = _make_manager(ai_observe_others=False)

    result = mgr.delete_conclusion("test-session", "conc-456", peer="user")

    assert result is True
    chris_peer = peers["chris"]
    scope = chris_peer.conclusions_scopes["chris"]
    assert scope.deleted == ["conc-456"]


def test_delete_conclusion_assistant_self_observation():
    mgr, session, peers = _make_manager(ai_observe_others=True)

    result = mgr.delete_conclusion("test-session", "conc-789", peer="hermes")

    assert result is True
    hermes_peer = peers["hermes"]
    scope = hermes_peer.conclusions_scopes["hermes"]
    assert scope.deleted == ["conc-789"]
