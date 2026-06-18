"""Tests for Honcho session context peer resolution."""

from types import SimpleNamespace

from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


class _FakeSummary:
    content = "summary"


class _FakeContext:
    summary = _FakeSummary()
    peer_representation = "representation"
    peer_card = ["fact"]
    messages = []


class _RecordingHonchoSession:
    def __init__(self):
        self.calls = []
        self.added_messages = []

    def context(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeContext()

    def add_messages(self, messages):
        self.added_messages.extend(messages)


class _RecordingPeer:
    def __init__(self, peer_id):
        self.id = peer_id
        self.message_calls = []

    def message(self, content, **kwargs):
        self.message_calls.append({"content": content, **kwargs})
        return {"peer_id": self.id, "content": content, **kwargs}


def _manager_with_cached_session(*, ai_observe_others=True):
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
    fake_honcho_session = _RecordingHonchoSession()
    mgr._cache[session.key] = session
    mgr._sessions_cache[session.honcho_session_id] = fake_honcho_session
    return mgr, fake_honcho_session


def test_session_context_user_alias_uses_assistant_observer_when_ai_can_observe_others():
    mgr, fake = _manager_with_cached_session(ai_observe_others=True)

    result = mgr.get_session_context("test-session", peer="user")

    assert result["summary"] == "summary"
    assert fake.calls == [
        {
            "summary": True,
            "peer_target": "chris",
            "peer_perspective": "hermes",
        }
    ]


def test_session_context_explicit_user_peer_matches_user_alias():
    mgr, fake = _manager_with_cached_session(ai_observe_others=True)

    mgr.get_session_context("test-session", peer="chris")

    assert fake.calls == [
        {
            "summary": True,
            "peer_target": "chris",
            "peer_perspective": "hermes",
        }
    ]


def test_session_context_user_alias_uses_user_self_observer_when_ai_cannot_observe_others():
    mgr, fake = _manager_with_cached_session(ai_observe_others=False)

    mgr.get_session_context("test-session", peer="user")

    assert fake.calls == [
        {
            "summary": True,
            "peer_target": "chris",
            "peer_perspective": "chris",
        }
    ]


def test_flush_session_forwards_message_metadata_and_configuration():
    mgr, fake = _manager_with_cached_session(ai_observe_others=True)
    session = mgr._cache["test-session"]
    mgr._peers_cache["chris"] = _RecordingPeer("chris")
    mgr._peers_cache["hermes"] = _RecordingPeer("hermes")

    session.add_message(
        "user",
        "[The user sent a text document: 'research.md'.]",
        metadata={"hermes_kind": "reference_document"},
        configuration={"reasoning": {"enabled": False}},
    )
    session.add_message("assistant", "Recibido")

    assert mgr._flush_session(session) is True

    assert fake.added_messages == [
        {
            "peer_id": "chris",
            "content": "[The user sent a text document: 'research.md'.]",
            "metadata": {"hermes_kind": "reference_document"},
            "configuration": {"reasoning": {"enabled": False}},
        },
        {
            "peer_id": "hermes",
            "content": "Recibido",
        },
    ]
