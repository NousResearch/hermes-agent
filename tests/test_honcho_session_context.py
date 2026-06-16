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

    def context(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeContext()


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
