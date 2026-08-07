"""B12 regression: honcho_search must return semantic CONCLUSIONS, not the
aggregated peer representation/card.

Bug: search_context called peer.context(search_query=...) which returns the
same long USER profile for any rare query, and _resolve_observer_target under
observation=off resolved the wrong (self) scope. Fix queries conclusions in
the assistant->target scope regardless of observation toggles.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


def _capturing_getter(capture, peer):
    def _get(pid):
        capture["observer"] = pid
        return peer
    return _get


def _session():
    return HonchoSession(
        key="s", user_peer_id="evgeniy", assistant_peer_id="hermes",
        honcho_session_id="sid", messages=[],
    )


def _manager(observe_others=False):
    cfg = HonchoClientConfig(observation_mode="directional")
    mgr = HonchoSessionManager(honcho=MagicMock(), config=cfg)
    # observation off by default (post-cutover) - reading must still work
    mgr._ai_observe_others = observe_others
    sess = _session()
    mgr._cache["s"] = sess
    return mgr


def _fake_peer_with_conclusions(contents, capture):
    """Fake peer whose conclusions_of(target).query returns given contents."""
    def conclusions_of(target):
        capture["target"] = target
        scope = MagicMock()
        scope.query.return_value = [SimpleNamespace(content=c) for c in contents]
        return scope
    peer = MagicMock()
    peer.conclusions_of.side_effect = conclusions_of
    return peer


class TestSearchReturnsConclusions:
    def test_returns_conclusion_excerpts_not_representation(self):
        mgr = _manager()
        capture = {}
        fake = _fake_peer_with_conclusions(
            ["MacBook Air M1, SSD 512 ГБ, разбитый дисплей", "мышь Logitech G502 X"], capture)
        mgr._get_or_create_peer = _capturing_getter(capture, fake)
        out = mgr.search_context("s", "MacBook G502", peer="user")
        assert "MacBook Air M1" in out
        assert "G502 X" in out
        # scope follows write direction: assistant observes user, not self
        assert capture["observer"] == "hermes"
        assert capture["target"] == "evgeniy"

    def test_scope_independent_of_observation_toggle(self):
        # even with observe_others True the read scope stays assistant->target
        mgr = _manager(observe_others=True)
        capture = {}
        fake = _fake_peer_with_conclusions(["факт"], capture)
        mgr._get_or_create_peer = _capturing_getter(capture, fake)
        mgr.search_context("s", "q", peer="user")
        assert capture["observer"] == "hermes" and capture["target"] == "evgeniy"

    def test_assistant_peer_self_scope(self):
        mgr = _manager()
        capture = {}
        fake = _fake_peer_with_conclusions(["identity fact"], capture)
        mgr._get_or_create_peer = _capturing_getter(capture, fake)
        mgr.search_context("s", "q", peer="ai")
        assert capture["observer"] == "hermes" and capture["target"] == "hermes"

    def test_empty_conclusions_fall_back_to_message_search(self, monkeypatch):
        # no conclusions -> upstream message search remains the fallback
        from plugins.memory.honcho import session as session_module

        mgr = _manager()
        empty = MagicMock()
        empty.conclusions_of.return_value.query.return_value = []
        mgr._get_or_create_peer = lambda pid: empty
        fake_client = MagicMock()
        fake_client.search.return_value = [
            SimpleNamespace(content="raw message hit", peer_id="evgeniy", session_id="s1")
        ]
        monkeypatch.setattr(session_module, "get_honcho_client", lambda *a, **k: fake_client)
        out = mgr.search_context("s", "q", peer="user")
        assert "raw message hit" in out
