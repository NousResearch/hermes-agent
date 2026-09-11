"""Tests for per-turn author attribution.

A shared session (group, channel, thread) carries turns from several
participants and from other agents, but the manager resolves one user peer
when the session is created. Every later turn was written under that peer,
so whoever created the session collected everyone else's facts.

``resolve_author_peer_id`` maps the turn's author onto its own peer and
``_flush_session`` writes each user message under it, joining the peer to
the Honcho session the first time it speaks.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSessionManager


def _config(**overrides) -> HonchoClientConfig:
    base = dict(api_key="test-key", peer_name="eri", ai_peer="hermes")
    base.update(overrides)
    return HonchoClientConfig(**base)


def _manager(config: HonchoClientConfig, runtime_id: str | None = None, runtime_id_alt: str | None = None) -> HonchoSessionManager:
    mgr = HonchoSessionManager(
        honcho=MagicMock(),
        config=config,
        runtime_user_peer_name=runtime_id,
        runtime_user_peer_name_alt=runtime_id_alt,
    )
    mgr._get_or_create_peer = MagicMock(side_effect=lambda pid: MagicMock(name=f"peer:{pid}"))
    mgr._get_or_create_honcho_session = MagicMock(return_value=(MagicMock(), []))
    return mgr


class TestResolveAuthorPeerId:

    def test_author_is_the_session_peer(self):
        """The session's own participant needs no second peer."""
        mgr = _manager(_config(), runtime_id="7654321")
        assert mgr.resolve_author_peer_id("telegram:group1", "7654321") is None


    def test_other_participant_gets_its_own_peer(self):
        mgr = _manager(_config(), runtime_id="7654321")
        assert mgr.resolve_author_peer_id("telegram:group1", "111222") == "111222"


    def test_pin_peer_name_collapses_authors(self):
        """pinPeerName is an explicit request to unify identities."""
        mgr = _manager(_config(pin_peer_name=True), runtime_id="7654321")
        assert mgr.resolve_author_peer_id("telegram:group1", "111222") is None

    def test_bot_author_lands_on_its_profile_peer(self):
        """A clean profile name that no configured peer claims is the peer as is."""
        mgr = _manager(_config(), runtime_id="7654321")
        assert mgr.resolve_author_peer_id("telegram:dm1", "bot:coder") == "coder"


    def test_bot_author_named_like_the_owner_never_lands_on_the_owner_peer(self):
        """``bot:eri`` with ``peerName: eri`` is another agent, not the operator."""
        for pinned in (False, True):
            mgr = _manager(_config(peer_name="eri", pin_peer_name=pinned), runtime_id="7654321")
            peer = mgr.resolve_author_peer_id("telegram:dm1", "bot:eri", is_bot=True)
            assert peer != mgr._declared_owner_peer_id()
            assert peer.startswith("eri-")

    def test_bot_author_never_lands_on_the_sessions_human_runtime_peer(self):
        """The human's peer on this session is not always ``peerName``: a bare or prefixed runtime id is one too."""
        cases = [
            (_config(), "coder", None, "bot:coder", "coder"),
            (_config(), "7654321", "coder", "bot:coder", "coder"),
            (_config(runtime_peer_prefix="tg_"), "7654321", None, "bot:tg_7654321", "tg_7654321"),
        ]
        for config, runtime_id, alt, author, human_peer in cases:
            mgr = _manager(config, runtime_id=runtime_id, runtime_id_alt=alt)
            assert human_peer in mgr._session_human_peer_ids("telegram:dm1")
            peer = mgr.resolve_author_peer_id("telegram:dm1", author, is_bot=True)
            assert peer != human_peer and peer.startswith(f"{human_peer}-")


    def test_connection_qualified_bot_author_keeps_its_connection(self):
        """The Desktop relays ``bot:<connection>/<profile>``; two connections' ``coder`` are two agents."""
        mgr = _manager(_config(user_peer_aliases={"bot:local/coder": "coder-here"}), runtime_id="7654321")
        east = mgr.resolve_author_peer_id("Bot-Chat", "bot:east/coder", is_bot=True)
        west = mgr.resolve_author_peer_id("Bot-Chat", "bot:west/coder", is_bot=True)
        assert east != west
        assert east.startswith("east-coder-") and west.startswith("west-coder-")
        assert mgr.resolve_author_peer_id("Bot-Chat", "bot:local/coder", is_bot=True) == "coder-here"


    def test_pin_peer_name_does_not_collapse_bot_authors(self):
        """The pin unifies the operator's accounts. A bot's words never land under the human's peer."""
        mgr = _manager(_config(pin_peer_name=True), runtime_id="7654321")
        assert mgr.resolve_author_peer_id("telegram:dm1", "bot:coder") == "coder"

    def test_platform_bot_gets_its_own_peer_even_when_pinned(self):
        """A Telegram bot arrives with a raw user id and the bot flag, never a ``bot:`` id."""
        mgr = _manager(_config(pin_peer_name=True, runtime_peer_prefix="tg_"), runtime_id="7654321")
        assert mgr.resolve_author_peer_id("telegram:dm1", "5551234", "SomeBot", is_bot=True) == "tg_5551234"


    def test_display_name_never_becomes_a_peer_id(self):
        """Display names are attacker-influenceable on most platforms."""
        mgr = _manager(_config(), runtime_id="7654321")
        assert mgr.resolve_author_peer_id("telegram:group1", None, "Alice") is None


class TestFlushAttributesMessages:
    @pytest.fixture(autouse=True)
    def _fake_sdk_session_module(self, monkeypatch):
        """The join imports SessionPeerConfig from the SDK at call time and skips silently without it."""
        module = types.ModuleType("honcho.session")
        module.SessionPeerConfig = lambda **kwargs: SimpleNamespace(**kwargs)
        monkeypatch.setitem(sys.modules, "honcho.session", module)

    def _session(self, mgr, key="telegram:group1"):
        return mgr.get_or_create(key)

    def test_author_message_written_under_the_author_peer(self):
        mgr = _manager(_config(), runtime_id="7654321")
        session = self._session(mgr)
        honcho_session = MagicMock()
        mgr._sessions_cache[session.honcho_session_id] = honcho_session

        session.add_message("user", "alice speaking", author_peer_id="alice")
        assert mgr._flush_session(session) is True

        written = honcho_session.add_messages.call_args[0][0]
        assert len(written) == 1
        # The peer object the message was built from is the author's, not the
        # session's — that is the whole point of the change.
        assert mgr._get_or_create_peer.call_args_list[-1][0][0] == "alice"


    def test_author_peer_joins_once(self):
        """A shared session's roster is open, so peers join when they write."""
        mgr = _manager(_config(), runtime_id="7654321")
        session = self._session(mgr)
        honcho_session = MagicMock()
        mgr._sessions_cache[session.honcho_session_id] = honcho_session

        session.add_message("user", "first", author_peer_id="alice")
        mgr._flush_session(session)
        session.add_message("user", "second", author_peer_id="alice")
        mgr._flush_session(session)

        assert honcho_session.add_peers.call_count == 1


    def test_join_failure_still_writes_under_the_author(self):
        """A failed join loses the observe config, never the attribution."""
        mgr = _manager(_config(), runtime_id="7654321")
        session = self._session(mgr)
        honcho_session = MagicMock()
        honcho_session.add_peers.side_effect = RuntimeError("network")
        mgr._sessions_cache[session.honcho_session_id] = honcho_session

        session.add_message("user", "alice speaking", author_peer_id="alice")
        assert mgr._flush_session(session) is True
        assert mgr._get_or_create_peer.call_args_list[-1][0][0] == "alice"
        # Not remembered as joined, so the next write retries the join.
        assert "alice" not in mgr._joined_author_peers.get(session.honcho_session_id, set())


class TestProviderReadsTheAuthor:
    def _provider(self) -> HonchoMemoryProvider:
        provider = HonchoMemoryProvider()
        provider._session_key = "telegram:group1"
        provider._manager = MagicMock()
        provider._cron_skipped = False
        provider._config = SimpleNamespace(message_max_chars=25000)
        return provider


    def test_sync_turn_attaches_the_resolved_author_peer(self):
        provider = self._provider()
        provider._session_initialized = True
        session = MagicMock()
        provider._manager.get_or_create.return_value = session
        provider._manager.resolve_author_peer_id.return_value = "alice"

        provider.on_turn_start(1, "hi", author_id="111222", author_name="Alice")
        provider.sync_turn("hi", "hello back")
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5)

        user_calls = [
            c for c in session.add_message.call_args_list if c[0][0] == "user"
        ]
        assert user_calls, "the user turn was never written"
        assert all(c[1]["author_peer_id"] == "alice" for c in user_calls)


    def test_sync_turn_resolves_before_the_write_thread_starts(self):
        """A following turn must not retag a write that is already queued."""
        provider = self._provider()
        provider._session_initialized = True
        session = MagicMock()
        provider._manager.get_or_create.return_value = session
        provider._manager.resolve_author_peer_id.return_value = "alice"

        provider.on_turn_start(1, "hi", author_id="111222")
        provider.sync_turn("hi", "hello back")
        provider._turn_author = {"id": "999", "name": None, "is_bot": False}
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5)

        provider._manager.resolve_author_peer_id.assert_called_once_with(
            "telegram:group1", "111222", None
        )
