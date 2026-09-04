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
    def test_no_author_keeps_the_session_peer(self):
        """An unnamed author is not attributable — never guess a peer for it."""
        mgr = _manager(_config(), runtime_id="7654321")
        assert mgr.resolve_author_peer_id("telegram:group1", None) is None
        assert mgr.resolve_author_peer_id("telegram:group1", "") is None

    def test_author_is_the_session_peer(self):
        """The session's own participant needs no second peer."""
        mgr = _manager(_config(), runtime_id="7654321")
        assert mgr.resolve_author_peer_id("telegram:group1", "7654321") is None

    def test_alt_runtime_id_is_the_session_peer(self):
        """A transport may name the participant by either id (Telegram UID or username)."""
        mgr = _manager(_config(), runtime_id="7654321", runtime_id_alt="eri_tg")
        assert mgr.resolve_author_peer_id("telegram:group1", "eri_tg") is None

    def test_other_participant_gets_its_own_peer(self):
        mgr = _manager(_config(), runtime_id="7654321")
        assert mgr.resolve_author_peer_id("telegram:group1", "111222") == "111222"

    def test_alias_wins(self):
        """An aliased account lands on its named peer whichever turn it wrote."""
        mgr = _manager(
            _config(user_peer_aliases={"111222": "alice"}),
            runtime_id="7654321",
        )
        assert mgr.resolve_author_peer_id("telegram:group1", "111222") == "alice"

    def test_runtime_prefix_applies(self):
        mgr = _manager(_config(runtime_peer_prefix="telegram_"), runtime_id="7654321")
        assert mgr.resolve_author_peer_id("telegram:group1", "111222") == "telegram_111222"

    def test_pin_peer_name_collapses_authors(self):
        """pinPeerName is an explicit request to unify identities."""
        mgr = _manager(_config(pin_peer_name=True), runtime_id="7654321")
        assert mgr.resolve_author_peer_id("telegram:group1", "111222") is None

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

    def test_unattributed_message_keeps_the_session_peer(self):
        mgr = _manager(_config(), runtime_id="7654321")
        session = self._session(mgr)
        honcho_session = MagicMock()
        mgr._sessions_cache[session.honcho_session_id] = honcho_session

        session.add_message("user", "owner speaking")
        assert mgr._flush_session(session) is True
        honcho_session.add_peers.assert_not_called()

    def test_assistant_message_ignores_author(self):
        """The reply is the agent's however the turn arrived."""
        mgr = _manager(_config(), runtime_id="7654321")
        session = self._session(mgr)
        honcho_session = MagicMock()
        mgr._sessions_cache[session.honcho_session_id] = honcho_session

        session.add_message("assistant", "reply", author_peer_id="alice")
        assert mgr._flush_session(session) is True
        honcho_session.add_peers.assert_not_called()

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

    def test_two_authors_each_join(self):
        mgr = _manager(_config(), runtime_id="7654321")
        session = self._session(mgr)
        honcho_session = MagicMock()
        mgr._sessions_cache[session.honcho_session_id] = honcho_session

        session.add_message("user", "from alice", author_peer_id="alice")
        session.add_message("user", "from bob", author_peer_id="bob")
        mgr._flush_session(session)

        assert honcho_session.add_peers.call_count == 2

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

    def test_on_turn_start_records_the_author(self):
        provider = self._provider()
        provider.on_turn_start(
            3, "hello", author_id="111222", author_name="Alice", author_is_bot=False
        )
        assert provider._turn_author == {
            "id": "111222",
            "name": "Alice",
            "is_bot": False,
        }

    def test_on_turn_start_without_author_kwargs(self):
        """Callers that never adopted the kwargs must keep working."""
        provider = self._provider()
        provider.on_turn_start(1, "hello")
        assert provider._turn_author == {"id": None, "name": None, "is_bot": False}

    def test_bot_authored_turn_is_flagged(self):
        provider = self._provider()
        provider.on_turn_start(2, "ping", author_id="bot-9", author_is_bot=True)
        assert provider._turn_author["is_bot"] is True

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

    def test_sync_turn_prefers_the_turn_author_keyword(self):
        """The manager passes the author with the turn; the stash is only a fallback."""
        provider = self._provider()
        provider._session_initialized = True
        provider._manager.get_or_create.return_value = MagicMock()
        provider._manager.resolve_author_peer_id.return_value = "alice"

        provider.on_turn_start(1, "hi", author_id="999")
        provider.sync_turn("hi", "hello back", turn_author={"id": "111222", "name": "Alice", "is_bot": False})
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5)

        provider._manager.resolve_author_peer_id.assert_called_once_with("telegram:group1", "111222", "Alice")

    def test_sync_turn_falls_back_to_the_stash(self):
        provider = self._provider()
        provider._session_initialized = True
        provider._manager.get_or_create.return_value = MagicMock()
        provider._manager.resolve_author_peer_id.return_value = None

        provider.on_turn_start(1, "hi", author_id="999", author_name="Nine")
        provider.sync_turn("hi", "hello back")
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5)

        provider._manager.resolve_author_peer_id.assert_called_once_with("telegram:group1", "999", "Nine")

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
