"""Bot-authored DMs write into their own Honcho session.

The turn author carries ``is_bot`` for a relayed DM. With ``a2aSessions`` on
(the default) ``sync_turn`` writes the whole turn into ``<session>:a2a:<bot>``
with the sender bot as that session's user peer. The human's session never
receives a bot's turn: with the flag off the turn is skipped.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSessionManager

BOT_AUTHOR = {"id": "bot:coder", "name": "coder", "is_bot": True}
HUMAN_AUTHOR = {"id": "111222", "name": "Alice", "is_bot": False}


def _provider(a2a_sessions: bool = True) -> HonchoMemoryProvider:
    provider = HonchoMemoryProvider()
    provider._session_key = "Bot-Chat"
    provider._manager = MagicMock()
    provider._manager.get_or_create.return_value = MagicMock()
    provider._cron_skipped = False
    provider._session_initialized = True
    provider._config = SimpleNamespace(message_max_chars=25000, a2a_sessions=a2a_sessions)
    return provider


def _sync(provider: HonchoMemoryProvider, **kwargs) -> None:
    provider.sync_turn("Message from coder: hi", "hello coder", **kwargs)
    if provider._sync_thread:
        provider._sync_thread.join(timeout=5)


class TestA2aRouting:
    def test_bot_turn_lands_in_its_own_session(self):
        provider = _provider()
        provider._manager.resolve_author_peer_id.return_value = "coder"

        _sync(provider, turn_author=BOT_AUTHOR)

        provider._manager.get_or_create.assert_called_once_with("Bot-Chat:a2a:bot-coder", user_peer_id="coder")
        session = provider._manager.get_or_create.return_value
        roles = [c[0][0] for c in session.add_message.call_args_list]
        assert roles == ["user", "assistant"]
        # The bot is the session's user peer; no per-message author is attached.
        assert session.add_message.call_args_list[0][1]["author_peer_id"] is None

    def test_bot_turn_never_touches_the_human_session(self):
        provider = _provider()
        provider._manager.resolve_author_peer_id.return_value = "coder"

        _sync(provider, turn_author=BOT_AUTHOR)

        keys = [c[0][0] for c in provider._manager.get_or_create.call_args_list]
        assert "Bot-Chat" not in keys

    def test_session_key_is_stable_across_turns(self):
        provider = _provider()
        provider._manager.resolve_author_peer_id.return_value = "coder"

        _sync(provider, turn_author=BOT_AUTHOR)
        _sync(provider, turn_author=BOT_AUTHOR)

        keys = {c[0][0] for c in provider._manager.get_or_create.call_args_list}
        assert keys == {"Bot-Chat:a2a:bot-coder"}

    def test_two_bots_get_two_sessions(self):
        provider = _provider()
        provider._manager.resolve_author_peer_id.side_effect = lambda key, author_id, name=None: author_id[4:]

        _sync(provider, turn_author=BOT_AUTHOR)
        _sync(provider, turn_author={"id": "bot:writer", "name": "writer", "is_bot": True})

        keys = [c[0][0] for c in provider._manager.get_or_create.call_args_list]
        assert keys == ["Bot-Chat:a2a:bot-coder", "Bot-Chat:a2a:bot-writer"]

    def test_bot_author_without_an_id_is_skipped(self):
        provider = _provider()

        _sync(provider, turn_author={"id": None, "name": "mystery", "is_bot": True})

        provider._manager.get_or_create.assert_not_called()

    def test_unresolvable_bot_peer_keeps_the_default_user_peer(self):
        """When no author peer resolves, the a2a session still opens under the resolved peer."""
        provider = _provider()
        provider._manager.resolve_author_peer_id.return_value = None

        _sync(provider, turn_author=BOT_AUTHOR)

        provider._manager.get_or_create.assert_called_once_with("Bot-Chat:a2a:bot-coder")

    def test_long_session_key_stays_within_the_honcho_limit(self):
        provider = _provider()
        provider._session_key = "x" * 95
        provider._manager.resolve_author_peer_id.return_value = "coder"

        _sync(provider, turn_author=BOT_AUTHOR)

        key = provider._manager.get_or_create.call_args[0][0]
        assert len(key) <= 100
        assert key == provider._a2a_session_key("bot:coder")


class TestFlagOff:
    def test_bot_turn_is_skipped(self):
        provider = _provider(a2a_sessions=False)

        _sync(provider, turn_author=BOT_AUTHOR)

        provider._manager.get_or_create.assert_not_called()
        provider._manager.resolve_author_peer_id.assert_not_called()

    def test_human_turn_still_writes(self):
        provider = _provider(a2a_sessions=False)
        provider._manager.resolve_author_peer_id.return_value = "alice"

        _sync(provider, turn_author=HUMAN_AUTHOR)

        provider._manager.get_or_create.assert_called_once_with("Bot-Chat")


class TestHumanTurnUnchanged:
    def test_human_turn_writes_into_the_session_under_the_author(self):
        provider = _provider()
        provider._manager.resolve_author_peer_id.return_value = "alice"

        _sync(provider, turn_author=HUMAN_AUTHOR)

        provider._manager.get_or_create.assert_called_once_with("Bot-Chat")
        session = provider._manager.get_or_create.return_value
        assert session.add_message.call_args_list[0][1]["author_peer_id"] == "alice"

    def test_human_turn_without_author_keeps_the_session_peer(self):
        provider = _provider()
        provider._manager.resolve_author_peer_id.return_value = None

        _sync(provider)

        provider._manager.get_or_create.assert_called_once_with("Bot-Chat")


class TestManagerUserPeerOverride:
    def test_get_or_create_uses_the_override_as_user_peer(self):
        mgr = HonchoSessionManager(honcho=MagicMock(), config=HonchoClientConfig(api_key="k", peer_name="eri", ai_peer="hermes"),
                                   runtime_user_peer_name="7654321")
        mgr._get_or_create_peer = MagicMock(side_effect=lambda pid: MagicMock(name=f"peer:{pid}"))
        mgr._get_or_create_honcho_session = MagicMock(return_value=(MagicMock(), []))

        session = mgr.get_or_create("Bot-Chat:a2a:bot-coder", user_peer_id="coder")

        assert session.user_peer_id == "coder"
        assert session.assistant_peer_id == "hermes"
        joined = [c[0][0] for c in mgr._get_or_create_peer.call_args_list]
        assert "7654321" not in joined


class TestConfigFlag:
    def _config(self, tmp_path, monkeypatch, raw: dict) -> HonchoClientConfig:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        path = tmp_path / "honcho.json"
        path.write_text(json.dumps({"apiKey": "k", **raw}))
        return HonchoClientConfig.from_global_config(config_path=path)

    def test_defaults_on(self, tmp_path, monkeypatch):
        assert self._config(tmp_path, monkeypatch, {}).a2a_sessions is True

    def test_root_flag(self, tmp_path, monkeypatch):
        assert self._config(tmp_path, monkeypatch, {"a2aSessions": False}).a2a_sessions is False

    def test_host_block_wins_over_root(self, tmp_path, monkeypatch):
        cfg = self._config(tmp_path, monkeypatch, {"a2aSessions": True, "hosts": {"hermes": {"a2aSessions": False}}})
        assert cfg.a2a_sessions is False

    def test_root_applies_when_host_block_is_silent(self, tmp_path, monkeypatch):
        cfg = self._config(tmp_path, monkeypatch, {"a2aSessions": False, "hosts": {"hermes": {"peerName": "eri"}}})
        assert cfg.a2a_sessions is False
