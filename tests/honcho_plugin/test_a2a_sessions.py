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
from tools.bot_relay import delivery_turn_author

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

        provider._manager.get_or_create.assert_called_once_with(provider._a2a_session_key({"id": "bot:coder", "is_bot": True}), user_peer_id="coder")
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


    def test_two_bots_get_two_sessions(self):
        provider = _provider()
        provider._manager.resolve_author_peer_id.side_effect = lambda key, author_id, name=None, **kw: author_id[4:]

        _sync(provider, turn_author=BOT_AUTHOR)
        _sync(provider, turn_author={"id": "bot:writer", "name": "writer", "is_bot": True})

        keys = [c[0][0] for c in provider._manager.get_or_create.call_args_list]
        assert keys == [provider._a2a_session_key({"id": "bot:coder", "is_bot": True}), provider._a2a_session_key({"id": "bot:writer", "is_bot": True})]


    def test_bot_colliding_with_this_agents_ai_peer_is_skipped(self):
        """One peer cannot be both sides of a session."""
        provider = _provider()
        provider._manager.resolve_author_peer_id.return_value = "hermes"
        provider._manager.assistant_peer_id.return_value = "hermes"

        _sync(provider, turn_author=BOT_AUTHOR)

        provider._manager.get_or_create.assert_not_called()


    def test_two_recipients_sharing_a_session_key_get_different_sessions(self):
        """Two profiles with one workspace and one session key must not merge a sender's DMs."""
        ivy, holly = _provider(), _provider()
        ivy._config = HonchoClientConfig(workspace_id="shared", ai_peer="ivy")
        holly._config = HonchoClientConfig(workspace_id="shared", ai_peer="holly")

        assert ivy._a2a_session_key(BOT_AUTHOR) != holly._a2a_session_key(BOT_AUTHOR)
        assert ivy._a2a_session_key(BOT_AUTHOR).startswith("Bot-Chat:a2a:ivy:bot-coder-")

    def test_same_named_senders_on_two_connections_get_two_sessions(self):
        """A relayed envelope's author carries the sender's connection, and the a2a key keeps it."""
        provider = _provider()
        east = delivery_turn_author("coder", "coder", "east")
        west = delivery_turn_author("coder", "coder", "west")

        assert east["id"] == "bot:east/coder"
        assert provider._a2a_session_key(east) != provider._a2a_session_key(west)
        assert provider._a2a_session_key(east).startswith("Bot-Chat:a2a:hermes-assistant:bot-east-coder-")


class TestToolWritesDuringBotTurn:
    def _tools_provider(self, author: dict) -> HonchoMemoryProvider:
        provider = _provider()
        provider._turn_author = dict(author)
        provider._manager.create_conclusion.return_value = True
        provider._manager.delete_conclusion.return_value = True
        provider._manager.set_peer_card.return_value = ["fact"]
        provider._manager.list_conclusions.return_value = []
        return provider

    def test_conclude_and_delete_are_refused(self):
        provider = self._tools_provider(BOT_AUTHOR)
        assert "error" in json.loads(provider._tool_conclude({"conclusion": "likes tea"}))
        assert "error" in json.loads(provider._tool_conclude({"delete_id": "c1"}))
        provider._manager.create_conclusion.assert_not_called()
        provider._manager.delete_conclusion.assert_not_called()


    def test_profile_card_write_is_refused_but_read_works(self):
        provider = self._tools_provider(BOT_AUTHOR)
        assert "error" in json.loads(provider._tool_profile({"card": ["fact"]}))
        provider._manager.set_peer_card.assert_not_called()
        provider._manager.get_peer_card.return_value = ["fact"]
        assert json.loads(provider._tool_profile({})) == {"result": ["fact"]}

    def test_memory_mirror_is_skipped(self):
        provider = self._tools_provider(BOT_AUTHOR)
        provider.on_memory_write("add", "user", "likes tea")
        assert provider._memwrite_thread is None
        provider._manager.create_conclusion.assert_not_called()


class TestFlagOff:
    def test_bot_turn_is_skipped(self):
        provider = _provider(a2a_sessions=False)

        _sync(provider, turn_author=BOT_AUTHOR)

        provider._manager.get_or_create.assert_not_called()
        provider._manager.resolve_author_peer_id.assert_not_called()


class TestHumanTurnUnchanged:
    def test_human_turn_writes_into_the_session_under_the_author(self):
        provider = _provider()
        provider._manager.resolve_author_peer_id.return_value = "alice"

        _sync(provider, turn_author=HUMAN_AUTHOR)

        provider._manager.get_or_create.assert_called_once_with("Bot-Chat")
        session = provider._manager.get_or_create.return_value
        assert session.add_message.call_args_list[0][1]["author_peer_id"] == "alice"


class TestManagerUserPeerOverride:
    def test_get_or_create_uses_the_override_as_user_peer(self):
        mgr = HonchoSessionManager(honcho=MagicMock(), config=HonchoClientConfig(api_key="k", peer_name="eri", ai_peer="hermes"),
                                   runtime_user_peer_name="7654321")
        mgr._get_or_create_peer = MagicMock(side_effect=lambda pid: MagicMock(name=f"peer:{pid}"))
        mgr._get_or_create_honcho_session = MagicMock(return_value=(MagicMock(), []))

        session = mgr.get_or_create("Bot-Chat:a2a:bot-coder-0123abcd", user_peer_id="coder")

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
