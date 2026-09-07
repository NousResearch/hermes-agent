"""CLI bot-chain transcript persistence (#100758).

A ``$Bot`` chain turn never runs the CLI session's own agent, so the lazy
session-row creation in ``run_agent._ensure_db_session`` never fires on that
path. ``_persist_bot_chain_exchange`` must therefore create the session row
itself; on a fresh session (chain as the very first turn) a bare messages
insert violates the messages→sessions foreign key and the exchange is
silently lost from session history and /retry context.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def cli_mod():
    import cli

    return cli


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


def _stub_cli(cli_mod, db):
    stub = object.__new__(cli_mod.HermesCLI)
    stub._session_db = db
    stub.session_id = "sess-cli-chain"
    stub.conversation_history = []
    stub.model = "deepseek-v4-flash"
    stub.max_turns = 10
    stub.reasoning_config = None
    return stub


class TestPersistBotChainExchange:
    def test_fresh_session_persists_exchange(self, cli_mod, db):
        """First turn of a fresh session: no pre-existing session row."""
        stub = _stub_cli(cli_mod, db)
        assert db.get_session("sess-cli-chain") is None

        stub._persist_bot_chain_exchange("$alpha what is 2+2?", "[alpha] 4")

        session = db.get_session("sess-cli-chain")
        assert session is not None
        rows = db.get_messages("sess-cli-chain")
        assert [r["role"] for r in rows] == ["user", "assistant"]
        assert rows[0]["content"] == "$alpha what is 2+2?"
        assert rows[1]["content"] == "[alpha] 4"
        assert [m["role"] for m in stub.conversation_history] == [
            "user",
            "assistant",
        ]

    def test_repeat_persist_is_idempotent(self, cli_mod, db):
        """The session-row ensure must not fail or duplicate on later chains."""
        stub = _stub_cli(cli_mod, db)
        stub._persist_bot_chain_exchange("$alpha one", "[alpha] 1")
        stub._persist_bot_chain_exchange("$alpha two", "[alpha] 2")

        rows = db.get_messages("sess-cli-chain")
        assert [r["content"] for r in rows] == [
            "$alpha one",
            "[alpha] 1",
            "$alpha two",
            "[alpha] 2",
        ]

    def test_no_session_db_still_updates_memory(self, cli_mod):
        stub = _stub_cli(cli_mod, None)

        stub._persist_bot_chain_exchange("$alpha hi", "[alpha] hi")

        assert [m["content"] for m in stub.conversation_history] == [
            "$alpha hi",
            "[alpha] hi",
        ]
