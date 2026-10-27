"""Tests for agent.turn_author: the per-turn author carried from a dispatcher to memory hooks."""


import pytest

from agent.turn_author import (
    TURN_AUTHOR_ENV,
    parse_turn_author,
    take_turn_author_from_env,
    turn_author_env,
    turn_author_from_env,
)

FAMILY = "\U0001F468\u200D\U0001F469\u200D\U0001F467"


class TestParseTurnAuthor:

    @pytest.mark.parametrize("raw", [
        None, 42, [], ["bot:alpha"], "not json", '"a string"', "[1, 2]", b"\xff",
        {}, {"is_bot": True}, {"id": "", "name": "   "},
    ])
    def test_junk_and_authors_without_id_or_name_return_none(self, raw):
        assert parse_turn_author(raw) is None


class TestEnvCarrier:
    def test_round_trip_through_env(self):
        author = {"id": "bot:alpha", "name": "Alpha", "is_bot": True}
        env = turn_author_env(author)
        assert set(env) == {TURN_AUTHOR_ENV}
        assert turn_author_from_env(env) == author


    def test_take_removes_the_variable(self):
        author = {"id": "bot:alpha", "name": "Alpha", "is_bot": True}
        env = dict(turn_author_env(author), OTHER="kept")
        assert take_turn_author_from_env(env) == author
        assert env == {"OTHER": "kept"}
        assert take_turn_author_from_env(env) is None
