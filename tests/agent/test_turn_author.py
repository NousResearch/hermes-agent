"""Tests for agent.turn_author: the per-turn author carried from a dispatcher to memory hooks."""

import json

import pytest

from agent.turn_author import (
    TURN_AUTHOR_ENV,
    a2a_key,
    parse_turn_author,
    turn_author_env,
    turn_author_from_env,
)


class TestParseTurnAuthor:
    def test_dict_is_normalized(self):
        out = parse_turn_author({"id": "  bot:alpha ", "name": "Alpha", "is_bot": True, "extra": 1})
        assert out == {"id": "bot:alpha", "name": "Alpha", "is_bot": True}

    def test_json_string_is_parsed(self):
        raw = json.dumps({"id": "bot:alpha", "name": "Alpha", "is_bot": 1})
        assert parse_turn_author(raw) == {"id": "bot:alpha", "name": "Alpha", "is_bot": True}

    def test_missing_fields_default(self):
        assert parse_turn_author({}) == {"id": None, "name": None, "is_bot": False}

    @pytest.mark.parametrize("raw", [None, 42, [], ["bot:alpha"], "not json", '"a string"', "[1, 2]", b"\xff"])
    def test_junk_returns_none(self, raw):
        assert parse_turn_author(raw) is None

    def test_non_string_fields_become_none(self):
        assert parse_turn_author({"id": 7, "name": ["x"], "is_bot": "yes"}) == {
            "id": None, "name": None, "is_bot": True,
        }

    def test_empty_and_whitespace_become_none(self):
        assert parse_turn_author({"id": "", "name": "   "}) == {"id": None, "name": None, "is_bot": False}

    def test_control_characters_are_stripped(self):
        out = parse_turn_author({"id": "bot:\x00al\x1bpha\n", "name": "Al\tpha\r"})
        assert out["id"] == "bot:alpha"
        assert out["name"] == "Alpha"

    def test_oversize_fields_are_capped(self):
        out = parse_turn_author({"id": "x" * 500, "name": "y" * 201})
        assert len(out["id"]) == 200
        assert len(out["name"]) == 200


class TestEnvCarrier:
    def test_round_trip_through_env(self):
        author = {"id": "bot:alpha", "name": "Alpha", "is_bot": True}
        env = turn_author_env(author)
        assert set(env) == {TURN_AUTHOR_ENV}
        assert turn_author_from_env(env) == author

    def test_env_json_is_compact(self):
        assert turn_author_env({"id": "a", "is_bot": True})[TURN_AUTHOR_ENV] == '{"id":"a","is_bot":true}'

    def test_absent_env_is_none(self):
        assert turn_author_from_env({}) is None

    def test_garbage_env_is_none(self):
        assert turn_author_from_env({TURN_AUTHOR_ENV: "{not json"}) is None


@pytest.mark.parametrize("author, expected", [
    ({"id": "bot:coder", "name": "coder", "is_bot": True}, "a2a:bot:coder"),
    ({"id": "5551234", "name": "SomeBot", "is_bot": True}, "a2a:5551234"),
    ({"id": "111222", "name": "Alice", "is_bot": False}, None),
    ({"id": None, "name": "mystery", "is_bot": True}, None),
    (None, None),
])
def test_a2a_key_names_a_bot_authors_turns(author, expected):
    assert a2a_key(author) == expected
