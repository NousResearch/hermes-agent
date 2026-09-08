"""Tests for agent.turn_author: the per-turn author carried from a dispatcher to memory hooks."""

import json

import pytest

from agent.turn_author import (
    TURN_AUTHOR_ENV,
    a2a_key,
    parse_turn_author,
    take_turn_author_from_env,
    turn_author_env,
    turn_author_from_env,
)

FAMILY = "\U0001F468\u200D\U0001F469\u200D\U0001F467"


class TestParseTurnAuthor:
    @pytest.mark.parametrize("raw, expected", [
        ({"id": "  bot:alpha ", "name": "Alpha", "is_bot": True, "extra": 1}, {"id": "bot:alpha", "name": "Alpha", "is_bot": True}),
        (json.dumps({"id": "bot:alpha", "name": "Alpha", "is_bot": 1}), {"id": "bot:alpha", "name": "Alpha", "is_bot": True}),
        ({"name": "Alpha"}, {"id": None, "name": "Alpha", "is_bot": False}),
        ({"id": 7, "name": "Alpha", "is_bot": "yes"}, {"id": None, "name": "Alpha", "is_bot": True}),
        ({"id": "", "name": " Alpha "}, {"id": None, "name": "Alpha", "is_bot": False}),
        ({"id": "bot:\x00al\x1bpha\n", "name": "Al\tpha\r"}, {"id": "bot:alpha", "name": "Alpha", "is_bot": False}),
        ({"id": "bot:alpha", "name": f"{FAMILY} Al\u00a0pha\u00a0"}, {"id": "bot:alpha", "name": f"{FAMILY} Al\u00a0pha", "is_bot": False}),
        ({"id": "x" * 500, "name": "y" * 201}, {"id": "x" * 200, "name": "y" * 200, "is_bot": False}),
        ({"id": "bot:cloud-1/alpha", "name": "Alpha", "is_bot": True}, {"id": "bot:cloud-1/alpha", "name": "Alpha", "is_bot": True}),
        ({"id": "bot:alpha", "name": "Alpha", "is_bot": True, "origin": " cloud-1 "}, {"id": "bot:cloud-1/alpha", "name": "Alpha", "is_bot": True}),
        ({"id": "bot:cloud-1/alpha", "name": "Alpha", "is_bot": True, "origin": "cloud-2"}, {"id": "bot:cloud-1/alpha", "name": "Alpha", "is_bot": True}),
        ({"id": "5551234", "name": "Alpha", "is_bot": True, "origin": "cloud-1"}, {"id": "5551234", "name": "Alpha", "is_bot": True}),
    ], ids=["dict", "json string", "missing id", "non-string id", "empty id", "control characters stripped",
            "format characters and nbsp survive", "oversize fields capped", "connection-qualified id survives",
            "origin qualifies a bare bot id", "origin never requalifies", "origin leaves a platform id alone"])
    def test_fields_are_normalized(self, raw, expected):
        assert parse_turn_author(raw) == expected

    @pytest.mark.parametrize("raw", [
        None, 42, [], ["bot:alpha"], "not json", '"a string"', "[1, 2]", b"\xff",
        {}, {"is_bot": True}, {"id": "", "name": "   "},
    ])
    def test_junk_and_authors_without_id_or_name_return_none(self, raw):
        assert parse_turn_author(raw) is None

    @pytest.mark.parametrize("flag, expected", [
        *((flag, True) for flag in (True, 1, "true", "1", " YES ")),
        *((flag, False) for flag in (False, 0, None, "false", "0", "no", "", "bot", [True], {"a": 1}, 1.0)),
    ])
    def test_bot_flag_accepts_only_booleans_and_truthy_strings(self, flag, expected):
        assert parse_turn_author({"id": "bot:alpha", "is_bot": flag})["is_bot"] is expected


class TestEnvCarrier:
    def test_round_trip_through_env(self):
        author = {"id": "bot:alpha", "name": "Alpha", "is_bot": True}
        env = turn_author_env(author)
        assert set(env) == {TURN_AUTHOR_ENV}
        assert turn_author_from_env(env) == author

    def test_env_json_is_compact(self):
        assert turn_author_env({"id": "a", "is_bot": True})[TURN_AUTHOR_ENV] == '{"id":"a","is_bot":true}'

    @pytest.mark.parametrize("env", [{}, {TURN_AUTHOR_ENV: "{not json"}])
    def test_absent_or_garbage_env_is_none(self, env):
        assert turn_author_from_env(env) is None

    def test_take_removes_the_variable(self):
        author = {"id": "bot:alpha", "name": "Alpha", "is_bot": True}
        env = dict(turn_author_env(author), OTHER="kept")
        assert take_turn_author_from_env(env) == author
        assert env == {"OTHER": "kept"}
        assert take_turn_author_from_env(env) is None


@pytest.mark.parametrize("author, expected", [
    ({"id": "bot:coder", "name": "coder", "is_bot": True}, "a2a:bot:coder"),
    ({"id": "bot:cloud-1/coder", "name": "coder", "is_bot": True}, "a2a:bot:cloud-1/coder"),
    ({"id": "5551234", "name": "SomeBot", "is_bot": True}, "a2a:5551234"),
    ({"id": "111222", "name": "Alice", "is_bot": False}, None),
    ({"id": None, "name": "mystery", "is_bot": True}, None),
    (None, None),
])
def test_a2a_key_names_a_bot_authors_turns(author, expected):
    assert a2a_key(author) == expected
