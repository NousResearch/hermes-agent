"""Tests for public per-turn session-key setters in gateway.session_context.

These wrappers mirror tools/approval.py's set_current_session_key /
reset_current_session_key pattern but are deliberately named distinctly
(``turn_session_key``) because they target a different ContextVar
(``_SESSION_KEY`` in gateway.session_context) than the approval-session
ContextVar (``_approval_session_key`` in tools.approval).
"""

from gateway.session_context import (
    get_session_env,
    set_current_turn_session_key,
    reset_current_turn_session_key,
)


def test_set_and_reset_round_trip(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_KEY", "env-session")
    previous = get_session_env("HERMES_SESSION_KEY")
    token = set_current_turn_session_key("session-abc")
    try:
        assert get_session_env("HERMES_SESSION_KEY") == "session-abc"
    finally:
        reset_current_turn_session_key(token)
    assert get_session_env("HERMES_SESSION_KEY") == previous


def test_nested_set_reset():
    outer = set_current_turn_session_key("outer")
    try:
        assert get_session_env("HERMES_SESSION_KEY") == "outer"
        inner = set_current_turn_session_key("inner")
        try:
            assert get_session_env("HERMES_SESSION_KEY") == "inner"
        finally:
            reset_current_turn_session_key(inner)
        assert get_session_env("HERMES_SESSION_KEY") == "outer"
    finally:
        reset_current_turn_session_key(outer)


def test_empty_string_normalized():
    token = set_current_turn_session_key("")
    try:
        assert get_session_env("HERMES_SESSION_KEY") == ""
    finally:
        reset_current_turn_session_key(token)


def test_none_normalized_to_empty():
    # set_current_turn_session_key(None) should not raise; treat as "".
    token = set_current_turn_session_key(None)
    try:
        assert get_session_env("HERMES_SESSION_KEY") == ""
    finally:
        reset_current_turn_session_key(token)
