"""Clamp hermes sessions browse --limit before SQLite fetch."""

from hermes_cli.sessions_cmd import clamp_sessions_browse_limit


def test_clamp_sessions_browse_limit_default():
    assert clamp_sessions_browse_limit(None) == 500
    assert clamp_sessions_browse_limit("") == 500
    assert clamp_sessions_browse_limit("nope") == 500


def test_clamp_sessions_browse_limit_floors_zero_and_negative():
    assert clamp_sessions_browse_limit(0) == 1
    assert clamp_sessions_browse_limit(-5) == 1


def test_clamp_sessions_browse_limit_caps_excessive():
    assert clamp_sessions_browse_limit(10_000_000) == 2000
    assert clamp_sessions_browse_limit(42) == 42
    assert clamp_sessions_browse_limit(500) == 500
