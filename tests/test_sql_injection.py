"""Tests that verify SQL injection mitigations in insights and state modules."""

import re

from agent.insights import InsightsEngine


def test_session_cols_no_injection_chars():
    """_SESSION_COLS must not contain SQL injection vectors."""
    cols = InsightsEngine._SESSION_COLS
    assert ";" not in cols
    assert "--" not in cols
    assert "'" not in cols
    assert "DROP" not in cols.upper()


def test_get_sessions_all_query_is_parameterized():
    """_GET_SESSIONS_ALL must use a ? placeholder for the cutoff value."""
    query = InsightsEngine._GET_SESSIONS_ALL
    assert "?" in query
    assert "started_at >= ?" in query
    # Must not embed any runtime-variable content via brace interpolation
    assert "{" not in query


def test_get_sessions_with_source_query_is_parameterized():
    """_GET_SESSIONS_WITH_SOURCE must use ? placeholders for both parameters."""
    query = InsightsEngine._GET_SESSIONS_WITH_SOURCE
    assert query.count("?") == 2
    assert "started_at >= ?" in query
    assert "source = ?" in query
    assert "{" not in query


def test_session_col_names_are_safe_identifiers():
    """Every column name listed in _SESSION_COLS must be a simple identifier."""
    cols = InsightsEngine._SESSION_COLS
    identifiers = [c.strip() for c in cols.split(",")]
    safe_identifier = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    for col in identifiers:
        assert safe_identifier.match(col), (
            f"Column name {col!r} is not a safe SQL identifier"
        )
