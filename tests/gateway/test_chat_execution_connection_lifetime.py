"""Execution-journal handles close deterministically, including failed writes."""
import sqlite3

import pytest

from gateway.chat_execution_recovery import ChatExecutionStore


@pytest.mark.parametrize("fail", [False, True])
def test_connection_closed_after_context_exit(tmp_path, fail):
    store = ChatExecutionStore(tmp_path)
    connection = None
    try:
        try:
            with store._db() as connection:
                connection.execute("SELECT 1")
                if fail:
                    raise ValueError("injected body failure")
        except ValueError:
            assert fail
        assert connection is not None
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    finally:
        if connection is not None:
            connection.close()
