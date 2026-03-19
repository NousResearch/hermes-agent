"""
Tests for SQL security in insights.py.

Issue #1911: Possible SQL injection via string formatting in execute()

This test verifies that:
1. SQL queries use parameterized inputs for user-controlled values
2. Column lists are constants, not user-controlled
3. The refactored code maintains functionality
"""

import pytest
from unittest.mock import MagicMock, patch


class TestInsightsSQLSecurity:
    """Security tests for InsightsEngine SQL queries."""

    def test_session_cols_is_constant(self):
        """Verify _SESSION_COLS is a hardcoded constant tuple."""
        from agent.insights import InsightsEngine
        
        # Should be a tuple of column names
        assert isinstance(InsightsEngine._SESSION_COLS, tuple)
        
        # Should contain expected columns
        expected_cols = {"id", "source", "model", "started_at", "ended_at"}
        actual_cols = set(InsightsEngine._SESSION_COLS)
        assert expected_cols.issubset(actual_cols)
        
        # Should not be modifiable at runtime
        assert all(isinstance(col, str) for col in InsightsEngine._SESSION_COLS)

    def test_session_select_is_constant(self):
        """Verify _SESSION_SELECT is a pre-built constant string."""
        from agent.insights import InsightsEngine
        
        # Should be a string
        assert isinstance(InsightsEngine._SESSION_SELECT, str)
        
        # Should start with SELECT
        assert InsightsEngine._SESSION_SELECT.startswith("SELECT")
        
        # Should contain all columns
        for col in InsightsEngine._SESSION_COLS:
            assert col in InsightsEngine._SESSION_SELECT

    def test_get_sessions_uses_parameterized_queries(self):
        """Verify _get_sessions uses parameterized queries for user input."""
        from agent.insights import InsightsEngine
        
        # Create mock db object that mimics SessionDB
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor
        
        mock_db = MagicMock()
        mock_db._conn = mock_conn
        
        engine = InsightsEngine(mock_db)
        
        # Call with source filter (user-controlled input)
        engine._get_sessions(cutoff=1234567890.0, source="telegram")
        
        # Verify execute was called
        mock_conn.execute.assert_called()
        
        # Get the call arguments
        call_args = mock_conn.execute.call_args
        query = call_args[0][0]
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get('params', ())
        
        # Query should use ? placeholders, not f-string interpolation
        assert "?" in query
        assert "source = ?" in query
        
        # Parameters should be passed separately
        assert params == (1234567890.0, "telegram")

    def test_get_sessions_without_source_uses_parameterized(self):
        """Verify _get_sessions without source also uses parameterized queries."""
        from agent.insights import InsightsEngine
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor
        
        mock_db = MagicMock()
        mock_db._conn = mock_conn
        
        engine = InsightsEngine(mock_db)
        engine._get_sessions(cutoff=1234567890.0, source=None)
        
        call_args = mock_conn.execute.call_args
        query = call_args[0][0]
        params = call_args[0][1] if len(call_args[0]) > 1 else ()
        
        # Should still use ? placeholder for cutoff
        assert "?" in query
        assert "started_at >= ?" in query
        
        # Should have single parameter
        assert params == (1234567890.0,)

    def test_no_fstring_in_execute(self):
        """Verify no f-strings are used directly in execute() calls.
        
        This test reads the source file directly to avoid inspect.getsource
        issues with method indentation.
        """
        import ast
        from pathlib import Path
        
        # Read the full source file
        source_path = Path(__file__).parent.parent.parent / "agent" / "insights.py"
        source = source_path.read_text()
        
        # Parse the AST
        tree = ast.parse(source)
        
        # Look for JoinedStr (f-string) nodes inside Call nodes where
        # the function is 'execute'
        class FStringInExecuteVisitor(ast.NodeVisitor):
            def __init__(self):
                self.found_fstring_in_execute = False
                
            def visit_Call(self, node):
                # Check if this is an execute call
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                
                if func_name == "execute":
                    # Check if any argument is an f-string
                    for arg in node.args:
                        if isinstance(arg, ast.JoinedStr):
                            self.found_fstring_in_execute = True
                
                self.generic_visit(node)
        
        visitor = FStringInExecuteVisitor()
        visitor.visit(tree)
        
        assert not visitor.found_fstring_in_execute, \
            "insights.py should not use f-strings in execute() calls"


class TestToolUsageSQL:
    """Tests for SQL in _get_tool_usage method."""

    def test_tool_usage_uses_parameterized_queries(self):
        """Verify _get_tool_usage uses parameterized queries."""
        from agent.insights import InsightsEngine
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor
        
        engine = InsightsEngine(mock_conn)
        engine._get_tool_usage(cutoff=1234567890.0, source="discord")
        
        # All execute calls should use ? placeholders
        for call in mock_conn.execute.call_args_list:
            query = call[0][0]
            assert "?" in query
            # Should not have direct string interpolation
            assert "discord" not in query  # Source should be in params, not query


class TestInsightsIntegration:
    """Integration tests for insights with mock database."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database wrapper with expected structure."""
        import sqlite3
        
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        
        # Create minimal schema
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                model TEXT,
                started_at REAL,
                ended_at REAL,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                billing_provider TEXT,
                billing_base_url TEXT,
                billing_mode TEXT,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                cost_status TEXT,
                cost_source TEXT
            )
        """)
        
        # Insert test data
        conn.execute("""
            INSERT INTO sessions (id, source, model, started_at, message_count)
            VALUES ('test-1', 'telegram', 'gpt-4', 1234567890.0, 10)
        """)
        conn.execute("""
            INSERT INTO sessions (id, source, model, started_at, message_count)
            VALUES ('test-2', 'discord', 'claude-3', 1234567891.0, 5)
        """)
        conn.commit()
        
        # Create a wrapper that mimics SessionDB
        class MockSessionDB:
            def __init__(self, connection):
                self._conn = connection
        
        return MockSessionDB(conn)

    def test_get_sessions_returns_correct_data(self, mock_db):
        """Test that queries return correct results."""
        from agent.insights import InsightsEngine
        
        engine = InsightsEngine(mock_db)
        
        # Get all sessions
        all_sessions = engine._get_sessions(cutoff=0.0)
        assert len(all_sessions) == 2
        
        # Get filtered sessions
        telegram_sessions = engine._get_sessions(cutoff=0.0, source="telegram")
        assert len(telegram_sessions) == 1
        assert telegram_sessions[0]["source"] == "telegram"
        
        discord_sessions = engine._get_sessions(cutoff=0.0, source="discord")
        assert len(discord_sessions) == 1
        assert discord_sessions[0]["source"] == "discord"

    def test_sql_injection_prevented(self, mock_db):
        """Test that SQL injection attempts are prevented."""
        from agent.insights import InsightsEngine
        
        engine = InsightsEngine(mock_db)
        
        # Attempt SQL injection via source parameter
        malicious_input = "'; DROP TABLE sessions; --"
        
        # Should not raise error or affect database
        result = engine._get_sessions(cutoff=0.0, source=malicious_input)
        
        # Should return empty (no matching source)
        assert result == []
        
        # Table should still exist
        cursor = mock_db._conn.execute("SELECT COUNT(*) FROM sessions")
        count = cursor.fetchone()[0]
        assert count == 2  # Original data intact
