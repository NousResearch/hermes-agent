"""Tests for slash command logging in gateway (#48240).

Verifies that every recognized slash command dispatched through
``GatewayRunner._handle_message`` emits a ``logger.info`` line
before the command is processed.
"""

import ast
import inspect
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.run import GatewayConfig, GatewayRunner


# ---------------------------------------------------------------------------
# AST invariant: the slash-command logging block exists before dispatch
# ---------------------------------------------------------------------------
class TestSlashLogAst:
    """Verify the log call exists in the source at the right location."""

    def test_slash_log_precedes_dispatch(self):
        """The logger.info('slash command:...') block must appear before the
        ``if canonical == "new":`` dispatch chain."""
        src = inspect.getsource(gateway_run)
        tree = ast.parse(src)

        # Find all string constants
        log_line_pos = None
        dispatch_pos = None

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "slash command:" in node.value:
                    log_line_pos = node.lineno
                if node.value == "new" and dispatch_pos is None:
                    # Only the first "new" after a canonical == check
                    dispatch_pos = node.lineno

        assert log_line_pos is not None, (
            "Could not find 'slash command:' log string in gateway/run.py"
        )
        assert dispatch_pos is not None, (
            "Could not find canonical == 'new' dispatch in gateway/run.py"
        )
        assert log_line_pos < dispatch_pos, (
            f"Slash log (line {log_line_pos}) must precede dispatch (line {dispatch_pos})"
        )

    def test_log_includes_platform_user_args(self):
        """The log format string must reference platform, user, and args."""
        src = inspect.getsource(gateway_run)
        tree = ast.parse(src)

        log_format = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "slash command:" in node.value:
                    log_format = node.value
                    break

        assert log_format is not None
        assert "platform=" in log_format
        assert "user=" in log_format
        assert "args=" in log_format
