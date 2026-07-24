"""Tests for tool_dispatch_helpers — focuses on logging behavior."""

import logging

import pytest

from agent.tool_dispatch_helpers import (
    _extract_file_mutation_targets,
    _is_untrusted_tool,
    _plan_tool_batch_segments,
    logger,
)


def test_uses_module_scoped_logger_not_root():
    """Verify that tool_dispatch_helpers uses module-scoped logger (__name__)
    instead of root logging, preventing log message duplication and
    namespace pollution."""
    assert logger.name == "agent.tool_dispatch_helpers"
    assert logger.name != "root"
    assert logger is not logging.getLogger()


def test_module_logger_has_no_handlers_by_default():
    """Module-scoped logger should not have its own handlers (relies on
    root handler propagation). This is the expected pattern for library
    loggers."""
    assert logger.handlers == []


def test_plan_tool_batch_segments_logs_to_module_logger(caplog):
    """Verify that _plan_tool_batch_segments emits logs through the
    module-scoped logger, not root logging."""
    caplog.set_level(logging.DEBUG, logger="agent.tool_dispatch_helpers")

    # Create a minimal tool call that triggers the logging path
    from types import SimpleNamespace

    # A tool call with non-dict args (triggers the debug log)
    bad_tc = SimpleNamespace(
        type="function",
        function=SimpleNamespace(
            name="read_file",
            arguments='"not a dict"',  # string, not dict — triggers debug log
        ),
    )

    _plan_tool_batch_segments([bad_tc])

    # Verify the log was emitted to the module logger
    assert any(
        "Non-dict args" in record.message and record.name == "agent.tool_dispatch_helpers"
        for record in caplog.records
    ), f"Expected module logger record, got: {[r.name for r in caplog.records]}"


def test_plan_tool_batch_segments_no_root_logger_calls(caplog):
    """Verify that _plan_tool_batch_segments does NOT emit to root logger."""
    caplog.set_level(logging.DEBUG, logger="root")

    from types import SimpleNamespace

    bad_tc = SimpleNamespace(
        type="function",
        function=SimpleNamespace(
            name="read_file",
            arguments='"not a dict"',
        ),
    )

    _plan_tool_batch_segments([bad_tc])

    # Root logger should NOT have received the debug message
    root_records = [r for r in caplog.records if r.name == "root"]
    assert len(root_records) == 0, (
        f"Root logger should not receive debug calls, got: {root_records}"
    )
