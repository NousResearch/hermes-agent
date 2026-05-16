from typing import Any
import pytest
from unittest.mock import patch

from environments.tool_context import _run_tool_in_thread


@pytest.mark.asyncio
async def test_run_tool_in_thread_async() -> None:
    """Test that _run_tool_in_thread uses a thread pool when an event loop is running."""
    with patch("environments.tool_context.handle_function_call") as mock_handle:
        mock_handle.return_value = "async_result"
        result = _run_tool_in_thread("test_tool", {"arg": "val"}, "test_task")
        assert result == "async_result"
        mock_handle.assert_called_once_with("test_tool", {"arg": "val"}, "test_task")


def test_run_tool_in_thread_sync() -> None:
    """Test that _run_tool_in_thread calls the function directly when no event loop is running."""
    with patch("environments.tool_context.handle_function_call") as mock_handle:
        mock_handle.return_value = "sync_result"
        result = _run_tool_in_thread("test_tool", {"arg": "val"}, "test_task")
        assert result == "sync_result"
        mock_handle.assert_called_once_with("test_tool", {"arg": "val"}, "test_task")


@pytest.mark.asyncio
async def test_run_tool_in_thread_timeout() -> None:
    """Test that a timeout is raised if the tool takes too long."""

    def slow_handle(*args: Any, **kwargs: Any) -> str:
        import time

        time.sleep(0.5)
        return "slow_result"

    with patch(
        "environments.tool_context.handle_function_call", side_effect=slow_handle
    ):
        # We patch Future.result to immediately throw TimeoutError to simulate timeout
        with patch(
            "concurrent.futures.Future.result", side_effect=TimeoutError("Timed out")
        ):
            with pytest.raises(TimeoutError):
                _run_tool_in_thread("test_tool", {"arg": "val"}, "test_task")


@pytest.mark.asyncio
async def test_run_tool_in_thread_exception() -> None:
    """Test that exceptions from the tool are propagated correctly in async context."""
    with patch(
        "environments.tool_context.handle_function_call",
        side_effect=ValueError("Test error"),
    ):
        with pytest.raises(ValueError, match="Test error"):
            _run_tool_in_thread("test_tool", {"arg": "val"}, "test_task")


def test_run_tool_in_thread_sync_exception() -> None:
    """Test that exceptions from the tool are propagated correctly in sync context."""
    with patch(
        "environments.tool_context.handle_function_call",
        side_effect=ValueError("Test sync error"),
    ):
        with pytest.raises(ValueError, match="Test sync error"):
            _run_tool_in_thread("test_tool", {"arg": "val"}, "test_task")
