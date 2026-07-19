"""Tests for agent_turn_cleanup_runtime_service."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.agent_turn_cleanup_runtime_service import cleanup_gateway_agent_turn


def test_cleanup_consumes_sidecar_and_clears_env():
    runner = SimpleNamespace(
        _consume_pending_turn_sidecar_notes=MagicMock(return_value=["n1"]),
        _clear_session_env=MagicMock(),
    )
    logger = MagicMock()
    cleanup_gateway_agent_turn(
        runner=runner,
        session_key="sk",
        session_env_tokens={"x": 1},
        logger=logger,
    )
    runner._consume_pending_turn_sidecar_notes.assert_called_once_with("sk")
    runner._clear_session_env.assert_called_once_with({"x": 1})
    logger.debug.assert_called_once()


def test_cleanup_skips_sidecar_when_no_session_key():
    runner = SimpleNamespace(
        _consume_pending_turn_sidecar_notes=MagicMock(),
        _clear_session_env=MagicMock(),
    )
    cleanup_gateway_agent_turn(
        runner=runner,
        session_key="",
        session_env_tokens=None,
        logger=MagicMock(),
    )
    runner._consume_pending_turn_sidecar_notes.assert_not_called()
    runner._clear_session_env.assert_called_once_with(None)
