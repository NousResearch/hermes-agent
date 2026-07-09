"""Gateway /resume workspace guard integration tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.slash_commands import GatewaySlashCommandsMixin


class _ResumeHarness(GatewaySlashCommandsMixin):
    def __init__(self):
        self._session_db = AsyncMock()
        self.session_store = MagicMock()
        self.session_store.get_or_create_session.return_value = SimpleNamespace(session_id="current-session")
        self.session_store.switch_session.return_value = SimpleNamespace(session_id="target-session")
        self.session_store.load_transcript.return_value = [{"role": "user", "content": "hello"}]

        self._resume_target_allowed = AsyncMock(return_value=True)
        self._session_key_for_source = MagicMock(return_value="telegram:user:chat")
        self._gateway_session_origin_for_id = MagicMock(return_value=None)
        self._same_matrix_room = MagicMock(return_value=False)
        self._release_running_agent_state = MagicMock()
        self._clear_session_boundary_security_state = MagicMock()
        self._set_session_reasoning_override = MagicMock()
        self._evict_cached_agent = MagicMock()
        self._session_model_overrides = {}
        self._pending_model_notes = {}
        self._last_resolved_model = {}


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )


def _event(args: str):
    event = MagicMock()
    event.source = _source()
    event.get_command_args.return_value = args
    return event


def _db_for_target(row: dict):
    db = AsyncMock()
    # First get_session(name) resolves the direct session id. Second
    # get_session(target_id) is the workspace-guard lookup after
    # resolve_resume_session_id().
    db.get_session = AsyncMock(side_effect=[{"id": "target-session"}, row])
    db.resolve_resume_session_id = AsyncMock(return_value="target-session")
    db.resolve_session_by_title = AsyncMock(return_value=None)
    db.get_session_title = AsyncMock(return_value="Target Session")
    return db


@pytest.mark.asyncio
async def test_gateway_resume_handler_blocks_cross_workspace_before_switch():
    harness = _ResumeHarness()
    harness._session_db = _db_for_target({
        "id": "target-session",
        "git_repo_root": "/workspace/other",
    })

    with patch("os.getcwd", return_value="/workspace/current"):
        response = await GatewaySlashCommandsMixin._handle_resume_command(
            harness, _event("target-session")
        )

    assert "Cross-workspace resume blocked" in response
    harness.session_store.switch_session.assert_not_called()
    harness._release_running_agent_state.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_resume_handler_allows_same_workspace_and_switches():
    harness = _ResumeHarness()
    harness._session_db = _db_for_target({
        "id": "target-session",
        "git_repo_root": "/workspace/current",
    })

    with patch("os.getcwd", return_value="/workspace/current"):
        response = await GatewaySlashCommandsMixin._handle_resume_command(
            harness, _event("target-session")
        )

    assert "Target Session" in response
    harness.session_store.switch_session.assert_called_once_with(
        "telegram:user:chat", "target-session"
    )


@pytest.mark.asyncio
async def test_gateway_resume_handler_allows_true_legacy_session_with_warning(caplog):
    harness = _ResumeHarness()
    harness._session_db = _db_for_target({
        "id": "target-session",
        "git_repo_root": None,
        "cwd": None,
    })

    with patch("os.getcwd", return_value="/workspace/current"), caplog.at_level("DEBUG"):
        response = await GatewaySlashCommandsMixin._handle_resume_command(
            harness, _event("target-session")
        )

    assert "Target Session" in response
    harness.session_store.switch_session.assert_called_once()
    assert "Workspace guard warning" in caplog.text


@pytest.mark.asyncio
async def test_gateway_resume_handler_blocks_legacy_cwd_mismatch_before_switch():
    harness = _ResumeHarness()
    harness._session_db = _db_for_target({
        "id": "target-session",
        "git_repo_root": None,
        "cwd": "/workspace/other",
    })

    with patch("os.getcwd", return_value="/workspace/current"):
        response = await GatewaySlashCommandsMixin._handle_resume_command(
            harness, _event("target-session")
        )

    assert "Cross-workspace resume blocked" in response
    harness.session_store.switch_session.assert_not_called()
    harness._release_running_agent_state.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_resume_handler_blocks_compaction_stamped_mismatch_before_switch():
    harness = _ResumeHarness()
    harness._session_db = _db_for_target({
        "id": "target-session",
        "git_repo_root": None,
        "cwd": None,
    })
    harness.session_store.load_transcript.return_value = [
        {
            "role": "user",
            "content": "[CONTEXT COMPACTION] summary\n<!-- HERMES_WORKSPACE:/workspace/other -->",
        }
    ]

    with patch("os.getcwd", return_value="/workspace/current"):
        response = await GatewaySlashCommandsMixin._handle_resume_command(
            harness, _event("target-session")
        )

    assert "Cross-workspace resume blocked" in response
    harness.session_store.switch_session.assert_not_called()
    harness._release_running_agent_state.assert_not_called()
