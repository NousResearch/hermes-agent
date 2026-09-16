"""Messaging checkpoint commands must not inspect host state for container sessions."""

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class _HostStoreTrap:
    def list_checkpoints(self, *_args, **_kwargs):
        raise AssertionError("container rollback read the host checkpoint store")

    def session_diff(self, *_args, **_kwargs):
        raise AssertionError("container diff read the host checkpoint store")


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id="user-1",
            chat_id="chat-1",
            chat_type="dm",
        ),
    )


@pytest.mark.asyncio
async def test_checkpoint_commands_refuse_container_session_before_store_access(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = None
    runner.config = None
    monkeypatch.setattr(runner, "_checkpoint_manager", lambda: _HostStoreTrap())

    rollback = await runner._handle_rollback_command(_event("/rollback"))
    session_diff = await runner._handle_diff_command(_event("/diff session"))

    assert "unavailable for terminal.backend=docker" in rollback
    assert session_diff == rollback