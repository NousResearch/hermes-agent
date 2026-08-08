"""``/update`` must give container users the container instructions.

A containerized deployment has no working tree, so ``/update`` fell through to
``gateway.update.not_git_repo`` — "not a git repository". That is technically
true and practically a dead end: it tells the user nothing about how to update.

The container-specific message already exists
(``hermes_cli.config.format_docker_update_message``) and is already used by the
CLI and web paths; these tests pin that the gateway slash-command path routes
there too.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event():
    return MessageEvent(
        text="/update",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id="1",
            chat_id="2",
            user_name="tester",
        ),
    )


def _make_runner():
    """Bare GatewayRunner — the handler only needs the mixin."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = None
    return runner


@pytest.fixture
def worktree_less_root(tmp_path):
    """A project root with no ``.git``, wired in via the module's ``__file__``."""
    fake_root = tmp_path / "app"
    (fake_root / "gateway").mkdir(parents=True)
    fake_file = str(fake_root / "gateway" / "slash_commands.py")
    with patch("gateway.slash_commands.__file__", fake_file):
        yield fake_root


@pytest.mark.asyncio
async def test_container_gets_docker_update_message(worktree_less_root):
    """Inside a container, /update returns the container instructions."""
    from hermes_cli.config import format_docker_update_message

    with patch("gateway.restart.is_container_restart_context", lambda: True), \
         patch("hermes_cli.config.is_managed", lambda: False):
        result = await _make_runner()._handle_update_command(_make_event())

    assert result == format_docker_update_message()


@pytest.mark.asyncio
async def test_non_container_keeps_the_git_message(worktree_less_root):
    """Outside a container the existing message is unchanged."""
    from hermes_cli.config import format_docker_update_message

    with patch("gateway.restart.is_container_restart_context", lambda: False), \
         patch("hermes_cli.config.is_managed", lambda: False):
        result = await _make_runner()._handle_update_command(_make_event())

    assert result != format_docker_update_message()
    assert "docker pull" not in result
