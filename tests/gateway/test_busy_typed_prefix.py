"""Busy notices spell commands with the typed prefix of the conversation's platform.

Matrix clients such as Element treat a typed ``/stop`` as a client command, so
the Matrix adapter sets ``typed_command_prefix = "!"`` and rewrites ``!stop``
to ``/stop``. When a notice tells a Matrix user to send ``/stop``, the client
intercepts the command and Hermes never receives it. Platforms whose prefix is
``/`` keep the slash spelling.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agent.onboarding
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.commands import resolve_command

_TYPED_PREFIX = {Platform.TELEGRAM: "/", Platform.MATRIX: "!"}

# A slash that starts a command spelling, as opposed to the " / " separator in
# "/goal status / pause / clear".
_SLASH_COMMAND = re.compile(r"(?<=[\s`(])/(?=[a-z])")


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.adapters = {
        platform: SimpleNamespace(typed_command_prefix=prefix) for platform, prefix in _TYPED_PREFIX.items()
    }
    runner.session_store = None
    runner.config = None
    runner._running_agents = {}
    return runner


def _make_event(platform: Platform, text: str) -> MessageEvent:
    source = SessionSource(platform=platform, user_id="u1", chat_id="c1", user_name="tester", chat_type="dm")
    return MessageEvent(text=text, source=source, message_id="m1")


async def _dispatch_busy_command(runner: GatewayRunner, event: MessageEvent) -> str:
    source = event.source
    quick_key = runner._session_key_for_source(source)
    return await runner._dispatch_busy_slash_command(event, resolve_command(event.get_command()), quick_key, source)


async def _refine_while_running(runner: GatewayRunner, event: MessageEvent) -> str:
    runner._running_agents[runner._session_key_for_source(event.source)] = MagicMock()
    _key, _agent, error = runner._idle_cached_agent_or_error(event, "refine")
    return error


async def _queued_behind_subagent_ack(runner: GatewayRunner, event: MessageEvent) -> str:
    return runner._compose_busy_ack_message(
        event, 0.0, None, None,
        is_steer_mode=False, is_queue_mode=True, is_redirect_mode=False,
        demoted_for_subagents=True, demoted_for_compression=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "reply_for"),
    [
        ("/model gpt-5", _dispatch_busy_command),
        ("/codex-runtime on", _dispatch_busy_command),
        ("/moa compare these", _dispatch_busy_command),
        ("/goal ship the release", _dispatch_busy_command),
        ("/loop 5m check the build", _dispatch_busy_command),
        ("/reasoning high", _dispatch_busy_command),
        ("/refine", _refine_while_running),
        ("follow-up while a subagent works", _queued_behind_subagent_ack),
    ],
)
async def test_busy_notice_uses_platform_typed_prefix(monkeypatch, text, reply_for):
    # Show the one-time busy-input hint on every call; it is part of the ack.
    monkeypatch.setattr(agent.onboarding, "is_seen", lambda *_args: False)
    monkeypatch.setattr(agent.onboarding, "mark_seen", lambda *_args: None)

    telegram = await reply_for(_make_runner(), _make_event(Platform.TELEGRAM, text))
    matrix = await reply_for(_make_runner(), _make_event(Platform.MATRIX, text))

    assert _SLASH_COMMAND.search(telegram), telegram
    assert matrix == _SLASH_COMMAND.sub(_TYPED_PREFIX[Platform.MATRIX], telegram)
