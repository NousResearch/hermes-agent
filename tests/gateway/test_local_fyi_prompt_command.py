"""``/fyi`` on a local session's canonical ``slash.exec`` (gateway/session_commands.py).

The registry's busy contract carries /fyi mid-run (``busy_handler=fyi`` → the steer slot with
context-only framing, ``gateway/run_busy.py``); the idle path is a prompt command:
``_hm_cmd_fyi`` rewrites the turn into the framed payload the client submits as its next
prompt. These pin the canonical wiring that lets Desktop/ACP/Ink dispatch /fyi through
``slash.exec`` instead of refusing it as unsupported.
"""

import asyncio

from gateway.platforms.event import MessageEvent
from gateway.run_busy import GatewayBusySessionMixin
from gateway.run_inbound import GatewayInboundMixin
from gateway.session_commands import _PROMPT_COMMANDS, command_verdict


def test_fyi_is_dispatchable_on_a_local_session():
    """Both halves of the wiring: the command is not in the canonical refusal set, and its
    busy variant resolves by the registry's ``busy_handler=fyi`` naming convention."""
    assert command_verdict('fyi', 'the deploy window moved to 22:00') == 'control'
    assert command_verdict('fyi', '') == 'control'
    assert 'fyi' in _PROMPT_COMMANDS
    assert GatewayBusySessionMixin._BUSY_SPECIAL_HANDLERS['fyi'] == '_busy_fyi_command'
    assert hasattr(GatewayBusySessionMixin, '_busy_fyi_command')


def test_idle_fyi_rewrites_the_turn_into_the_framed_payload():
    """``_hm_cmd_fyi`` reports not-handled with the event text rewritten, so
    ``execute_command`` returns the framed payload as the client's ``send`` directive."""
    event = MessageEvent(text='/fyi the staging database was restored at 3pm')
    event.allow_gateway_control = True
    handled, output = asyncio.run(GatewayInboundMixin._hm_cmd_fyi(
        object.__new__(GatewayInboundMixin), event, None, None))
    assert handled is False and output is None
    assert event.text.startswith('FYI — additional context from the user.')
    assert 'the staging database was restored at 3pm' in event.text
    assert '/fyi' not in event.text  # the command shell never reaches the model


def test_idle_fyi_without_a_payload_answers_usage():
    event = MessageEvent(text='/fyi')
    event.allow_gateway_control = True
    handled, output = asyncio.run(GatewayInboundMixin._hm_cmd_fyi(
        object.__new__(GatewayInboundMixin), event, None, None))
    assert handled is True
    assert output  # a usage hint the client shows instead of submitting a turn
