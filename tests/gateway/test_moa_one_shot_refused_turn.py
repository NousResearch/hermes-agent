"""A `/moa <prompt>` refused before its turn claim must not leave the MoA override armed.

`/moa` arms its one-shot snapshot and installs the MoA override during idle-command dispatch,
then falls through as the turn. The lobby, external-drain and concurrent-session gates can
still refuse that turn. No turn then runs to settle the snapshot, so before this fix the next
ordinary message silently ran through MoA.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

PRIOR = {"provider": "openrouter", "model": "gpt-4"}
_MOA_CFG = {"moa": {"default_preset": "default"}}


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text, message_type=MessageType.TEXT, source=make_restart_source(), message_id="m1",
    )


def _runner(gate: str):
    runner, _ = make_restart_runner()
    runner._external_drain_active = gate == "drain"
    if gate.startswith("lobby"):
        runner._is_telegram_topic_root_lobby = lambda _source: True
        runner._should_send_telegram_lobby_reminder = lambda _source: gate == "lobby"
    if gate == "limit":
        runner._claim_active_session_slot = lambda _key, _source: (None, "session limit reached")
    key = runner._session_key_for_source(make_restart_source())
    runner._session_state(key).conversation.model_override = dict(PRIOR)
    return runner, key


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", ["lobby", "lobby-debounced", "drain", "limit"])
async def test_refused_moa_turn_restores_the_prior_override(gate):
    runner, key = _runner(gate)

    with patch("hermes_cli.config.load_config", return_value=_MOA_CFG):
        reply = await runner._handle_message(_event("/moa compare these two plans"))

    # The debounced lobby refusal is silent.
    assert (reply is None) == (gate == "lobby-debounced")
    conv = runner._session_state(key).conversation
    assert conv.model_override == PRIOR
    assert not conv.one_turn_restore
    assert GatewayRunner._one_turn_restore_armed(runner, key) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["hello", "/moa compare these two plans"])
async def test_refused_turn_keeps_an_earlier_once_snapshot(text):
    """Only the snapshot this message armed is settled: a `/model --once` armed by an earlier
    command keeps its snapshot and override, even when the refused message was a `/moa`, and
    still waits for the turn it was meant for."""
    runner, key = _runner("drain")
    conv = runner._session_state(key).conversation
    once_override = {"provider": "anthropic", "model": "claude-opus-5-5"}
    snapshot = {"had_override": True, "override": dict(PRIOR)}
    conv.model_override = dict(once_override)
    conv.one_turn_restore = dict(snapshot)

    with patch("hermes_cli.config.load_config", return_value=_MOA_CFG):
        reply = await runner._handle_message(_event(text))

    assert "draining" in reply.lower()
    assert conv.model_override == once_override
    assert conv.one_turn_restore == snapshot
