from unittest.mock import AsyncMock, patch

import pytest

from gateway.session import build_session_key
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source
from tests.gateway.test_35994_reset_button_deadlock import (
    _make_event as make_reset_event,
    _make_runner_with_cached_agent,
)


def _pending_interaction(session_key: str):
    from tools.clarify_interaction import create_clarify_interaction

    return create_clarify_interaction(
        session_key=session_key,
        platform="telegram",
        question="Deploy where?",
        choices=["staging", "production"],
        chat_id="c1",
        user_id="u1",
        ttl_seconds=3600,
    )


def _status(interaction_id: str) -> str:
    from tools.clarify_interaction import get_interaction

    interaction = get_interaction(interaction_id)
    assert interaction is not None
    return interaction.status


@pytest.mark.asyncio
async def test_new_session_cancels_pending_durable_clarify(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        runner = _make_runner_with_cached_agent(lambda: None)
        session_key = next(iter(runner.session_store._entries))
        interaction = _pending_interaction(session_key)

        await runner._handle_reset_command(make_reset_event("/new"))

        assert _status(interaction.interaction_id) == "cancelled"
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_interrupt_cancels_pending_durable_clarify(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        runner, _adapter = make_restart_runner()
        source = make_restart_source()
        session_key = build_session_key(source)
        interaction = _pending_interaction(session_key)

        await runner._interrupt_and_clear_session(
            session_key,
            source,
            interrupt_reason="test interrupt",
            invalidation_reason="test_interrupt",
        )

        assert _status(interaction.interaction_id) == "cancelled"
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_gateway_shutdown_cancels_all_pending_durable_clarify(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        runner, adapter = make_restart_runner()
        adapter.disconnect = AsyncMock()
        interaction = _pending_interaction("session-not-currently-running")
        late_interaction = None

        async def create_prompt_during_shutdown(_active_agents):
            nonlocal late_interaction
            late_interaction = _pending_interaction("session-created-during-drain")

        runner._finalize_shutdown_agents = create_prompt_during_shutdown

        with patch("gateway.status.remove_pid_file"), patch(
            "gateway.status.write_runtime_status"
        ):
            await runner.stop()

        assert _status(interaction.interaction_id) == "cancelled"
        assert late_interaction is not None
        assert _status(late_interaction.interaction_id) == "cancelled"
    finally:
        reset_hermes_home_override(token)
