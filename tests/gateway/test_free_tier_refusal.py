"""A continuation admission refusal never falls back to transcript writes."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_gateway_does_not_persist_refused_input():
    from gateway.run_turn import GatewayTurnMixin
    store = SimpleNamespace(append_to_transcript=AsyncMock(), update_session=AsyncMock())
    runner = SimpleNamespace(async_session_store=store, _session_db=None,
                             _refresh_agent_cache_message_count=AsyncMock())
    await GatewayTurnMixin._hmwa_persist_turn_transcript(
        runner, event=None, source=None, session_entry=None, session_key="s",
        agent_result={"refusal_reason": "free_tier_limit"}, agent_messages=[],
        prepared=None, response="sign in", agent_failed_early=False,
        hidden_reasoning_incomplete=False, is_context_overflow_failure=False)
    store.append_to_transcript.assert_not_called()
    store.update_session.assert_not_called()
