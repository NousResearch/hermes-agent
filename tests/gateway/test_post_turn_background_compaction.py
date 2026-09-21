"""Post-turn threshold compaction stays off the gateway reply path."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_post_turn_compaction_is_background_deduped_and_fenced_to_session():
    release = asyncio.Event()
    session_key = "agent:main:test"
    entry = SimpleNamespace(session_id="sid-1")
    agent = SimpleNamespace(
        compression_defer_threshold_to_post_turn=True,
        context_compressor=SimpleNamespace(threshold_tokens=1234),
    )
    state = SimpleNamespace(
        turn=SimpleNamespace(agent=agent),
        conversation=SimpleNamespace(run_generation=7),
    )
    runner = SimpleNamespace(
        _post_turn_compaction_tasks={},
        _session_key_for_source=lambda source: session_key,
        _peek_session_state=lambda key: state,
        _retain_background_task=lambda task: task,
        async_session_store=SimpleNamespace(
            load_transcript=AsyncMock(return_value=[{"role": "user"}])
        ),
    )

    async def run_hygiene(*args, **kwargs):
        await release.wait()

    runner._hmwa_run_session_hygiene = AsyncMock(side_effect=run_hygiene)
    first = GatewayRunner._schedule_post_turn_background_compaction(
        runner, session_entry=entry, source=object(),
        agent_result={"final_response": "done"}, final_response="done", event=object(),
    )
    second = GatewayRunner._schedule_post_turn_background_compaction(
        runner, session_entry=entry, source=object(),
        agent_result={"final_response": "done"}, final_response="done", event=object(),
    )

    assert first is second
    await asyncio.sleep(0)
    runner.async_session_store.load_transcript.assert_awaited_once_with("sid-1")
    runner._hmwa_run_session_hygiene.assert_awaited_once()
    assert runner._hmwa_run_session_hygiene.await_args.kwargs["trigger_tokens"] == 1234

    release.set()
    await first