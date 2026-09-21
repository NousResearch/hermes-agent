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
        persistent=SimpleNamespace(run_generation=7),
    )
    store = SimpleNamespace(lookup_by_session_key=lambda key: entry)
    runner = SimpleNamespace(
        _post_turn_compaction_tasks={},
        _session_key_for_source=lambda source: session_key,
        _peek_session_state=lambda key: state,
        _is_session_run_current=lambda key, generation: generation == state.persistent.run_generation,
        _retain_background_task=lambda task: task,
        session_store=store,
        async_session_store=SimpleNamespace(
            load_transcript=AsyncMock(return_value=[{"role": "user"}])
        ),
    )

    async def run_hygiene(*args, **kwargs):
        await release.wait()

    runner._hmwa_run_session_hygiene = AsyncMock(side_effect=run_hygiene)
    event = SimpleNamespace(_agent_turn_succeeded=True)
    first = GatewayRunner._schedule_post_turn_background_compaction(
        runner, session_entry=entry, source=object(),
        agent_result="done", final_response="done", event=event,
    )
    second = GatewayRunner._schedule_post_turn_background_compaction(
        runner, session_entry=entry, source=object(),
        agent_result="done", final_response="done", event=event,
    )

    assert first is second
    await asyncio.sleep(0)
    runner.async_session_store.load_transcript.assert_awaited_once_with("sid-1")
    runner._hmwa_run_session_hygiene.assert_awaited_once()
    assert runner._hmwa_run_session_hygiene.await_args.kwargs["trigger_tokens"] == 1234
    assert runner._hmwa_run_session_hygiene.await_args.kwargs["cache_owner"] is agent
    assert runner._hmwa_run_session_hygiene.await_args.kwargs["commit_authority_check"]()

    release.set()
    await first


def test_failed_turn_text_cannot_schedule_post_turn_compaction():
    runner = SimpleNamespace(
        _session_key_for_source=lambda source: "agent:main:test",
        _peek_session_state=lambda key: (_ for _ in ()).throw(
            AssertionError("failed turn must stop before session state lookup")
        ),
    )

    task = GatewayRunner._schedule_post_turn_background_compaction(
        runner,
        session_entry=SimpleNamespace(session_id="sid-1"),
        source=object(),
        agent_result="Provider request failed",
        final_response="Provider request failed",
        event=SimpleNamespace(_agent_turn_succeeded=False),
    )

    assert task is None


@pytest.mark.asyncio
async def test_post_turn_compaction_drops_stale_route_while_transcript_load_is_blocked():
    load_started = asyncio.Event()
    release_load = asyncio.Event()
    session_key = "agent:main:test"
    entry = SimpleNamespace(session_id="sid-1")
    live = {"entry": entry}
    state = SimpleNamespace(
        turn=SimpleNamespace(
            agent=SimpleNamespace(
                compression_defer_threshold_to_post_turn=True,
                context_compressor=SimpleNamespace(threshold_tokens=1234),
            )
        ),
        persistent=SimpleNamespace(run_generation=7),
    )

    async def load_transcript(session_id):
        load_started.set()
        await release_load.wait()
        return [{"role": "user"}]

    runner = SimpleNamespace(
        _post_turn_compaction_tasks={},
        _session_key_for_source=lambda source: session_key,
        _peek_session_state=lambda key: state,
        _is_session_run_current=lambda key, generation: generation == state.persistent.run_generation,
        _retain_background_task=lambda task: task,
        session_store=SimpleNamespace(lookup_by_session_key=lambda key: live["entry"]),
        async_session_store=SimpleNamespace(load_transcript=AsyncMock(side_effect=load_transcript)),
        _hmwa_run_session_hygiene=AsyncMock(),
    )

    task = GatewayRunner._schedule_post_turn_background_compaction(
        runner, session_entry=entry, source=object(), agent_result="done",
        final_response="done", event=SimpleNamespace(_agent_turn_succeeded=True),
    )
    await load_started.wait()
    live["entry"] = SimpleNamespace(session_id="sid-2")
    state.persistent.run_generation = 8
    release_load.set()
    await task

    runner._hmwa_run_session_hygiene.assert_not_awaited()


def test_post_turn_commit_authority_rechecks_live_route_and_generation():
    from agent.conversation_compression import CompressionCommitFence

    authority = {"current": True}
    fence = CompressionCommitFence(admission_check=lambda: authority["current"])
    authority["current"] = False

    assert fence.begin_commit() is False
    assert fence.is_cancelled


def test_stale_post_turn_cleanup_cannot_evict_agent_reused_by_successor_turn():
    cached_agent = object()
    cache = {"agent:main:test": (cached_agent, "sig")}
    runner = SimpleNamespace(
        _agent_cache_lock=None,
        _agent_cache=cache,
        _is_session_run_current=lambda key, generation: False,
    )

    GatewayRunner._evict_cached_agent(
        runner,
        "agent:main:test",
        expected_agent=cached_agent,
        run_generation=7,
    )

    assert cache["agent:main:test"][0] is cached_agent
