"""The gateway's cache-edge owner never blocks its event loop on the claim, and an evicted agent handed to
a release-time job is released exactly once whatever happens to that job (cancelled before its first step
by shutdown, or never scheduled because the loop stopped)."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway.run_watchers import drain_pending_cache_releases


class _Edge:
    def __init__(self, action):
        self.action, self.threads = action, []

    def claim_edge(self, *, releasing=False, now=None):
        self.threads.append(threading.get_ident())
        return self.action


def _runner(action=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._gateway_loop = asyncio.get_running_loop()
    runner.released = []
    runner._commit_then_release_soft = lambda agent, key, boundary=False: runner.released.append(key)
    runner._spawn_release_thread = lambda target, args, name, **kw: target(*args)
    agent = SimpleNamespace(session_id="s1", _prompt_cache_warmer=_Edge(action))
    runner._agent_cache = {"k1": (agent, "sig")}
    runner._agent_cache_lock = threading.Lock()
    return runner, agent


@pytest.mark.asyncio
async def test_the_due_scan_claims_off_the_event_loop():
    runner, agent = _runner()
    await runner._compact_idle_sessions_before_cache_expiry()
    assert agent._prompt_cache_warmer.threads and threading.get_ident() not in agent._prompt_cache_warmer.threads


@pytest.mark.asyncio
@pytest.mark.parametrize("fate", ["completes", "cancelled_before_first_step", "loop_never_ran_it"])
async def test_an_evicted_agent_is_released_exactly_once(fate):
    runner, agent = _runner(action=None)
    assert runner._compact_then_release("k1", agent) is True
    if fate == "loop_never_ran_it":
        drain_pending_cache_releases(runner)  # shutdown, before the loop ran the callback
    else:
        await asyncio.sleep(0)  # the loop runs the callback: the job task now exists
        (task,) = runner._cache_expiry_jobs.values()
        if fate == "cancelled_before_first_step":
            task.cancel()  # what _stop_release_runtime_state does to every background task
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
        drain_pending_cache_releases(runner)
    await asyncio.sleep(0)
    assert runner.released == ["k1"]
    if fate != "loop_never_ran_it":
        assert threading.get_ident() not in agent._prompt_cache_warmer.threads
