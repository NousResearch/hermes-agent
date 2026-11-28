"""Background idle compaction (``compression.idle_compact_after_seconds``) on every route (#97390).

The housekeeping tick compacts a cached chat once its idle policy has passed, without waiting for the
next message, also where no prompt-cache keep-alive runs (local models, custom endpoints): the chat's
cached agent has no warmer there, so the cache edge never claims it. It runs the cache-edge job (one per
key, the claim gate, the manual-compression pipeline) with the ``idle`` trigger, once per idle stretch,
and only when the turn-start idle gates admit the stored transcript. A channel override sets or
disables the policy for one channel.
"""

import asyncio
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionSource

BIG_HISTORY = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "word " * 400,
                "timestamp": time.time() - 3600} for i in range(12)]
SMALL_HISTORY = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}",
                  "timestamp": time.time() - 3600} for i in range(6)]


KEY = "agent:main:telegram:dm:c1"  # the session key of ``_source()``


def _source(chat_id="c1"):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, user_id="u1", chat_type="dm")


class _Store:
    def __init__(self, entries, history):
        self.entries, self.history = entries, history

    def lookup_by_session_key(self, key):
        return self.entries.get(key)

    def load_transcript(self, session_id):
        return list(self.history)

    def _named_profile_for_key(self, key):
        return None

    def _persistence_for_key(self, key):
        return None

    def _lazy(self, name, factory):
        return self.__dict__.setdefault(name, factory())


def _idle_agent(session_id, *, idle_after=120, idle_for=600.0):
    """A cached agent on a route without keep-alives (no ``_prompt_cache_warmer``)."""
    compressor = SimpleNamespace(threshold_tokens=2_000, summary_target_ratio=0.2, last_compression_rough_tokens=0,
                                 awaiting_real_usage_after_compression=False,
                                 get_active_compression_failure_cooldown=lambda: None)
    return SimpleNamespace(session_id=session_id, compression_enabled=True,
                           compression_idle_compact_after_seconds=idle_after,
                           _last_activity_ts=time.time() - idle_for, context_compressor=compressor,
                           _cached_system_prompt="SYSTEM", _usage_anchor=None, tools=None, api_mode="chat_completions")


@pytest.fixture
def make_runner():
    from gateway.run import GatewayRunner

    runners = []

    def make(agent, history=BIG_HISTORY):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig()
        now = datetime.now()
        entry = SessionEntry(session_key=KEY, session_id=agent.session_id, created_at=now, updated_at=now,
                             origin=_source(), platform=Platform.TELEGRAM, chat_type="dm")
        runner.session_store = _Store({KEY: entry}, history)
        runner.session_authority = None
        runner._agent_cache = {KEY: (agent, "sig")}
        runner._agent_cache_lock = threading.Lock()
        runner.compacted = []

        async def compress(source, entry, history, request, *, trigger=None):
            runner.compacted.append((entry.session_id, trigger))
            return "Compressed"
        runner._run_manual_compression = compress
        runners.append(runner)
        return runner

    yield make
    for runner in runners:
        for task in list(getattr(runner, "_background_tasks", ())):
            task.cancel()


async def _scan(runner):
    await runner._compact_idle_sessions_before_cache_expiry()
    await asyncio.gather(*getattr(runner, "_background_tasks", ()), return_exceptions=True)


@pytest.mark.asyncio
async def test_an_idle_chat_without_keep_alives_compacts_in_the_background_once(make_runner):
    """Past its policy the chat is compacted by the tick (no message needed), and only once per idle
    stretch: the next tick leaves it alone until the chat has new activity."""
    agent = _idle_agent("s1")
    runner = make_runner(agent)

    await _scan(runner)
    await _scan(runner)

    assert runner.compacted == [("s1", "idle")]


@pytest.mark.asyncio
async def test_a_due_idle_policy_is_not_preempted_by_the_cache_edge_offload(make_runner):
    """A route with a warmer claims ``offload`` once its cache window is over; a chat whose idle
    policy is due on that tick is compacted on it, not one housekeeping interval later."""
    agent = _idle_agent("s1")
    agent._prompt_cache_warmer = SimpleNamespace(claim_edge=lambda *, releasing=False: "offload")
    runner = make_runner(agent)

    await _scan(runner)

    assert runner.compacted == [("s1", "idle")]


@pytest.mark.asyncio
async def test_an_evicted_idle_chat_compacts_at_release(make_runner):
    """The idle-TTL sweep runs before the scan in the same tick: a chat due at its eviction is
    compacted by the release job rather than lost with the agent."""
    agent = _idle_agent("s1")
    runner = make_runner(agent)
    runner._agent_cache.clear()

    await runner._compact_then_release_job(KEY, agent)

    assert runner.compacted == [("s1", "idle")]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["policy off", "not idle long enough", "below the floor", "input buffered",
                                  "codex compacts its own thread"])
async def test_the_background_run_leaves_the_chat_alone(make_runner, case):
    agent = _idle_agent("s1", idle_after=0 if case == "policy off" else 120,
                        idle_for=30.0 if case == "not idle long enough" else 600.0)
    if case == "codex compacts its own thread":
        agent.api_mode, agent.codex_app_server_auto_compaction = "codex_app_server", "native"
    runner = make_runner(agent, SMALL_HISTORY if case == "below the floor" else BIG_HISTORY)
    if case == "input buffered":
        # Batches are keyed by the adapter's own batch key (here with a sender suffix), not the session key.
        runner.adapters = {Platform.TELEGRAM: SimpleNamespace(
            _pending_text_batches={"telegram:c1:from:u1": SimpleNamespace(source=_source())})}

    await _scan(runner)

    assert runner.compacted == []


@pytest.mark.asyncio
async def test_input_that_arrives_while_the_job_waits_for_a_slot_cancels_it(make_runner):
    """The claim checked the adapters, but a job may wait behind other summaries: input buffered
    by the time it runs must not be summarised past."""
    from gateway.run_watchers_idle import IDLE_ACTION

    agent = _idle_agent("s1")
    runner = make_runner(agent)
    runner.adapters = {Platform.TELEGRAM: SimpleNamespace(_pending_messages={KEY: object()})}

    await runner._compact_claimed_session(KEY, runner.session_store.entries[KEY], agent, IDLE_ACTION)

    assert runner.compacted == []


def test_a_channel_override_sets_or_disables_the_idle_policy_for_that_channel_only():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig.from_dict({"platforms": {"discord": {"enabled": True, "channel_overrides": {
        "quiet": {"idle_compact_after_seconds": 0},
        "eager": {"idle_compact_after_seconds": "60"},
        "other": {"model": "some/model"},
    }}}})
    user_config = {"compression": {"idle_compact_after_seconds": 1800}}

    def policy(chat_id, cfg=user_config):
        source = SessionSource(platform=Platform.DISCORD, chat_id=chat_id, user_id="u1", chat_type="group")
        return runner._idle_compact_after_for_source(source, cfg)

    assert policy("quiet") == 0
    assert policy("eager") == 60
    assert policy("other") == 1800
    assert policy("unlisted") == 1800
    assert policy("eager", {}) == 60  # the channel turns it on where the global key is off
