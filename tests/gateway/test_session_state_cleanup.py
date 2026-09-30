"""Regression tests for _release_running_agent_state and SessionDB shutdown.

Before this change, running-agent state lived in three dicts that drifted
out of sync:

  self._running_agents       — AIAgent instance per session key
  self._running_agents_ts    — start timestamp per session key
  self._busy_ack_ts          — last busy-ack timestamp per session key

Six cleanup sites did ``del self._running_agents[key]`` without touching
the other two; one site only popped ``_running_agents`` and
``_running_agents_ts``; and only the stale-eviction site cleaned all
three.  Each missed entry was a small persistent leak.

Also: SessionDB connections were never closed on gateway shutdown,
leaving WAL locks in place until Python actually exited.
"""

from unittest.mock import MagicMock


def _make_runner():
    """Bare GatewayRunner wired with just the state the helper touches."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    return runner


class TestReleaseRunningAgentStateUnit:
    def test_pops_all_three_dicts(self):
        runner = _make_runner()
        runner._running_agents["k"] = MagicMock()
        runner._running_agents_ts["k"] = 123.0
        runner._busy_ack_ts["k"] = 456.0

        runner._release_running_agent_state("k")

        assert "k" not in runner._running_agents
        assert "k" not in runner._running_agents_ts
        assert "k" not in runner._busy_ack_ts

    def test_idempotent_on_missing_key(self):
        """Calling twice (or on an absent key) must not raise."""
        runner = _make_runner()
        runner._release_running_agent_state("missing")
        runner._release_running_agent_state("missing")  # still fine






class TestSessionResetZombieRace:
    """Regression for #28686 — a session_reset racing the in-flight run's
    guarded release must not leave a dead agent locking the slot forever.
    """

    def test_generation_guard_blocks_then_unconditional_release_evicts(self):
        runner = _make_runner()
        runner._session_run_generation = {}
        key = "agent:main:telegram:private:1"

        gen_n = runner._begin_session_run_generation(key)
        dead_agent = MagicMock()
        runner._running_agents[key] = dead_agent
        runner._running_agents_ts[key] = 1.0
        runner._busy_ack_ts[key] = 1.0

        # session_reset bumps the generation while gen-N is still in flight.
        runner._invalidate_session_run_generation(key, reason="session_reset")

        # gen-N's own guarded release is correctly blocked — slot would be a
        # zombie if nothing else cleared it (the pre-fix behaviour).
        assert runner._release_running_agent_state(key, run_generation=gen_n) is False
        assert runner._running_agents.get(key) is dead_agent

        # The fix: unconditional release (no run_generation) always clears it.
        assert runner._release_running_agent_state(key) is True
        assert key not in runner._running_agents
        assert key not in runner._running_agents_ts
        assert key not in runner._busy_ack_ts

