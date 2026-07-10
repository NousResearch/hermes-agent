"""Regression tests for #62151 — cron/gateway inactivity watchdogs that can
never fire because their own "still waiting" heartbeat keeps resetting the
metric they check.

Root cause: ``interruptible_api_call`` (agent/chat_completion_helpers.py)
touches the agent's activity tracker every ~30s purely to announce "still
waiting for the response" — including while the underlying call is wedged
before it ever opens a socket (e.g. a hung DNS lookup). Every watchdog that
read ``seconds_since_activity`` to decide whether a run is dead (cron's
inactivity timeout, the gateway's per-session inactivity timeout, and the
stale ``_running_agents`` lock eviction) was reading that same
self-refreshing field, so none of them could ever detect this class of hang.

The fix splits the signal in two on ``AIAgent``:
  - ``seconds_since_activity`` / ``_touch_activity(desc)``: human-facing
    "what's it doing" status, refreshed by heartbeats too.
  - ``seconds_since_progress`` / ``_touch_activity(desc, progress=False)``:
    watchdog-facing "is it actually moving", NOT refreshed by "still
    waiting" heartbeats — only by genuine forward movement (a new call
    attempt starting, a tool finishing, a stream chunk arriving).
"""

from unittest.mock import patch

from run_agent import AIAgent


def _make_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


class TestTouchActivityProgressSplit:
    def test_default_touch_advances_both_clocks(self):
        agent = _make_agent()
        agent._touch_activity("starting API call #1")
        summary = agent.get_activity_summary()
        assert summary["seconds_since_activity"] < 0.5
        assert summary["seconds_since_progress"] < 0.5

    def test_progress_false_advances_activity_but_not_progress(self):
        agent = _make_agent()
        # Simulate a call attempt starting (real progress)...
        agent._touch_activity("starting API call #1")
        first_progress_ts = agent._last_progress_ts

        # ...then a "still waiting" heartbeat, exactly like
        # interruptible_api_call's poll loop emits every ~30s while the
        # worker thread is still alive, with or without an actual hang.
        agent._touch_activity(
            "waiting for non-streaming response (30s elapsed)", progress=False
        )

        summary = agent.get_activity_summary()
        # The human-facing clock is refreshed (still legitimately "just
        # touched" from a status point of view)...
        assert summary["seconds_since_activity"] < 0.5
        # ...but the watchdog-facing clock must NOT have moved.
        assert agent._last_progress_ts == first_progress_ts

    def test_wedged_call_is_visible_via_seconds_since_progress(self):
        """Simulates the reported bug: a call attempt starts, then only
        progress=False heartbeats fire for a long time (the network call is
        wedged before ever sending a byte). seconds_since_activity stays
        fresh forever (the old, blind signal); seconds_since_progress grows
        unbounded (the new signal a watchdog can actually trust).
        """
        agent = _make_agent()
        agent._touch_activity("starting API call #2")

        # Back-date the progress clock to simulate real elapsed time without
        # sleeping in the test — the heartbeat loop would otherwise touch it
        # every 30s for hours in the real incident.
        agent._last_progress_ts -= 5400  # 90 minutes "stuck"

        # The poll loop keeps announcing "still waiting" every ~30s, as it
        # does for the entire 2.5h+ in the reported incident.
        agent._touch_activity(
            "waiting for non-streaming response (5400s elapsed)", progress=False
        )

        summary = agent.get_activity_summary()
        # Old behavior (bug): this is what every watchdog used to read, and
        # it never leaves the single-digit-seconds range no matter how long
        # the call has really been stuck.
        assert summary["seconds_since_activity"] < 1.0
        # Fixed behavior: this is what the watchdogs now read, and it
        # correctly reflects the 90 minutes of no real progress.
        assert summary["seconds_since_progress"] >= 5400

    def test_get_activity_summary_without_prior_touch_does_not_raise(self):
        """Older/mocked agents (or an agent whose loop hasn't started yet)
        may have no _last_progress_ts. get_activity_summary() must fall back
        to _last_activity_ts rather than raising AttributeError."""
        agent = _make_agent()
        agent._touch_activity("boot")
        assert hasattr(agent, "_last_progress_ts")
        del agent._last_progress_ts
        summary = agent.get_activity_summary()
        assert summary["seconds_since_progress"] < 1.0
