"""Tests for background-review fork bridge sub-session routing isolation.

Regression coverage for the claude-bridge CLI-session collision (PRD
bridge-subsession-routing, diagnosed 2026-06-12): the background-review fork
PINS the parent's ``session_id`` for prefix-cache parity, which made it emit the
SAME claude-bridge routing key as the main agent → both routed into ONE Claude
CLI session → the fork's restricted "memory/skill-only" turn contaminated the
main agent's resumed session (the "I suddenly only have memory/skill tools" bug).

The fix: ``_spawn_background_review`` sets ``review_agent._bridge_route_suffix =
"review"`` (an object attribute, seam A — NOT a contextvar, because the fork is
spawned on a bare ``threading.Thread`` with no contextvar copy). The
``_session_routing.py`` provider mixin then appends a distinct ``-review`` tag to
the bridge routing key so the fork gets its OWN CLI session.

These tests drive the REAL ``_spawn_background_review`` path (no stub escape
hatch — the bare-Thread topology is exactly what must be exercised) and assert
the marker is set, plus revert-discrimination (the test must go RED if the
``_bridge_route_suffix = "review"`` line is removed).
"""

from unittest.mock import patch


def _make_agent_stub(agent_cls):
    """Minimal AIAgent-like object with just enough state for _spawn_background_review."""
    agent = object.__new__(agent_cls)
    agent.model = "test-model"
    agent.platform = "test"
    agent.provider = "claude-bridge-f2"
    agent.session_id = "sess-123"
    agent.quiet_mode = True
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 5
    agent._skill_nudge_interval = 5
    agent.background_review_callback = None
    agent.status_callback = None
    agent._cached_system_prompt = "PARENT-SYSTEM-PROMPT-BYTES"
    import datetime as _dt
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.enabled_toolsets = ["memory", "skills"]
    agent.disabled_toolsets = []
    agent._chat_id = "c1"
    agent._chat_name = "chan"
    agent._chat_type = "channel"
    return agent


class _SyncThread:
    """Drop-in replacement for threading.Thread that runs the target inline."""

    def __init__(self, *, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def _spawn_and_capture_suffix():
    """Drive the REAL _spawn_background_review and capture _bridge_route_suffix.

    Returns the value assigned to review_agent._bridge_route_suffix (or the
    sentinel ``"<unset>"`` if the production code never set it — the revert
    state). Captured via a __setattr__ spy, exactly like the cache-parity test.
    """
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)
    captured = {"suffix": "<unset>"}

    class _Recorder:
        def __init__(self, *args, **kwargs):
            pass

        def run_conversation(self, *args, **kwargs):
            raise RuntimeError("stop after recording — don't call the API")

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    orig_setattr = _Recorder.__setattr__

    def _spy_setattr(self, name, value):
        if name == "_bridge_route_suffix":
            captured["suffix"] = value
        orig_setattr(self, name, value)

    with patch.object(run_agent, "AIAgent", _Recorder), \
            patch("threading.Thread", _SyncThread), \
            patch.object(_Recorder, "__setattr__", _spy_setattr):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )
    return captured["suffix"]


def test_review_fork_sets_bridge_route_suffix():
    """The real spawn path must mark the fork with _bridge_route_suffix='review'.

    This is the emission half of the fix. If the
    ``review_agent._bridge_route_suffix = "review"`` line is removed from
    background_review.py, this assertion goes RED (revert-discrimination).
    """
    suffix = _spawn_and_capture_suffix()
    assert suffix == "review", (
        f"Background-review fork was not marked for bridge sub-session isolation: "
        f"_bridge_route_suffix={suffix!r} (expected 'review'). Without this the "
        f"fork shares the parent's bridge CLI session and contaminates it."
    )
