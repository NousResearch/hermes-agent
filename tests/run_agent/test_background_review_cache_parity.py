"""Tests that the background review fork inherits the parent's cached system prompt.

Regression coverage for issue #25322 (and PR #17276's first root cause): the
background review's outbound HTTP request must carry the same system bytes as
the parent's so Anthropic/OpenRouter's exact-prefix cache key matches.

Without this, every review rebuilds the system prompt from scratch — fresh
``_hermes_now()`` timestamp, fresh ``session_id``, and a different skills
prompt under the (former) narrow toolset — and the prefix-cache miss costs
roughly the full uncached system-prompt cost per nudge (~26% end-to-end on
Sonnet 4.5 per the contributor's measurement).
"""

from unittest.mock import patch


def _make_agent_stub(agent_cls):
    """Create a minimal AIAgent-like object with just enough state for _spawn_background_review."""
    agent = object.__new__(agent_cls)
    agent.model = "test-model"
    agent.platform = "test"
    agent.provider = "openai"
    agent.session_id = "sess-123"
    agent.quiet_mode = True
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 5
    agent._skill_nudge_interval = 5
    agent.background_review_callback = None
    agent.status_callback = None
    agent._cached_system_prompt = (
        "PARENT-SYSTEM-PROMPT-BYTES — must be inherited verbatim "
        "for prefix-cache parity"
    )
    import datetime as _dt
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    return agent


class _SyncThread:
    """Drop-in replacement for threading.Thread that runs the target inline."""

    def __init__(self, *, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


class _ReviewAgentRecorder:
    """Stand-in for the review-fork AIAgent that records the prompt assignment."""

    def __init__(self, *args, **kwargs):
        self._cached_system_prompt = None
        self._memory_write_origin = None
        self._memory_write_context = None
        self._memory_store = None
        self._memory_enabled = None
        self._user_profile_enabled = None
        self._memory_nudge_interval = None
        self._skill_nudge_interval = None
        self.suppress_status_output = None

    def run_conversation(self, *args, **kwargs):
        raise RuntimeError("stop after recording state — don't actually call the API")

    def shutdown_memory_provider(self):
        pass

    def close(self):
        pass


def test_review_fork_inherits_parent_cached_system_prompt():
    """The review fork's _cached_system_prompt must equal the parent's byte-for-byte.

    Anthropic's prefix cache keys on exact bytes; any divergence (timestamp
    minute tick, fresh session_id, narrower skills_prompt) shifts the key
    and forces a full re-cache. Inheriting the parent's cached prompt is
    the cheap, mechanical fix.
    """
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)

    captured = {}
    parent_prompt = agent._cached_system_prompt

    # Hook the assignment site: record what gets put on the review agent.
    real_recorder_init = _ReviewAgentRecorder.__init__

    def _recorder_init(self, *args, **kwargs):
        real_recorder_init(self, *args, **kwargs)
        # The actual production code assigns _cached_system_prompt AFTER __init__,
        # so we need to capture it on attribute set. Use a property-style sentinel
        # via __setattr__ on this instance.

    with patch.object(run_agent, "AIAgent", _ReviewAgentRecorder), \
         patch("threading.Thread", _SyncThread):
        # Wrap the recorder's __setattr__ so we can see the _cached_system_prompt
        # write that _spawn_background_review performs after construction.
        orig_setattr = _ReviewAgentRecorder.__setattr__

        def _spy_setattr(self, name, value):
            if name == "_cached_system_prompt":
                captured["written_prompt"] = value
            orig_setattr(self, name, value)

        with patch.object(_ReviewAgentRecorder, "__setattr__", _spy_setattr):
            agent._spawn_background_review(
                messages_snapshot=[],
                review_memory=True,
                review_skills=False,
            )

    assert "written_prompt" in captured, (
        "_spawn_background_review never assigned _cached_system_prompt on the review agent"
    )
    assert captured["written_prompt"] == parent_prompt, (
        f"Review fork's _cached_system_prompt diverged from parent's. "
        f"Got {captured['written_prompt']!r}, expected {parent_prompt!r}. "
        "This breaks Anthropic/OpenRouter prefix-cache parity (#25322)."
    )


def test_review_fork_pins_session_start_and_session_id():
    """Defensive complement to cached-system-prompt inheritance.

    Even though ``_cached_system_prompt`` inheritance short-circuits the
    normal rebuild path, pinning ``session_start`` and ``session_id`` to
    the parent's guarantees byte-identical output from any code path that
    re-renders parts of the system prompt (compression, plugin hooks).
    """
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)

    captured = {}

    class _Recorder:
        def __init__(self, *args, **kwargs):
            self._cached_system_prompt = None
            self._memory_write_origin = None
            self._memory_write_context = None
            self._memory_store = None
            self._memory_enabled = None
            self._user_profile_enabled = None
            self._memory_nudge_interval = None
            self._skill_nudge_interval = None
            self.suppress_status_output = None
            self.session_start = None
            self.session_id = None

        def run_conversation(self, *args, **kwargs):
            captured["session_start"] = self.session_start
            captured["session_id"] = self.session_id
            raise RuntimeError("stop after recording")

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    with patch.object(run_agent, "AIAgent", _Recorder), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert captured.get("session_start") == agent.session_start, (
        "Review fork did not inherit parent's session_start — "
        "system-prompt rebuild paths would diverge."
    )
    assert captured.get("session_id") == agent.session_id, (
        "Review fork did not inherit parent's session_id — "
        "system-prompt rebuild paths would diverge."
    )
