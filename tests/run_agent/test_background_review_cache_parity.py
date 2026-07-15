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
    # Non-None so the test catches a missing-kwarg regression.
    agent.enabled_toolsets = ["memory", "skills", "terminal"]
    agent.disabled_toolsets = ["spotify", "feishu_doc"]
    # Non-None so the test catches reasoning_config NOT being inherited —
    # which would put the fork into a different Anthropic cache namespace.
    agent.reasoning_config = {"enabled": True, "effort": "medium"}
    return agent


class _SyncThread:
    """Drop-in replacement for threading.Thread that runs the target inline."""

    def __init__(self, *, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def _make_recorder_class(captured=None, record_on_run=()):
    """Build a Recorder class standing in for the review-fork AIAgent.

    Keeps the stub attribute list in ONE place: when
    ``_spawn_background_review`` starts touching a new fork attribute, only
    this factory needs the extra stub — not one copy per test.

    ``captured`` (dict): if given, ``__init__`` stores the full constructor
    kwargs under ``captured["init_kwargs"]`` so tests can assert on both
    kwarg values and kwarg *presence*.
    ``record_on_run``: instance attribute names copied into ``captured`` when
    ``run_conversation`` fires — for values the production code assigns
    after construction.
    """

    class _Recorder:
        def __init__(self, *args, **kwargs):
            if captured is not None:
                captured["init_kwargs"] = dict(kwargs)
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
            if captured is not None:
                for _name in record_on_run:
                    captured[_name] = getattr(self, _name)
            raise RuntimeError(
                "stop after recording — don't actually call the API"
            )

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    return _Recorder


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

    _Recorder = _make_recorder_class()

    with patch.object(run_agent, "AIAgent", _Recorder), \
         patch("threading.Thread", _SyncThread):
        # The production code assigns _cached_system_prompt AFTER __init__,
        # so wrap the recorder's __setattr__ to see that post-construction
        # write from _spawn_background_review.
        orig_setattr = _Recorder.__setattr__

        def _spy_setattr(self, name, value):
            if name == "_cached_system_prompt":
                captured["written_prompt"] = value
            orig_setattr(self, name, value)

        with patch.object(_Recorder, "__setattr__", _spy_setattr):
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
    _Recorder = _make_recorder_class(
        captured, record_on_run=("session_start", "session_id")
    )

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


def test_review_fork_inherits_parent_toolset_config():
    """``tools[]`` byte-stability: fork must inherit parent's toolset config."""
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)

    captured = {}
    _Recorder = _make_recorder_class(captured)

    with patch.object(run_agent, "AIAgent", _Recorder), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    init_kwargs = captured.get("init_kwargs", {})
    assert init_kwargs.get("enabled_toolsets") == agent.enabled_toolsets, (
        f"enabled_toolsets mismatch: {init_kwargs.get('enabled_toolsets')!r} "
        f"vs expected {agent.enabled_toolsets!r}"
    )
    assert init_kwargs.get("disabled_toolsets") == agent.disabled_toolsets, (
        f"disabled_toolsets mismatch: {init_kwargs.get('disabled_toolsets')!r} "
        f"vs expected {agent.disabled_toolsets!r}"
    )


def test_review_fork_inherits_parent_reasoning_config():
    """``reasoning_config`` parity on the default (non-routed) path.

    The fork must inherit the parent's value so the request body's
    ``thinking`` / ``output_config`` match — Anthropic's cache is
    namespaced by ``thinking`` presence.
    """
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)

    captured = {}
    _Recorder = _make_recorder_class(captured)

    with patch.object(run_agent, "AIAgent", _Recorder), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    init_kwargs = captured.get("init_kwargs", {})
    assert init_kwargs.get("reasoning_config") == agent.reasoning_config, (
        f"reasoning_config mismatch: {init_kwargs.get('reasoning_config')!r} "
        f"vs expected {agent.reasoning_config!r}"
    )


def test_routed_review_fork_does_not_inherit_reasoning_config():
    """Routed aux path: the fork must NOT inherit the parent's reasoning_config.

    When ``auxiliary.background_review.{provider,model}`` routes the review
    to a different model, cache parity is moot (the cache is cold on that
    model regardless) and the parent's effort vocabulary may be invalid for
    the routed model/provider (OpenRouter ``extra_body.reasoning.effort`` is
    forwarded unclamped; codex_responses passes ``max``/``ultra`` through
    unmapped except on gpt-5.6/xAI). The routed fork must fall back to
    provider defaults, mirroring the ``not _routed`` gate on
    ``_cached_system_prompt`` inheritance.
    """
    import run_agent
    import agent.background_review as bg_review

    agent_stub = _make_agent_stub(run_agent.AIAgent)

    captured = {}
    _Recorder = _make_recorder_class(captured)

    routed_runtime = {
        "provider": "openrouter",
        "model": "aux-cheap-model",
        "api_key": "test-key",
        "base_url": None,
        "api_mode": None,
        "credential_pool": None,
        "request_overrides": {},
        "max_tokens": None,
        "command": None,
        "args": [],
        "routed": True,
    }

    with patch.object(run_agent, "AIAgent", _Recorder), \
         patch.object(bg_review, "_resolve_review_runtime",
                      return_value=routed_runtime), \
         patch("threading.Thread", _SyncThread):
        agent_stub._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    init_kwargs = captured.get("init_kwargs", {})
    assert "reasoning_config" not in init_kwargs, (
        f"Routed review fork was passed the parent's reasoning_config "
        f"({init_kwargs.get('reasoning_config')!r}). On the routed path the "
        "cache is cold (no parity benefit) and the parent's effort value may "
        "be invalid for the routed model/provider — it must be omitted so "
        "the fork uses provider defaults."
    )
