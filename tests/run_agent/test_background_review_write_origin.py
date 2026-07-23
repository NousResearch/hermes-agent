"""Constructor-path regression tests for review-fork memory-write markers.

The background review fork must carry ``_memory_write_origin`` /
``_memory_write_context`` = "background_review" BEFORE ``AIAgent.__init__``
runs: the context engine's ``on_session_start`` fires DURING init (from
agent_init), and downstream engines (e.g. hermes-lcm) detect review forks by
walking the call stack for a frame whose agent has
``_memory_write_origin == "background_review"``. Set only after construction,
the marker is invisible at that moment and the engine binds the fork's fresh
session_id as a real session.

The suites in test_background_review.py / test_background_review_cache_parity.py
use fake review agents and assert state after construction, so they cannot
catch a regression here. These tests construct real ``AIAgent`` objects and
probe the markers synchronously from ``on_session_start`` itself.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
from unittest.mock import patch

from agent.context_engine import ContextEngine


class _ProbeEngine(ContextEngine):
    """Context engine that inspects the constructing agent from on_session_start.

    Mirrors how a downstream engine (hermes-lcm) detects review forks: walk
    the Python call stack for a frame whose ``self``/``agent`` local is an
    AIAgent, then read the memory-write markers off it. on_session_start
    fires inside agent_init, i.e. mid-``__init__`` — whatever we observe
    here is exactly what a real engine would observe.
    """

    def __init__(self):
        super().__init__()
        self.observed: list[dict] = []

    @property
    def name(self) -> str:
        return "probe"

    def update_from_response(self, usage):
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None):
        return messages

    def on_session_start(self, session_id: str, **kwargs) -> None:
        import inspect

        from run_agent import AIAgent

        frame = inspect.currentframe()
        try:
            frame = frame.f_back
            while frame is not None:
                for local_name in ("self", "agent"):
                    candidate = frame.f_locals.get(local_name)
                    if isinstance(candidate, AIAgent):
                        self.observed.append(
                            {
                                "session_id": session_id,
                                "origin": getattr(
                                    candidate, "_memory_write_origin", None
                                ),
                                "context": getattr(
                                    candidate, "_memory_write_context", None
                                ),
                            }
                        )
                        return
                frame = frame.f_back
        finally:
            del frame
        self.observed.append(
            {"session_id": session_id, "origin": None, "context": None}
        )


def _parent_stub():
    """Minimal parent agent for spawn_background_review_thread.

    Same shape as _bare_agent() in test_background_review.py, plus
    _current_main_runtime (consumed by _resolve_review_runtime).
    """
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.model = "openrouter/auto"
    agent.platform = "cli"
    agent.provider = "openai"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "test-key-1234567890"
    agent.api_mode = ""
    agent.session_id = "parent-session"
    agent._parent_session_id = ""
    agent._credential_pool = None
    agent._memory_store = object()
    agent._memory_enabled = False
    agent._user_profile_enabled = False
    agent._cached_system_prompt = "test-cached-system-prompt"
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.background_review_callback = None
    agent.status_callback = None
    agent._safe_print = lambda *_args, **_kwargs: None
    agent._current_main_runtime = lambda: {
        "api_key": "test-key-1234567890",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": None,
    }
    return agent


@contextlib.contextmanager
def _probe_engine_installed(engine, cfg):
    """Route AIAgent init through the probe engine (offline, no tools)."""
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        yield


def test_review_fork_marker_visible_during_init(monkeypatch, tmp_path):
    """The real review fork must expose the markers to on_session_start.

    Drives the actual _ReviewAgent construction in
    agent/background_review.py's _run_review_in_thread (synchronously, with
    run_conversation stubbed out) and asserts the probe engine saw
    "background_review" for BOTH markers while __init__ was still running.
    """
    import agent.background_review as bg_review
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    engine = _ProbeEngine()
    cfg = {"context": {"engine": "probe"}, "agent": {}}

    failures: list = []
    run_calls: list = []

    parent = _parent_stub()
    parent._emit_auxiliary_failure = lambda *a, **kw: failures.append(a)

    with (
        _probe_engine_installed(engine, cfg),
        patch.object(
            AIAgent,
            "run_conversation",
            lambda self, **kwargs: run_calls.append(kwargs),
        ),
    ):
        target, _prompt = bg_review.spawn_background_review_thread(
            parent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
        )
        target()  # synchronous: no thread, exceptions surface via failures[]

    assert not failures, f"background review failed: {failures}"
    assert run_calls, "review fork never reached run_conversation"
    assert len(engine.observed) == 1, engine.observed
    seen = engine.observed[0]
    assert seen["origin"] == "background_review", (
        "on_session_start fired during the review fork's __init__ but the "
        f"agent's _memory_write_origin was {seen['origin']!r} — a downstream "
        "engine would bind the fork's session as a real session and "
        "bulk-ingest the parent conversation snapshot."
    )
    assert seen["context"] == "background_review", seen


def test_normal_construction_keeps_default_markers(monkeypatch, tmp_path):
    """A plain AIAgent keeps the default markers, seen from on_session_start too."""
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    engine = _ProbeEngine()
    cfg = {"context": {"engine": "probe"}, "agent": {}}

    with _probe_engine_installed(engine, cfg):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    # Defaults after construction.
    assert agent._memory_write_origin == "assistant_tool"
    assert agent._memory_write_context == "foreground"

    # And the probe saw the same defaults DURING __init__ — honoring
    # pre-seeded values must not leave a normal agent unmarked.
    assert len(engine.observed) == 1, engine.observed
    seen = engine.observed[0]
    assert seen["origin"] == "assistant_tool", seen
    assert seen["context"] == "foreground", seen
