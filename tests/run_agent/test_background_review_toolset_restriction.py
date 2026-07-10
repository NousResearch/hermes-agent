"""Tests that the background review agent restricts tools at runtime, not at schema time.

Regression coverage for issue #15204 (the background skill-review agent must
not perform non-skill side effects like terminal, send_message, delegate_task)
combined with issue #25322 / PR #17276 (the review fork must hit the parent's
Anthropic/OpenRouter prefix cache).

Reconciling the two: the fork now inherits the parent's full ``tools`` schema
so the cache-key matches, and enforces the memory+skills restriction at
runtime via a thread-local whitelist on the existing
``get_pre_tool_call_block_message`` gate. Safety is preserved mechanically
(any non-whitelisted dispatch is blocked) without the schema-level narrowing
that caused the prefix-cache miss.
"""

from types import SimpleNamespace
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
    agent._cached_system_prompt = None
    import datetime as _dt
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    # Non-None so the test catches a missing-kwarg regression.
    agent.enabled_toolsets = ["memory", "skills", "terminal"]
    agent.disabled_toolsets = ["spotify", "feishu_doc"]
    return agent


class _SyncThread:
    """Drop-in replacement for threading.Thread that runs the target inline."""

    def __init__(self, *, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def test_background_review_matches_parent_toolset_config():
    """Fork must receive parent's toolset config so ``tools[]`` cache key matches."""
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)
    captured = {}

    def _capture_init(self, *args, **kwargs):
        captured["enabled_toolsets"] = kwargs.get("enabled_toolsets", "UNSET")
        captured["disabled_toolsets"] = kwargs.get("disabled_toolsets", "UNSET")
        raise RuntimeError("stop after capturing init args")

    with patch.object(run_agent.AIAgent, "__init__", _capture_init), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "enabled_toolsets" in captured, "AIAgent.__init__ was not called"
    assert captured["enabled_toolsets"] == agent.enabled_toolsets, (
        f"enabled_toolsets mismatch: {captured['enabled_toolsets']!r} "
        f"vs expected {agent.enabled_toolsets!r}"
    )
    assert captured["disabled_toolsets"] == agent.disabled_toolsets, (
        f"disabled_toolsets mismatch: {captured['disabled_toolsets']!r} "
        f"vs expected {agent.disabled_toolsets!r}"
    )


def test_background_review_installs_thread_local_whitelist():
    """The review fork must install a memory/skills-only thread-local whitelist.

    The schema-level toolset narrowing was lifted (for prefix-cache parity),
    so #15204's safety contract now relies on the runtime whitelist gate to
    deny terminal/send_message/delegate_task at dispatch time. Verify the
    whitelist is set with exactly the memory+skills tool names.
    """
    import run_agent
    from hermes_cli import plugins as _plugins

    captured = {}

    def _capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)
        captured["deny_msg_fmt"] = deny_msg_fmt
        # Stop here — we just want to see what gets installed.
        raise RuntimeError("stop after capturing whitelist")

    agent = _make_agent_stub(run_agent.AIAgent)

    def _no_init(self, *args, **kwargs):
        # Don't crash AIAgent.__init__; let execution flow reach
        # set_thread_tool_whitelist.
        return None

    with patch.object(run_agent.AIAgent, "__init__", _no_init), \
         patch.object(_plugins, "set_thread_tool_whitelist", _capture_whitelist), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "whitelist" in captured, "set_thread_tool_whitelist was not called"
    whitelist = captured["whitelist"]
    # memory + skills tools must be allowed
    assert "memory" in whitelist
    assert "skill_manage" in whitelist
    assert "skill_view" in whitelist
    assert "skills_list" in whitelist
    # dangerous tools must NOT be in the whitelist
    assert "terminal" not in whitelist
    assert "send_message" not in whitelist
    assert "delegate_task" not in whitelist
    assert "web_search" not in whitelist
    assert "execute_code" not in whitelist


def test_background_review_agent_tools_are_limited():
    """Verify the resolved memory+skills toolsets only contain memory and skill tools.

    Sanity check on the source of truth for what the runtime whitelist is
    derived from — if a future PR adds e.g. `terminal` to the `memory`
    toolset, the review-fork safety contract silently breaks.
    """
    from toolsets import resolve_multiple_toolsets

    expected_tools = set(resolve_multiple_toolsets(["memory", "skills"]))

    assert "memory" in expected_tools
    assert "skill_manage" in expected_tools
    assert "skill_view" in expected_tools
    assert "skills_list" in expected_tools

    assert "terminal" not in expected_tools
    assert "send_message" not in expected_tools
    assert "delegate_task" not in expected_tools
    assert "web_search" not in expected_tools
    assert "execute_code" not in expected_tools


def test_background_review_excludes_memory_when_disabled():
    """A memory-disabled profile must NOT get the memory tool in the review fork.

    Regression for #54937 layer 2: the whitelist hardcoded ["memory", "skills"],
    so a skill-review fork on a profile with memory_enabled=false still granted
    the LLM the MEMORY.md read/write tool, contaminating a profile that opted
    out of built-in memory. The whitelist must gate "memory" on the flag.
    """
    import run_agent
    from hermes_cli import plugins as _plugins

    captured = {}

    def _capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)
        raise RuntimeError("stop after capturing whitelist")

    agent = _make_agent_stub(run_agent.AIAgent)
    agent._memory_enabled = False
    agent._user_profile_enabled = False

    def _no_init(self, *args, **kwargs):
        return None

    with patch.object(run_agent.AIAgent, "__init__", _no_init), \
         patch.object(_plugins, "set_thread_tool_whitelist", _capture_whitelist), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=False,
            review_skills=True,
        )

    whitelist = captured["whitelist"]
    # Skill tools still allowed...
    assert "skill_manage" in whitelist
    assert "skill_view" in whitelist
    # ...but the built-in memory tool must be gated out.
    assert "memory" not in whitelist


def test_background_review_includes_memory_when_user_profile_enabled():
    """user_profile_enabled alone (USER.md) still needs the memory tool."""
    import run_agent
    from hermes_cli import plugins as _plugins

    captured = {}

    def _capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)
        raise RuntimeError("stop after capturing whitelist")

    agent = _make_agent_stub(run_agent.AIAgent)
    agent._memory_enabled = False
    agent._user_profile_enabled = True

    def _no_init(self, *args, **kwargs):
        return None

    with patch.object(run_agent.AIAgent, "__init__", _no_init), \
         patch.object(_plugins, "set_thread_tool_whitelist", _capture_whitelist), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "memory" in captured["whitelist"]


def test_background_review_interrupt_skips_spawn():
    """An interrupted parent turn must not even spawn the review worker."""
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)
    agent._interrupt_requested = True

    class _ExplodingThread:
        def __init__(self, *args, **kwargs):
            raise AssertionError("background review thread should not be created")

    with patch("run_agent.threading.Thread", _ExplodingThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )


def test_cancelled_background_review_skips_model_work(monkeypatch):
    """A cancelled review worker must stop before review model/tool execution."""
    import run_agent
    from agent.background_review import _BackgroundReviewCancellation, _run_review_in_thread

    agent = _make_agent_stub(run_agent.AIAgent)
    agent._interrupt_requested = False
    agent._background_review_shutdown_requested = False
    agent._credential_pool = None
    agent._safe_print = lambda *_a, **_k: None
    agent.memory_notifications = "on"

    token = _BackgroundReviewCancellation()
    token.cancel("runtime shutdown")

    monkeypatch.setattr("agent.background_review.thread_scoped_silence", lambda: _NullContext())
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kwargs: [])
    monkeypatch.setattr("hermes_cli.plugins.set_thread_tool_whitelist", lambda *_a, **_k: None)
    monkeypatch.setattr("hermes_cli.plugins.clear_thread_tool_whitelist", lambda: None)
    monkeypatch.setattr("tools.terminal_tool.set_approval_callback", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent.AIAgent, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(run_agent.AIAgent, "shutdown_memory_provider", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(run_agent.AIAgent, "close", lambda self, *args, **kwargs: None)

    def _unexpected_run(*_args, **_kwargs):
        raise AssertionError("cancelled background review should not run model/tool work")

    monkeypatch.setattr(run_agent.AIAgent, "run_conversation", _unexpected_run)

    _run_review_in_thread(
        agent,
        messages_snapshot=[{"role": "user", "content": "hi"}],
        prompt="review memory",
        cancellation=token,
    )


def test_cancelling_registered_background_review_interrupts_active_worker():
    """Parent cancellation reaches a registered review agent already in flight."""
    from agent.background_review import (
        _register_background_review_cancellation,
        _unregister_background_review_cancellation,
        cancel_background_reviews,
    )

    parent = SimpleNamespace(
        _interrupt_requested=False,
        _background_review_shutdown_requested=False,
        _runtime_shutdown_requested=False,
    )
    token = _register_background_review_cancellation(parent)

    class ReviewAgent:
        reason = None

        def interrupt(self, reason):
            self.reason = reason

    review_agent = ReviewAgent()
    token.attach_review_agent(review_agent)
    try:
        cancel_background_reviews(parent, reason="parent interrupt")
        assert token.is_cancelled()
        assert review_agent.reason == "parent interrupt"
    finally:
        _unregister_background_review_cancellation(parent, token)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
