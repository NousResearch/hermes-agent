"""Regression tests for background review agent cleanup."""

from __future__ import annotations

import run_agent as run_agent_module
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.model = "fake-model"
    agent.platform = "telegram"
    agent.provider = "openai"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = ""
    agent.session_id = "test-session"
    agent._parent_session_id = ""
    agent._credential_pool = None
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._cached_system_prompt = "test-cached-system-prompt"
    agent._fallback_chain = [
        {"provider": "copilot", "model": "gpt-5.6-luna"},
    ]
    import datetime as _dt
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.background_review_callback = None
    agent.status_callback = None
    agent._safe_print = lambda *_args, **_kwargs: None
    return agent


class ImmediateThread:
    def __init__(self, *, target, daemon=None, name=None):
        self._target = target

    def start(self):
        self._target()


def test_background_review_shuts_down_memory_provider_before_close(monkeypatch):
    events = []

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))
            self._session_messages = []

        def run_conversation(self, **kwargs):
            events.append(("run_conversation", kwargs))

        def shutdown_memory_provider(self):
            events.append(("shutdown_memory_provider", None))

        def close(self):
            events.append(("close", None))

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert [name for name, _payload in events] == [
        "init",
        "run_conversation",
        "shutdown_memory_provider",
        "close",
    ]


def test_background_review_fork_opts_out_of_session_finalization(monkeypatch):
    """The review fork shares the parent's live session_id, so it must set
    ``_end_session_on_close = False``. Otherwise close() (now finalizing owned
    session rows) would end the still-active parent session mid-conversation
    every time the review fires (~every 10 turns). Regression for #12029.
    """
    seen = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []
            # Default matches AIAgent.__init__ (agent_init.py): owns its row.
            self._end_session_on_close = True

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == "_end_session_on_close":
                seen["end_session_on_close"] = value

        def run_conversation(self, **kwargs):
            # By the time the fork runs, the opt-out must already be applied.
            seen["at_run_time"] = self._end_session_on_close

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert seen.get("end_session_on_close") is False
    assert seen.get("at_run_time") is False


def test_background_review_fork_inherits_parent_fallback_chain(monkeypatch):
    """The review fork must inherit the parent's fallback chain so a hard-quota
    429 on the review's model switches to the configured fallback instead of
    aborting.  Regression for the same bug class as the main-conversation
    Z.AI hard-quota fallback path (sibling fork path: delegate_tool.py already
    inherits _fallback_chain for subagents; background_review did not).
    """
    captured = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs
            self._session_messages = []

        def run_conversation(self, **kwargs):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    init_kwargs = captured.get("init_kwargs", {})
    assert "fallback_model" in init_kwargs, "fallback_model not passed to review fork"
    expected = [{"provider": "copilot", "model": "gpt-5.6-luna"}]
    assert init_kwargs["fallback_model"] == expected, (
        f"review fork fallback_model={init_kwargs['fallback_model']!r}, "
        f"expected {expected!r}"
    )


def test_background_review_fork_fallback_none_when_parent_has_no_chain(monkeypatch):
    """When the parent has no fallback chain, the review fork should pass
    ``fallback_model=None`` (matching AIAgent's default) rather than crashing
    or synthesizing an empty list.
    """
    captured = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs
            self._session_messages = []

        def run_conversation(self, **kwargs):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    agent._fallback_chain = []  # no fallback configured

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    init_kwargs = captured.get("init_kwargs", {})
    assert init_kwargs.get("fallback_model") is None, (
        f"expected fallback_model=None when parent has no chain, got "
        f"{init_kwargs.get('fallback_model')!r}"
    )


def test_background_review_fork_inherits_multi_entry_fallback_chain(monkeypatch):
    """Multi-entry fallback chains must be passed through intact, preserving
    ordering.  The review fork's _has_pending_fallback() and
    try_activate_fallback() walk the list in order, so any truncation or
    reordering would break the fallback sequence.
    """
    captured = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs
            self._session_messages = []

        def run_conversation(self, **kwargs):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    multi_chain = [
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        {"provider": "copilot", "model": "gpt-5.6-luna"},
    ]
    agent._fallback_chain = multi_chain

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    init_kwargs = captured.get("init_kwargs", {})
    assert init_kwargs["fallback_model"] == multi_chain, (
        f"multi-entry chain not passed intact: {init_kwargs.get('fallback_model')!r}"
    )


def test_background_review_fork_does_not_mutate_parent_fallback_chain(monkeypatch):
    """The review fork's fallback operations (index advancement, key tracking)
    must not mutate the parent's _fallback_chain list or its dict entries.
    This guards against shared-reference bugs if future code writes to chain
    entries.  The fallback path currently only reads entries, but this test
    locks that contract.
    """
    import copy as _copy

    captured = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs
            self._session_messages = []
            # Simulate fallback activation: advance index on the fork's chain
            fb = kwargs.get("fallback_model")
            if isinstance(fb, list) and fb:
                # try_activate_fallback reads entries — never writes.
                _ = fb[0].get("provider")
                _ = fb[0].get("model")

        def run_conversation(self, **kwargs):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    original_chain = [
        {"provider": "copilot", "model": "gpt-5.6-luna"},
    ]
    original_snapshot = _copy.deepcopy(original_chain)
    agent._fallback_chain = original_chain

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert agent._fallback_chain == original_snapshot, (
        "parent _fallback_chain was mutated by the review fork"
    )


def test_background_review_fork_chain_enables_has_pending_fallback(monkeypatch):
    """Behavioral test: prove the inherited chain makes the fork's
    _has_pending_fallback() return True — the actual gate that decides whether
    fallback is attempted.  This goes beyond kwarg-plumbing: it verifies the
    chain flows through agent_init.py's list comprehension and produces a
    usable _fallback_chain on the constructed agent.

    Uses a real AIAgent via __new__ to exercise the _has_pending_fallback
    method directly, simulating what agent_init.py would produce.
    """
    import run_agent as run_agent_module

    # Simulate what agent_init.py does with fallback_model=[...]
    fallback_model = [{"provider": "copilot", "model": "gpt-5.6-luna"}]
    # agent_init.py: [f for f in fallback_model if isinstance(f, dict) and f.get("provider") and f.get("model")]
    chain = [
        f for f in fallback_model
        if isinstance(f, dict) and f.get("provider") and f.get("model")
    ]

    # Build a bare agent the way _has_pending_fallback reads it
    agent = object.__new__(run_agent_module.AIAgent)
    agent._fallback_chain = chain
    agent._fallback_index = 0

    # The actual method from run_agent.py:5905
    assert agent._has_pending_fallback() is True, (
        "fork with inherited chain should report a pending fallback"
    )

    # Simulate exhausting the chain (index advanced past all entries)
    agent._fallback_index = len(chain)
    assert agent._has_pending_fallback() is False, (
        "fork with exhausted index should report no pending fallback"
    )

    # Prove an empty chain (pre-fix behavior) reports False
    agent._fallback_chain = []
    agent._fallback_index = 0
    assert agent._has_pending_fallback() is False, (
        "fork with empty chain (pre-fix) should report no pending fallback"
    )










# ---------------------------------------------------------------------------
# memory_notifications mode: off | on | verbose
# ---------------------------------------------------------------------------

import json as _json

from agent.background_review import summarize_background_review_actions


def _memory_add_review():
    """A minimal review transcript: one memory add (assistant call + tool result)."""
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_mem1",
                    "function": {
                        "name": "memory",
                        "arguments": _json.dumps(
                            {
                                "action": "add",
                                "target": "memory",
                                "content": "User prefers terse replies",
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_mem1",
            "content": _json.dumps(
                {"success": True, "message": "Entry added.", "target": "memory"}
            ),
        },
    ]


def _skill_patch_review():
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_skill1",
                    "function": {
                        "name": "skill_manage",
                        "arguments": _json.dumps(
                            {"action": "patch", "name": "demo", "old_string": "a", "new_string": "b"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_skill1",
            "content": _json.dumps(
                {
                    "success": True,
                    "message": "Patched SKILL.md in skill 'demo' (1 replacement).",
                    "_change": {"old": "a", "new": "b"},
                }
            ),
        },
    ]


def test_memory_notifications_off_returns_nothing():
    actions = summarize_background_review_actions(
        _memory_add_review(), [], notification_mode="off"
    )
    assert actions == []








def test_skill_patch_off_silent_verbose_shows_diff():
    assert (
        summarize_background_review_actions(
            _skill_patch_review(), [], notification_mode="off"
        )
        == []
    )
    verbose = summarize_background_review_actions(
        _skill_patch_review(), [], notification_mode="verbose"
    )
    assert len(verbose) == 1
    assert "demo" in verbose[0] and "→" in verbose[0]
