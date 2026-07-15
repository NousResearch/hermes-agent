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


# ---------------------------------------------------------------------------
# Lifecycle tests: config parsing → set/clear thread-local protection
# Exercises the config-to-guard path that _run_review_in_thread uses.
# ---------------------------------------------------------------------------

def _parse_protection_config(raw_value):
    """Replicate the config parsing logic from _run_review_in_thread."""
    import json as _json

    protected_names = set()
    parse_failed = False

    _raw = raw_value
    if isinstance(_raw, str):
        try:
            _raw = _json.loads(_raw)
        except _json.JSONDecodeError:
            parse_failed = True
            _raw = None
    if isinstance(_raw, list):
        protected_names = {str(n).strip() for n in _raw if str(n).strip()}
    elif _raw is None or _raw == []:
        pass
    else:
        parse_failed = True

    return protected_names, parse_failed


class TestReviewProtectedConfigParsing:
    """Test config parsing logic that drives the protection lifecycle."""

    def test_yaml_list(self):
        names, failed = _parse_protection_config(["skill-a", "skill-b"])
        assert not failed
        assert names == {"skill-a", "skill-b"}

    def test_json_array_string(self):
        names, failed = _parse_protection_config('["skill-a", "skill-b"]')
        assert not failed
        assert names == {"skill-a", "skill-b"}

    def test_empty_list(self):
        names, failed = _parse_protection_config([])
        assert not failed
        assert names == set()

    def test_none_value(self):
        names, failed = _parse_protection_config(None)
        assert not failed
        assert names == set()

    def test_whitespace_trimmed(self):
        names, failed = _parse_protection_config(["  spaced  "])
        assert not failed
        assert names == {"spaced"}

    def test_scalar_string_malformed(self):
        names, failed = _parse_protection_config("my-skill")
        assert failed

    def test_dict_malformed(self):
        names, failed = _parse_protection_config({"key": "value"})
        assert failed

    def test_int_malformed(self):
        names, failed = _parse_protection_config(42)
        assert failed

    def test_invalid_json_string(self):
        names, failed = _parse_protection_config('not json')
        assert failed


class TestReviewProtectedLifecycle:
    """Test the set/clear lifecycle as _run_review_in_thread uses it."""

    def test_protection_set_then_cleared(self):
        """Simulate: config loads, set is called, review runs, clear in finally."""
        from tools.skill_manager_tool import (
            set_review_protected_skills,
            clear_review_protected_skills,
            _review_protected_guard,
        )

        clear_review_protected_skills()
        # Simulate config load
        names, failed = _parse_protection_config(["important-skill"])
        assert not failed

        try:
            set_review_protected_skills(names or None)
            # During review, writes are blocked
            assert _review_protected_guard("patch", "important-skill") is not None
        finally:
            clear_review_protected_skills()

        # After review, writes are allowed
        assert _review_protected_guard("patch", "important-skill") is None

    def test_fail_closed_on_malformed(self):
        """Simulate: malformed config → deny-all → review runs → clear."""
        from tools.skill_manager_tool import (
            set_review_protected_skills,
            clear_review_protected_skills,
            _review_protected_guard,
        )

        clear_review_protected_skills()
        names, failed = _parse_protection_config("my-skill")
        assert failed

        try:
            set_review_protected_skills({"*"})
            # During review, ALL writes blocked
            assert _review_protected_guard("patch", "any") is not None
            assert _review_protected_guard("edit", "other") is not None
        finally:
            clear_review_protected_skills()

        assert _review_protected_guard("patch", "any") is None

    def test_empty_config_no_protection(self):
        """Simulate: empty config → no protection."""
        from tools.skill_manager_tool import (
            set_review_protected_skills,
            clear_review_protected_skills,
            _review_protected_guard,
        )

        clear_review_protected_skills()
        names, failed = _parse_protection_config([])
        assert not failed

        try:
            set_review_protected_skills(names or None)
            assert _review_protected_guard("patch", "any") is None
        finally:
            clear_review_protected_skills()

    def test_exception_during_review_still_cleans_up(self):
        """Simulate: exception in run_conversation → finally clears protection."""
        from tools.skill_manager_tool import (
            set_review_protected_skills,
            clear_review_protected_skills,
            _review_protected_guard,
        )

        clear_review_protected_skills()

        try:
            set_review_protected_skills({"blocked-skill"})
            raise RuntimeError("simulated review crash")
        except RuntimeError:
            pass
        finally:
            clear_review_protected_skills()

        assert _review_protected_guard("patch", "blocked-skill") is None
