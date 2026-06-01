"""Issue #18871: background review fork must inherit the parent's
reasoning_config / service_tier / request_overrides so it does not silently
fall back to a transport-local ``medium`` default.

These tests mock ``AIAgent.__init__`` to capture the kwargs that
``_spawn_background_review`` passes when constructing the review agent,
then assert the parent runtime fields flow through verbatim (or, when the
parent has none, that the spawn passes None rather than a transport
default). The point is to lock the contract — if a future refactor
re-introduces the field omission, these tests fail loud.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


def _import_background_review():
    """Import the module under test fresh — guards against stale module
    state from earlier tests in the same session."""
    if "agent.background_review" in sys.modules:
        return sys.modules["agent.background_review"]
    return importlib.import_module("agent.background_review")


def _make_parent_agent(
    *,
    reasoning_config: Dict[str, Any] | None = None,
    service_tier: str | None = None,
    request_overrides: Dict[str, Any] | None = None,
) -> MagicMock:
    """Build a parent AIAgent mock with the fields _spawn_background_review
    reads via getattr()."""
    parent = MagicMock(spec=[])  # bare spec — getattr on missing returns the default
    parent.model = "test-model"
    parent.platform = None
    parent.provider = "test"
    parent.session_id = "parent-session-1"
    parent._credential_pool = None
    parent._memory_store = None
    parent._memory_enabled = False
    parent._user_profile_enabled = False
    parent._cached_system_prompt = None
    parent.session_start = None
    parent.reasoning_config = reasoning_config
    parent.service_tier = service_tier
    parent.request_overrides = request_overrides
    parent.enabled_toolsets = ["web", "terminal"]
    parent.disabled_toolsets = None
    return parent


@pytest.fixture
def captured_spawn_kwargs():
    """Patch AIAgent so we can capture the kwargs passed by
    _spawn_background_review without actually running the agent.

    The review spawn does ``from run_agent import AIAgent`` *inside* the
    function to dodge a circular import at module-load, so we patch
    ``run_agent.AIAgent`` (the import resolves against this name).
    """
    import run_agent
    captured: Dict[str, Any] = {}

    def _fake_ai_agent(*args, **kwargs):
        captured.update(kwargs)
        instance = MagicMock()
        instance._memory_write_origin = None
        instance._memory_write_context = None
        instance._memory_store = None
        instance._memory_enabled = False
        instance._user_profile_enabled = False
        instance._memory_nudge_interval = 0
        instance._skill_nudge_interval = 0
        instance.suppress_status_output = False
        instance._cached_system_prompt = None
        instance.session_id = "parent-session-1"
        instance.session_start = None
        # Mirror the spawn's after-init attributes so the rest of the
        # function can read them without blowing up.
        instance.reasoning_config = kwargs.get("reasoning_config")
        instance.service_tier = kwargs.get("service_tier")
        instance.request_overrides = kwargs.get("request_overrides")
        # Short-circuit the conversation step so the test only exercises
        # the constructor kwargs, not the full review loop.
        instance.run_conversation.side_effect = SystemExit(
            "test-short-circuit"
        )
        return instance

    with patch.object(run_agent, "AIAgent", side_effect=_fake_ai_agent):
        yield captured


def _call_spawn(parent: MagicMock, captured: Dict[str, Any]) -> None:
    """Invoke _run_review_in_thread with a minimal viable parent and let
    it run far enough to construct the review AIAgent. We don't care
    about the review's behaviour after spawn — only the captured kwargs.
    """
    bg = _import_background_review()
    # _run_review_in_thread reads runtime via agent._current_main_runtime()
    # then constructs an AIAgent and calls .run_conversation(). We short-
    # circuit the conversation step so the test only exercises the kwargs
    # the review agent was constructed with.
    parent._current_main_runtime = MagicMock(
        return_value={"base_url": None, "api_key": None, "api_mode": None}
    )
    parent._safe_print = MagicMock()
    parent.background_review_callback = None
    # We rely on the fake_ai_agent fixture to short-circuit construction.
    # After construction, the function calls run_conversation — patch
    # MagicMock.run_conversation to raise so we exit the thread cleanly
    # without running real LLM calls.
    def _raise(*_a, **_kw):
        raise SystemExit("test-short-circuit")
    # Wire run_conversation on the MagicMock subclass that _fake_ai_agent
    # returns. Since the fixture sets instance.run_conversation, we can
    # simply check captured and not worry about it.
    captured["_expected_run_conversation"] = True
    try:
        bg._run_review_in_thread(
            parent,
            messages_snapshot=[],
            prompt="test prompt",
        )
    except SystemExit:
        # Raised by run_conversation to bail out of the thread
        pass
    except Exception as e:
        # If run_conversation was called, it succeeded (MagicMock returns
        # another MagicMock). The function will continue to
        # shutdown_memory_provider() / close() / _safe_print. Those are
        # all MagicMock no-ops, so we shouldn't see errors — but if we
        # do, re-raise for visibility.
        if "test-short-circuit" not in str(e):
            raise


class TestReasoningConfigInheritance:
    """Issue #18871: background review must inherit the parent's runtime
    reasoning/transport config rather than falling back to defaults."""

    def test_reasoning_config_inherited_when_parent_sets_xhigh(
        self, captured_spawn_kwargs
    ):
        parent_cfg = {"effort": "xhigh", "enabled": True}
        parent = _make_parent_agent(reasoning_config=parent_cfg)
        _call_spawn(parent, captured_spawn_kwargs)
        assert captured_spawn_kwargs.get("reasoning_config") == parent_cfg, (
            "Review fork must inherit parent's reasoning_config verbatim — "
            "otherwise it falls back to transport-medium on Codex Responses."
        )

    def test_reasoning_config_inherited_when_parent_disables_thinking(
        self, captured_spawn_kwargs
    ):
        """Critical case from issue #18871: parent disables reasoning but
        a fresh AIAgent without reasoning_config falls back to medium.
        """
        parent_cfg = {"enabled": False, "effort": "none"}
        parent = _make_parent_agent(reasoning_config=parent_cfg)
        _call_spawn(parent, captured_spawn_kwargs)
        assert captured_spawn_kwargs.get("reasoning_config") == parent_cfg

    def test_service_tier_inherited(self, captured_spawn_kwargs):
        parent = _make_parent_agent(service_tier="auto")
        _call_spawn(parent, captured_spawn_kwargs)
        assert captured_spawn_kwargs.get("service_tier") == "auto"

    def test_request_overrides_inherited(self, captured_spawn_kwargs):
        parent_overrides = {"temperature": 0.2, "top_p": 0.95}
        parent = _make_parent_agent(request_overrides=parent_overrides)
        _call_spawn(parent, captured_spawn_kwargs)
        assert captured_spawn_kwargs.get("request_overrides") == parent_overrides

    def test_all_three_inherited_together(self, captured_spawn_kwargs):
        """A parent carrying all three fields passes all three through."""
        parent = _make_parent_agent(
            reasoning_config={"effort": "high"},
            service_tier="default",
            request_overrides={"timeout": 30},
        )
        _call_spawn(parent, captured_spawn_kwargs)
        assert captured_spawn_kwargs["reasoning_config"] == {"effort": "high"}
        assert captured_spawn_kwargs["service_tier"] == "default"
        assert captured_spawn_kwargs["request_overrides"] == {"timeout": 30}

    def test_none_when_parent_has_none(self, captured_spawn_kwargs):
        """Parent has no fields set — review sees None, NOT a transport default.

        This is what was broken before the fix: a fresh AIAgent with
        reasoning_config=None would later resolve to medium on Codex
        routes, which is the bug #18871 reports.
        """
        parent = _make_parent_agent(
            reasoning_config=None, service_tier=None, request_overrides=None
        )
        _call_spawn(parent, captured_spawn_kwargs)
        # Must be present in kwargs (we explicitly pass it), not absent
        # — otherwise the constructor falls back to the AIAgent default
        # rather than the parent's intent of "no override".
        assert "reasoning_config" in captured_spawn_kwargs
        assert captured_spawn_kwargs["reasoning_config"] is None
        assert "service_tier" in captured_spawn_kwargs
        assert captured_spawn_kwargs["service_tier"] is None
        assert "request_overrides" in captured_spawn_kwargs
        assert captured_spawn_kwargs["request_overrides"] is None

    def test_enabled_toolsets_still_inherited(self, captured_spawn_kwargs):
        """Sanity check: pre-existing fields are still cloned. The new
        fields must not regress the old ones."""
        parent = _make_parent_agent()
        parent.enabled_toolsets = ["web", "terminal", "memory", "skills"]
        _call_spawn(parent, captured_spawn_kwargs)
        assert captured_spawn_kwargs["enabled_toolsets"] == [
            "web", "terminal", "memory", "skills"
        ]
