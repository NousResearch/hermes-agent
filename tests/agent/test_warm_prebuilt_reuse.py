"""Tests for prompt-warmer prebuild reuse in _restore_or_build_system_prompt.

The warm hook (agent_init) builds the session preamble
once, warms the server with it, and stores it on the agent. The first
turn's fresh-build path reuses that exact string — gated on the same
runtime-identity check the DB-restore path trusts — so the warmed prefix
provably matches what the session sends. The prebuild is consumed
one-shot; drift or an explicit system_message falls back to a fresh
build.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

def _restore_or_build_system_prompt(*args, **kwargs):
    # Late-bound: resolve through sys.modules at call time. Some suite
    # tests (test_empty_tool_name_loop_dampening) purge agent.* from
    # sys.modules mid-run; a top-level `from ... import` here would keep
    # calling the pre-purge module object while mock.patch targets the
    # re-imported one, so the drift-gate patch would silently miss.
    from agent.conversation_loop import (
        _restore_or_build_system_prompt as _impl,
    )
    return _impl(*args, **kwargs)


def _make_agent(prebuilt=None):
    agent = MagicMock()
    agent._cached_system_prompt = None
    agent.session_id = "test-session-id"
    agent.model = "qwen-small"
    agent.provider = "llamacpp"
    agent.platform = "cli"
    agent._session_db = None
    agent._use_prompt_caching = False
    agent._build_system_prompt = MagicMock(return_value="FRESH_BUILD")
    agent._warm_prebuilt_system_prompt = prebuilt
    return agent


class TestWarmPrebuiltReuse:
    def test_new_session_reuses_prebuilt_verbatim(self):
        agent = _make_agent(prebuilt="WARMED PREAMBLE")
        with patch(
            "agent.conversation_loop._stored_prompt_matches_runtime",
            return_value=True,
        ):
            _restore_or_build_system_prompt(agent, None, [])

        assert agent._cached_system_prompt == "WARMED PREAMBLE"
        agent._build_system_prompt.assert_not_called()
        # Consumed one-shot: a later compression-invalidation rebuild must
        # never resurrect the session-open build.
        assert agent._warm_prebuilt_system_prompt is None

    def test_runtime_drift_falls_back_to_fresh_build(self):
        agent = _make_agent(prebuilt="WARMED PREAMBLE")
        with patch(
            "agent.conversation_loop._stored_prompt_matches_runtime",
            return_value=False,
        ):
            _restore_or_build_system_prompt(agent, None, [])

        assert agent._cached_system_prompt == "FRESH_BUILD"
        agent._build_system_prompt.assert_called_once_with(None)
        assert agent._warm_prebuilt_system_prompt is None

    def test_explicit_system_message_bypasses_prebuilt(self):
        agent = _make_agent(prebuilt="WARMED PREAMBLE")

        _restore_or_build_system_prompt(agent, "explicit system", [])

        assert agent._cached_system_prompt == "FRESH_BUILD"
        agent._build_system_prompt.assert_called_once_with("explicit system")
        assert agent._warm_prebuilt_system_prompt is None

    def test_non_string_prebuilt_is_ignored(self):
        agent = _make_agent(prebuilt=MagicMock())

        _restore_or_build_system_prompt(agent, None, [])

        assert agent._cached_system_prompt == "FRESH_BUILD"
        agent._build_system_prompt.assert_called_once_with(None)

    def test_absent_prebuilt_keeps_existing_behavior(self):
        agent = _make_agent(prebuilt=None)

        _restore_or_build_system_prompt(agent, None, [])

        assert agent._cached_system_prompt == "FRESH_BUILD"
        agent._build_system_prompt.assert_called_once_with(None)

    def test_stored_session_prompt_still_wins_over_prebuilt(self):
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": "STORED"}
        agent = _make_agent(prebuilt="WARMED PREAMBLE")
        agent._session_db = db

        with patch(
            "agent.conversation_loop._stored_prompt_matches_runtime",
            return_value=True,
        ):
            _restore_or_build_system_prompt(
                agent, None, [{"role": "user", "content": "hi"}]
            )

        assert agent._cached_system_prompt == "STORED"
        agent._build_system_prompt.assert_not_called()
