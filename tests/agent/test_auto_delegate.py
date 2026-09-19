#!/usr/bin/env python3
"""
Tests for agent.auto_delegate (opt-in dispatcher mode, delegation.auto_delegate).

Behavior contracts under test:
- Disabled by default; enabled only when delegation.auto_delegate is truthy.
- try_auto_delegate returns None whenever the mode is off or delegation
  fails — dispatcher mode never hard-fails a turn.
- On success it returns a terminal turn-result dict shaped like a normal
  run_conversation result (final_response + messages with normal role
  alternation: user message then one assistant summary).
- The parent message list is never mutated in place; a copy is returned.
- v0.21.3: the handoff is skipped inside a delegated child agent, and the
  result mirrors the keys ``turn_finalizer.finalize_turn`` emits.

Run with: python -m pytest tests/agent/test_auto_delegate.py -v
"""

import inspect
import json
import unittest
from unittest.mock import MagicMock, patch

from agent.auto_delegate import (
    _extract_first_summary,
    _load_delegation_config,
    auto_delegate_enabled,
    try_auto_delegate,
)


def _make_agent(**attrs):
    agent = MagicMock()
    # Real agents carry ``_delegate_depth`` (0 for a top-level agent); a bare
    # MagicMock attribute would be truthy and trip the child-context guard.
    agent._delegate_depth = 0
    for k, v in attrs.items():
        setattr(agent, k, v)
    return agent


class TestAutoDelegateConfig(unittest.TestCase):
    """delegation.auto_delegate is opt-in and defaults to False."""

    def test_disabled_by_default(self):
        with patch("agent.auto_delegate._load_delegation_config", return_value={}):
            self.assertFalse(auto_delegate_enabled())

    def test_disabled_when_explicit_false(self):
        with patch(
            "agent.auto_delegate._load_delegation_config",
            return_value={"auto_delegate": False},
        ):
            self.assertFalse(auto_delegate_enabled())

    def test_enabled_when_true(self):
        with patch(
            "agent.auto_delegate._load_delegation_config",
            return_value={"auto_delegate": True},
        ):
            self.assertTrue(auto_delegate_enabled())

    def test_enabled_accepts_yaml_truthy_strings(self):
        # YAML `yes`/`true` can arrive as strings via some merge paths.
        with patch(
            "agent.auto_delegate._load_delegation_config",
            return_value={"auto_delegate": "yes"},
        ):
            self.assertTrue(auto_delegate_enabled())

    def test_registered_in_default_config(self):
        # The knob must exist in DEFAULT_CONFIG (a reader without a registered
        # key is never shown/migrated) and default to False.
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        self.assertIn("auto_delegate", DEFAULT_CONFIG["delegation"])
        self.assertIs(DEFAULT_CONFIG["delegation"]["auto_delegate"], False)

    def test_reads_real_config_file(self):
        # E2E-ish: write a real config.yaml under the isolated HERMES_HOME
        # (tests/conftest.py points HERMES_HOME at a per-test tempdir) and
        # verify the shared loader path picks the key up.
        import os

        from hermes_cli.config import _LOAD_CONFIG_CACHE
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
        os.makedirs(hermes_home, exist_ok=True)
        config_path = os.path.join(hermes_home, "config.yaml")

        # 1) No config file -> disabled (default False).
        if os.path.exists(config_path):
            os.remove(config_path)
        _LOAD_CONFIG_CACHE.clear()
        try:
            self.assertFalse(auto_delegate_enabled())
        finally:
            _LOAD_CONFIG_CACHE.clear()

        # 2) Config file with delegation.auto_delegate: true -> enabled.
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("delegation:\n  auto_delegate: true\n")
        _LOAD_CONFIG_CACHE.clear()
        try:
            self.assertTrue(auto_delegate_enabled())
        finally:
            _LOAD_CONFIG_CACHE.clear()
            if os.path.exists(config_path):
                os.remove(config_path)


class TestExtractFirstSummary(unittest.TestCase):
    def test_ok_summary_extracted(self):
        payload = {
            "results": [{"status": "ok", "summary": "  done.  ", "task_index": 0}],
            "total_duration_seconds": 1.2,
        }
        self.assertEqual(_extract_first_summary(payload), "done.")

    def test_error_status_returns_none(self):
        payload = {"results": [{"status": "error", "summary": None}], "total_duration_seconds": 0.1}
        self.assertIsNone(_extract_first_summary(payload))

    def test_empty_results_returns_none(self):
        self.assertIsNone(_extract_first_summary({"results": []}))
        self.assertIsNone(_extract_first_summary({}))
        self.assertIsNone(_extract_first_summary(None))
        self.assertIsNone(_extract_first_summary("not a dict"))

    def test_blank_summary_returns_none(self):
        payload = {"results": [{"status": "ok", "summary": "   "}]}
        self.assertIsNone(_extract_first_summary(payload))


class TestTryAutoDelegate(unittest.TestCase):
    def test_returns_none_when_disabled(self):
        agent = _make_agent()
        with patch("agent.auto_delegate.auto_delegate_enabled", return_value=False):
            result = try_auto_delegate(agent, "do the thing", [{"role": "user", "content": "hi"}])
        self.assertIsNone(result)

    def test_returns_none_without_agent_or_message(self):
        with patch("agent.auto_delegate.auto_delegate_enabled", return_value=True):
            self.assertIsNone(try_auto_delegate(None, "task", []))
            self.assertIsNone(try_auto_delegate(_make_agent(), "", []))
            self.assertIsNone(try_auto_delegate(_make_agent(), "   ", []))
            self.assertIsNone(try_auto_delegate(_make_agent(), 12345, []))
            # Defaulted kwargs (loop-state handoff shape) are also guarded.
            self.assertIsNone(try_auto_delegate(_make_agent()))

    def test_skips_inside_delegated_child(self):
        # A nested handoff would burn budget re-dispatching the same goal; the
        # child's own turn must run normally.
        agent = _make_agent(_delegate_depth=1)
        with patch("agent.auto_delegate.auto_delegate_enabled", return_value=True), patch(
            "tools.delegate_tool.delegate_task"
        ) as mock_delegate:
            result = try_auto_delegate(agent, "task", [])
        self.assertIsNone(result)
        mock_delegate.assert_not_called()

    def test_dispatches_and_shapes_terminal_result(self):
        agent = _make_agent()
        summary = "Child finished: wrote /tmp/out.txt."
        ok_payload = json.dumps(
            {"results": [{"status": "ok", "summary": summary, "task_index": 0}]}
        )
        user_msg = {"role": "user", "content": "write a report"}
        with patch("agent.auto_delegate.auto_delegate_enabled", return_value=True), patch(
            "tools.delegate_tool.delegate_task", return_value=ok_payload
        ) as mock_delegate:
            result = try_auto_delegate(agent, "write a report", [user_msg], "task-1")

        self.assertIsNotNone(result)
        if result is None:  # pragma: no cover - unreachable, appeases checkers
            return
        self.assertEqual(result["final_response"], summary)
        self.assertTrue(result["completed"])
        self.assertFalse(result["failed"])
        self.assertFalse(result["interrupted"])
        self.assertFalse(result["partial"])
        self.assertTrue(result["auto_delegated"])
        self.assertEqual(result["turn_exit_reason"], "auto_delegated")
        # Aligned with turn_finalizer.finalize_turn's terminal-result keys.
        for key in ("messages", "api_calls", "model", "provider", "session_id", "last_reasoning"):
            self.assertIn(key, result)
        # Role alternation: exactly one assistant message appended after the
        # user message; the input list is not mutated in place.
        roles = [m["role"] for m in result["messages"]]
        self.assertEqual(roles, ["user", "assistant"])
        self.assertEqual(result["messages"][-1]["content"], summary)
        self.assertEqual(result["task_id"], "task-1")

        # delegate_task received the user message as a self-contained goal
        # plus the parent agent context.
        _, kwargs = mock_delegate.call_args
        self.assertEqual(kwargs["goal"], "write a report")
        self.assertIs(kwargs["parent_agent"], agent)
        self.assertIn("auto-delegated", kwargs["context"])

    def test_keyword_call_matches_loop_handoff(self):
        # The conversation_loop call site passes the _LoopState slots by
        # keyword; that shape must work and must not mutate s.messages.
        agent = _make_agent()
        summary = "summary text"
        ok_payload = json.dumps({"results": [{"status": "ok", "summary": summary}]})
        user_msg = {"role": "user", "content": "task"}
        messages = [user_msg]
        with patch("agent.auto_delegate.auto_delegate_enabled", return_value=True), patch(
            "tools.delegate_tool.delegate_task", return_value=ok_payload
        ):
            result = try_auto_delegate(
                agent, user_message="task", messages=messages, effective_task_id="task-9"
            )
        self.assertIsNotNone(result)
        if result is None:  # pragma: no cover
            return
        self.assertEqual(result["final_response"], summary)
        self.assertEqual(result["task_id"], "task-9")
        self.assertEqual(len(messages), 1, "input messages list must not be mutated")
        self.assertEqual(result["messages"][0], user_msg)

    def test_does_not_mutate_input_messages(self):
        agent = _make_agent()
        ok_payload = json.dumps({"results": [{"status": "ok", "summary": "s"}]})
        user_msg = {"role": "user", "content": "task"}
        messages = [user_msg]
        with patch("agent.auto_delegate.auto_delegate_enabled", return_value=True), patch(
            "tools.delegate_tool.delegate_task", return_value=ok_payload
        ):
            try_auto_delegate(agent, "task", messages)
        self.assertEqual(len(messages), 1, "input messages list must not be mutated")

    def test_falls_back_when_delegate_raises(self):
        agent = _make_agent()
        with patch("agent.auto_delegate.auto_delegate_enabled", return_value=True), patch(
            "tools.delegate_tool.delegate_task", side_effect=RuntimeError("boom")
        ):
            result = try_auto_delegate(agent, "task", [])
        self.assertIsNone(result)

    def test_falls_back_on_error_status(self):
        agent = _make_agent()
        err_payload = json.dumps({"results": [{"status": "error", "summary": None}]})
        with patch("agent.auto_delegate.auto_delegate_enabled", return_value=True), patch(
            "tools.delegate_tool.delegate_task", return_value=err_payload
        ):
            result = try_auto_delegate(agent, "task", [])
        self.assertIsNone(result)

    def test_falls_back_on_unparseable_result(self):
        agent = _make_agent()
        with patch("agent.auto_delegate.auto_delegate_enabled", return_value=True), patch(
            "tools.delegate_tool.delegate_task", return_value="<html>not json</html>"
        ):
            result = try_auto_delegate(agent, "task", [])
        self.assertIsNone(result)

    def test_falls_back_on_non_string_result(self):
        agent = _make_agent()
        with patch("agent.auto_delegate.auto_delegate_enabled", return_value=True), patch(
            "tools.delegate_tool.delegate_task", return_value=None
        ):
            result = try_auto_delegate(agent, "task", [])
        self.assertIsNone(result)


class TestLoopCallSite(unittest.TestCase):
    """The handoff is wired into the turn loop after the codex branch."""

    def test_call_site_is_after_codex_handoff_and_before_loop(self):
        # v0.21.3: the loop body lives in ``_run_conversation_turn`` (called by
        # ``run_conversation``), after the ``_LoopState`` construction.
        from agent import conversation_loop as cl

        src = inspect.getsource(cl._run_conversation_turn)
        codex_idx = src.index('agent.api_mode == "codex_app_server"')
        handoff_idx = src.index("try_auto_delegate(")
        loop_idx = src.index("while (s.api_call_count < agent.max_iterations")
        self.assertLess(codex_idx, handoff_idx)
        self.assertLess(handoff_idx, loop_idx)
        # Wrapped so a dispatcher failure can never hard-fail the turn.
        self.assertIn("except Exception:", src[handoff_idx:loop_idx])


if __name__ == "__main__":
    unittest.main()
