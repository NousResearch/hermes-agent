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

import json
import threading
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
            "results": [{"status": "completed", "summary": "  done.  ", "task_index": 0}],
            "total_duration_seconds": 1.2,
        }
        self.assertEqual(_extract_first_summary(payload), "done.")

    def test_error_status_returns_none(self):
        payload = {"results": [{"status": "failed", "summary": None}], "total_duration_seconds": 0.1}
        self.assertIsNone(_extract_first_summary(payload))

    def test_empty_results_returns_none(self):
        self.assertIsNone(_extract_first_summary({"results": []}))
        self.assertIsNone(_extract_first_summary({}))
        self.assertIsNone(_extract_first_summary(None))
        self.assertIsNone(_extract_first_summary("not a dict"))

    def test_blank_summary_returns_none(self):
        payload = {"results": [{"status": "completed", "summary": "   "}]}
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
            {"results": [{"status": "completed", "summary": summary, "task_index": 0}]}
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
        ok_payload = json.dumps({"results": [{"status": "completed", "summary": summary}]})
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
        ok_payload = json.dumps({"results": [{"status": "completed", "summary": "s"}]})
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
        err_payload = json.dumps({"results": [{"status": "failed", "summary": None}]})
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


class TestAutoDelegateE2E(unittest.TestCase):
    """Real ``delegate_task`` chain, not a mock at the tool boundary.

    The child agent is stubbed at the ``run_agent.AIAgent`` boundary (no real
    LLM call), but ``_normalize_task_list`` → ``_build_children`` →
    ``_run_batch`` → ``_finalize_child_results`` → ``_build_result_entry`` →
    ``_extract_first_summary`` all run for real against a temp ``HERMES_HOME``
    with ``delegation.auto_delegate: true`` written to ``config.yaml``. This
    exercises the actual result-entry status contract (``"completed"``) — a
    mock at the ``delegate_task`` boundary previously hid the ``"ok"`` vs
    ``"completed"`` mismatch that made dispatcher mode silently no-op.
    """

    def _make_real_parent(self):
        parent = MagicMock()
        parent.base_url = "https://openrouter.ai/api/v1"
        parent.api_key = "***"
        parent.provider = "openrouter"
        parent.api_mode = "chat_completions"
        parent.model = "anthropic/claude-sonnet-4"
        parent.platform = "cli"
        parent.providers_allowed = None
        parent.providers_ignored = None
        parent.providers_order = None
        parent.provider_sort = None
        parent._session_db = None
        parent._delegate_depth = 0
        parent._active_children = []
        parent._active_children_lock = threading.Lock()
        parent._print_fn = None
        parent.tool_progress_callback = None
        parent.thinking_callback = None
        return parent

    def test_real_delegate_chain_dispatches_and_returns_summary(self):
        import os

        from hermes_cli.config import _LOAD_CONFIG_CACHE
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
        config_path = os.path.join(hermes_home, "config.yaml")
        os.makedirs(hermes_home, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("delegation:\n  auto_delegate: true\n")
        _LOAD_CONFIG_CACHE.clear()
        try:
            parent = self._make_real_parent()
            summary = "Child finished: wrote /tmp/out.txt."
            user_msg = {"role": "user", "content": "write a report"}
            with patch("run_agent.AIAgent") as MockAgent:
                mock_child = MagicMock()
                mock_child.run_conversation.return_value = {
                    "final_response": summary,
                    "completed": True,
                    "api_calls": 1,
                }
                MockAgent.return_value = mock_child
                result = try_auto_delegate(
                    parent,
                    user_message="write a report",
                    messages=[user_msg],
                    effective_task_id="task-e2e",
                )
        finally:
            _LOAD_CONFIG_CACHE.clear()
            if os.path.exists(config_path):
                os.remove(config_path)

        self.assertIsNotNone(
            result, "auto_delegate must dispatch, not silently fall back to the normal loop"
        )
        if result is None:  # pragma: no cover - unreachable on success
            return
        self.assertEqual(result["final_response"], summary)
        self.assertTrue(result["auto_delegated"])
        self.assertEqual(result["turn_exit_reason"], "auto_delegated")
        self.assertEqual(result["task_id"], "task-e2e")


if __name__ == "__main__":
    unittest.main()
