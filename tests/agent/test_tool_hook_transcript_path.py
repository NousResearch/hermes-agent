"""Regression tests for ``tool_hook_transcript_path``.

Feature PR #39939 exposes a redacted transcript snapshot path to the
``pre_tool_call`` / ``post_tool_call`` hooks. ``tool_hook_transcript_path`` is
the gate every tool-execution path calls to decide whether a snapshot should be
written at all. It is load-bearing for the spec's *no-listener, no-I/O fast
path*: when nothing is listening it must return ``""`` **without** invoking the
agent's snapshot writer, so tool execution stays allocation-free.
"""

import ast
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.agent_runtime_helpers import tool_hook_transcript_path


def _agent_with_writer(return_value="/tmp/session_s.json"):
    """A stand-in agent whose ``_hook_transcript_path`` records its calls."""
    writer = MagicMock(return_value=return_value)
    return SimpleNamespace(_hook_transcript_path=writer), writer


class TestToolHookTranscriptPath:
    def test_no_messages_returns_empty_without_touching_writer(self, monkeypatch):
        # Even with a live listener, an empty message list means there is
        # nothing to snapshot — the writer must never be reached.
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        agent, writer = _agent_with_writer()
        assert tool_hook_transcript_path(agent, []) == ""
        assert tool_hook_transcript_path(agent, None) == ""
        writer.assert_not_called()

    def test_no_listener_is_no_write_fast_path(self, monkeypatch):
        # The core invariant: no registered pre/post listener => return "" and
        # never invoke the writer, so no snapshot file is produced.
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
        agent, writer = _agent_with_writer()
        assert tool_hook_transcript_path(agent, [{"role": "user", "content": "hi"}]) == ""
        writer.assert_not_called()

    def test_pre_tool_call_listener_triggers_writer(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.plugins.has_hook", lambda name: name == "pre_tool_call"
        )
        agent, writer = _agent_with_writer("/tmp/session_pre.json")
        messages = [{"role": "user", "content": "hi"}]
        assert tool_hook_transcript_path(agent, messages) == "/tmp/session_pre.json"
        writer.assert_called_once_with(messages)

    def test_post_tool_call_listener_triggers_writer(self, monkeypatch):
        # A post-only listener is enough to require the snapshot.
        monkeypatch.setattr(
            "hermes_cli.plugins.has_hook", lambda name: name == "post_tool_call"
        )
        agent, writer = _agent_with_writer("/tmp/session_post.json")
        assert (
            tool_hook_transcript_path(agent, [{"role": "user", "content": "hi"}])
            == "/tmp/session_post.json"
        )
        writer.assert_called_once()

    def test_writer_returning_none_is_normalized_to_empty(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        agent, _ = _agent_with_writer(return_value=None)
        assert tool_hook_transcript_path(agent, [{"role": "user", "content": "hi"}]) == ""

    def test_writer_exception_is_swallowed(self, monkeypatch):
        # A failing snapshot writer must never break tool execution.
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        writer = MagicMock(side_effect=RuntimeError("disk full"))
        agent = SimpleNamespace(_hook_transcript_path=writer)
        assert tool_hook_transcript_path(agent, [{"role": "user", "content": "hi"}]) == ""

    def test_missing_or_non_callable_writer_returns_empty(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        messages = [{"role": "user", "content": "hi"}]
        assert tool_hook_transcript_path(SimpleNamespace(), messages) == ""
        assert (
            tool_hook_transcript_path(
                SimpleNamespace(_hook_transcript_path="not-callable"), messages
            )
            == ""
        )

    def test_has_hook_failure_is_swallowed(self, monkeypatch):
        # If the hook-registry probe itself raises, fail closed (no snapshot).
        def _boom(name):
            raise RuntimeError("registry unavailable")

        monkeypatch.setattr("hermes_cli.plugins.has_hook", _boom)
        agent, writer = _agent_with_writer()
        assert tool_hook_transcript_path(agent, [{"role": "user", "content": "hi"}]) == ""
        writer.assert_not_called()


class TestPostHookEmissionsForwardTranscriptPath:
    """Source guard: every post_tool_call emission in the concurrent and
    sequential tool executors forwards ``transcript_path``.

    ``transcript_path`` must reach the post hook on *every* tool-execution
    outcome — success, block, guardrail-block, cancel, thread-missing-result,
    and timeout. A concurrent-timeout emission that omitted it (silently
    defaulting to ``""``) was a real gap; this guard fails loudly if any such
    emission call regresses, on a path a behavioral test rarely reaches.
    """

    def test_every_terminal_post_hook_emission_threads_transcript_path(self):
        from agent import tool_executor

        tree = ast.parse(inspect.getsource(tool_executor))
        emit_names = {
            "_emit_terminal_post_tool_call",
            "_emit_cancelled_terminal_post_tool_call",
        }
        target_fns = {
            "execute_tool_calls_concurrent",
            "execute_tool_calls_sequential",
        }

        missing = []
        for fn in ast.walk(tree):
            if not (isinstance(fn, ast.FunctionDef) and fn.name in target_fns):
                continue
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in emit_names
                ):
                    if "transcript_path" not in {kw.arg for kw in node.keywords}:
                        missing.append(f"{fn.name}: {node.func.id} at line {node.lineno}")

        assert not missing, (
            "post_tool_call emission(s) missing transcript_path — the hook "
            f"would receive an empty transcript on that path: {missing}"
        )
