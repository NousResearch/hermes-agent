"""Regression tests for clean stdout in `hermes chat -Q -q`.

Dedicated automation wrappers rely on quiet single-query stdout containing only
one human answer. Tool/rendering noise produced inside the agent loop must not
share that stream.
"""

import contextlib
import io


class _NoisyAgent:
    def run_conversation(self, *, user_message, conversation_history):
        print("review diff: noisy tool renderer output")
        print("Timeout - denying command")
        return {"final_response": "clean final answer", "messages": []}


def test_quiet_single_query_redirects_agent_stdout_to_stderr():
    from cli import _run_quiet_single_query_conversation

    stdout = io.StringIO()
    stderr = io.StringIO()

    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = _run_quiet_single_query_conversation(
            _NoisyAgent(),
            user_message="do work",
            conversation_history=[],
        )
        response = result.get("final_response", "")
        if response:
            print(response)

    assert stdout.getvalue() == "clean final answer\n"
    assert "review diff: noisy tool renderer output" in stderr.getvalue()
    assert "Timeout - denying command" in stderr.getvalue()
