"""Loop-level tests for HTTP 408 request-timeout recovery in AIAgent.

Symptom-level companion to tests/agent/test_error_classifier.py::Test408RequestTimeout
(which asserts the *classification*). These drive the REAL run_conversation loop
and assert the *turn outcome*: a 408 must be retried as a transient timeout and
produce a real assistant turn — NOT abort into an empty assistant bubble (the
"disappeared conversation" / blank-turn symptom), and NOT silently compress
away conversation history.

Regression guarded: before agent/error_classifier.py grew a `408` branch, a 408
fell through to the generic "other 4xx -> non-retryable format_error" abort,
which persisted an empty assistant turn (blank bubble). The fix classifies 408
as `timeout` (retryable, NOT should_compress) so the loop retries the SAME
request — jitter-type 408s from GitHub Copilot near the prompt-size band clear
on the next attempt without destroying history.
"""

import pytest

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from run_agent import AIAgent
import run_agent


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    """Short-circuit retry backoff so the loop tests run fast."""
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = None
    return resp


def _make_408_error(*, message="Request Timeout", use_status_code=True, oversized=False):
    """Create an exception mimicking an HTTP 408.

    oversized=True reproduces GitHub Copilot's user_request_timeout body
    ("Timed out reading request body ... use a smaller request size").
    """
    if oversized:
        message = ("Error code: 408 - {'error': {'message': 'Timed out reading "
                   "request body. Try again, or use a smaller request size.', "
                   "'code': 'user_request_timeout'}}")
    err = Exception(message)
    if use_status_code:
        err.status_code = 408
    return err


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.githubcopilot.com",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = True   # prove we DON'T compress even when enabled
        a.save_trajectories = False
        return a


class TestHTTP408Loop:
    """A 408 must retry-and-recover into a real turn, never abort blank,
    never auto-compress."""

    def test_408_recovers_into_real_turn_not_blank_bubble(self, agent):
        """First call 408s, second succeeds → a real assistant answer, no abort."""
        err_408 = _make_408_error()
        ok_resp = _mock_response(content="Recovered after 408 retry", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_408, ok_resp]

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=prefill)

        # Symptom assertion: 408 did NOT abort into a failed/blank turn.
        assert result.get("failed") is not True
        assert result["completed"] is True
        assert result["final_response"] == "Recovered after 408 retry"

    def test_408_does_not_compress_history(self, agent):
        """A 408 must NOT call _compress_context — history stays intact."""
        err_408 = _make_408_error()
        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_408, ok_resp]

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=prefill)

        # The core user-requested guarantee: 408 never silently compacts.
        mock_compress.assert_not_called()
        assert result["completed"] is True

    def test_408_retries_same_request_not_shrunk(self, agent):
        """Retry after 408 must resend the SAME request (plain retry), not a
        compressed/shrunk one — proving it took the timeout path, not the
        payload_too_large/compression path."""
        err_408 = _make_408_error()
        ok_resp = _mock_response(content="OK", finish_reason="stop")

        request_payloads = []

        def _side_effect(**kwargs):
            request_payloads.append(kwargs)
            if len(request_payloads) == 1:
                raise err_408
            return ok_resp

        agent.client.chat.completions.create.side_effect = _side_effect

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=prefill)

        assert result["completed"] is True
        assert len(request_payloads) == 2
        mock_compress.assert_not_called()
        # Same request body length on retry (no compression shrank it).
        assert len(request_payloads[1]["messages"]) == len(request_payloads[0]["messages"])

    def test_oversized_body_408_also_retries_without_compression(self, agent):
        """The Copilot 'reading request body / smaller request size' 408 must
        ALSO retry-not-compress — it is jitter near the size band, not a hard
        overflow. (This is the exact case the user challenged: 'why must 408
        compress?' — answer: it must not.)"""
        err_408 = _make_408_error(oversized=True)
        ok_resp = _mock_response(content="Recovered oversized 408", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_408, ok_resp]

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_not_called()
        assert result.get("failed") is not True
        assert result["completed"] is True
        assert result["final_response"] == "Recovered oversized 408"

    def test_408_via_message_string_without_status_code(self, agent):
        """A 408 surfaced only via message text (no status_code attr) must
        still recover, not abort."""
        err = _make_408_error(use_status_code=False, message="error code: 408 Request Timeout")
        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err, ok_resp]

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_not_called()
        assert result["completed"] is True
