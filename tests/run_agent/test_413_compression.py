"""Tests for payload/context-length → compression retry logic in AIAgent.

Verifies that:
- HTTP 413 errors trigger history compression and retry
- HTTP 400 context-length errors trigger compression (not generic 4xx abort)
- Preflight compression proactively compresses oversized sessions before API calls
"""

import pytest
#pytestmark = pytest.mark.skip(reason="Hangs in non-interactive environments")



import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import SUMMARY_PREFIX
from run_agent import AIAgent
import run_agent


# ---------------------------------------------------------------------------
# Fast backoff for compression retry tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_compression_sleep(monkeypatch):
    """Short-circuit the 2s time.sleep between compression retries.

    Production code has ``time.sleep(2)`` in multiple places after a 413/context
    compression, for rate-limit smoothing. Tests assert behavior, not timing.
    """
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None, usage=None):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = SimpleNamespace(**usage) if usage else None
    return resp


def _make_413_error(*, use_status_code=True, message="Request entity too large"):
    """Create an exception that mimics a 413 HTTP error."""
    err = Exception(message)
    if use_status_code:
        err.status_code = 413
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
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = False
        a.save_trajectories = False
        return a


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHTTP413Compression:
    """413 errors should trigger compression, not abort as generic 4xx."""

    def test_413_triggers_compression(self, agent):
        """A 413 error should call _compress_context and retry, not abort."""
        # First call raises 413; second call succeeds after compression.
        err_413 = _make_413_error()
        ok_resp = _mock_response(content="Success after compression", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_413, ok_resp]

        # Prefill so there are multiple messages for compression to reduce
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
            # Compression reduces 3 messages down to 1
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True
        assert result["final_response"] == "Success after compression"

    def test_413_not_treated_as_generic_4xx(self, agent):
        """413 must NOT hit the generic 4xx abort path; it should attempt compression."""
        err_413 = _make_413_error()
        ok_resp = _mock_response(content="Recovered", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_413, ok_resp]

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
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        # If 413 were treated as generic 4xx, result would have "failed": True
        assert result.get("failed") is not True
        assert result["completed"] is True

    def test_413_error_message_detection(self, agent):
        """413 detected via error message string (no status_code attr)."""
        err = _make_413_error(use_status_code=False, message="error code: 413")
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
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True

    def test_413_clears_conversation_history_on_persist(self, agent):
        """After 413-triggered compression, _persist_session must receive None history.

        Bug: _compress_context() creates a new session and resets _last_flushed_db_idx=0,
        but if conversation_history still holds the original (pre-compression) list,
        _flush_messages_to_session_db computes flush_from = max(len(history), 0) which
        exceeds len(compressed_messages), so messages[flush_from:] is empty and nothing
        is written to the new session → "Session found but has no messages" on resume.
        """
        err_413 = _make_413_error()
        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_413, ok_resp]

        big_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(200)
        ]

        persist_calls = []

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(
                agent, "_persist_session",
                side_effect=lambda msgs, hist: persist_calls.append(hist),
            ),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "summary"}],
                "compressed prompt",
            )
            agent.run_conversation("hello", conversation_history=big_history)

        assert len(persist_calls) >= 1, "Expected at least one _persist_session call"
        for hist in persist_calls:
            assert hist is None, (
                f"conversation_history should be None after mid-loop compression, "
                f"got list with {len(hist)} items"
            )

    def test_context_overflow_clears_conversation_history_on_persist(self, agent):
        """After context-overflow compression, _persist_session must receive None history."""
        err_400 = Exception(
            "Error code: 400 - This endpoint's maximum context length is 128000 tokens. "
            "However, you requested about 270460 tokens."
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_400, ok_resp]

        big_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(200)
        ]

        persist_calls = []

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(
                agent, "_persist_session",
                side_effect=lambda msgs, hist: persist_calls.append(hist),
            ),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "summary"}],
                "compressed prompt",
            )
            agent.run_conversation("hello", conversation_history=big_history)

        assert len(persist_calls) >= 1
        for hist in persist_calls:
            assert hist is None, (
                f"conversation_history should be None after context-overflow compression, "
                f"got list with {len(hist)} items"
            )

    def test_400_context_length_triggers_compression(self, agent):
        """A 400 with 'maximum context length' should trigger compression, not abort as generic 4xx.

        OpenRouter returns HTTP 400 (not 413) for context-length errors. Before
        the fix, this was caught by the generic 4xx handler which aborted
        immediately — now it correctly triggers compression+retry.
        """
        err_400 = Exception(
            "Error code: 400 - {'error': {'message': "
            "\"This endpoint's maximum context length is 204800 tokens. "
            "However, you requested about 270460 tokens.\", 'code': 400}}"
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="Recovered after compression", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_400, ok_resp]

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
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        # Must NOT have "failed": True (which would mean the generic 4xx handler caught it)
        assert result.get("failed") is not True
        assert result["completed"] is True
        assert result["final_response"] == "Recovered after compression"

    def test_400_reduce_length_triggers_compression(self, agent):
        """A 400 with 'reduce the length' should trigger compression."""
        err_400 = Exception(
            "Error code: 400 - Please reduce the length of the messages"
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_400, ok_resp]

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
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True

    def test_context_length_retry_rebuilds_request_after_compression(self, agent):
        """Retry must send the compressed transcript, not the stale oversized payload."""
        err_400 = Exception(
            "Error code: 400 - {'error': {'message': "
            "\"This endpoint's maximum context length is 128000 tokens. "
            "Please reduce the length of the messages.\"}}"
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="Recovered after real compression", finish_reason="stop")

        request_payloads = []

        def _side_effect(**kwargs):
            request_payloads.append(kwargs)
            if len(request_payloads) == 1:
                raise err_400
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
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed summary"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        assert result["completed"] is True
        assert len(request_payloads) == 2
        assert len(request_payloads[1]["messages"]) < len(request_payloads[0]["messages"])
        assert request_payloads[1]["messages"][0] == {
            "role": "system",
            "content": "compressed prompt",
        }
        assert request_payloads[1]["messages"][1] == {
            "role": "user",
            "content": "compressed summary",
        }

    def test_413_cannot_compress_further(self, agent):
        """When compression can't reduce messages, return partial result."""
        err_413 = _make_413_error()
        agent.client.chat.completions.create.side_effect = [err_413]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # Compression returns same number of messages → can't compress further
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "same prompt",
            )
            result = agent.run_conversation("hello")

        assert result["completed"] is False
        assert result.get("partial") is True
        assert "413" in result["error"]


class TestCompressionTodoSnapshotRole:
    """Todo state preserved during compression must not masquerade as a user turn."""

    def test_compression_does_not_append_todo_snapshot_as_latest_user_message(self, agent):
        """Regression: auto-preserved todo state was appended as role=user at the tail.

        That made the restored todo snapshot look like the latest real user request
        after context compaction, causing stale tasks to override the user's actual
        next message. Compression may preserve todo state, but not as a user turn.
        """
        agent._todo_store.write([
            {
                "id": "task-contamination",
                "content": "定位串任务/旧任务恢复/上下文压缩误接管的根因",
                "status": "in_progress",
            }
        ])

        with patch.object(agent.context_compressor, "compress", return_value=[
            {"role": "user", "content": f"{SUMMARY_PREFIX}\nPrior context"},
            {"role": "user", "content": "最新真实用户消息"},
        ]):
            compressed, _system_prompt = agent._compress_context(
                [{"role": "user", "content": "old context"}],
                "You are helpful.",
                approx_tokens=1000,
            )

        assert compressed[-1] == {
            "role": "user",
            "content": "最新真实用户消息",
        }
        assert not any(
            msg.get("role") == "user"
            and "CURRENT SESSION TODO STATE" in str(msg.get("content", ""))
            for msg in compressed
        )



    def test_run_conversation_filters_legacy_auto_todo_snapshots_from_history(self, agent):
        """Resume/history loading must drop auto todo snapshots before API calls.

        Older builds persisted rendered todo snapshots as role=user. Even after
        gateway filtering, CLI/session resume can replay those messages unless
        run_conversation sanitizes the provided history.
        """
        legacy_snapshot = (
            "[Your active task list was preserved across context compression]\n"
            "- [>] stale. stale old task (in_progress)"
        )
        new_snapshot = (
            "[CURRENT SESSION TODO STATE — NOT A USER REQUEST]\n"
            "- [>] stale. stale old task (in_progress)"
        )
        history = [
            {"role": "user", "content": "real prior question"},
            {"role": "user", "content": legacy_snapshot},
            {"role": "assistant", "content": "prior answer"},
            {"role": "user", "content": new_snapshot},
        ]

        ok_resp = _mock_response(content="clean", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("latest real ask", conversation_history=history)

        sent_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
        user_contents = [m.get("content", "") for m in sent_messages if m.get("role") == "user"]
        assert "real prior question" in user_contents
        assert "latest real ask" in user_contents
        assert not any("Your active task list was preserved" in str(c) for c in user_contents)
        assert not any("CURRENT SESSION TODO STATE" in str(c) for c in user_contents)
        assert result["completed"] is True


    def test_run_conversation_does_not_filter_user_messages_that_quote_todo_marker(self, agent):
        """Only actual snapshot-shaped messages are filtered; user questions quoting markers remain."""
        quoted = "Why did Hermes show [CURRENT SESSION TODO STATE — NOT A USER REQUEST] in my chat?"
        history = [{"role": "user", "content": quoted}]

        ok_resp = _mock_response(content="kept", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("latest real ask", conversation_history=history)

        sent_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
        user_contents = [m.get("content", "") for m in sent_messages if m.get("role") == "user"]
        assert quoted in user_contents
        assert result["completed"] is True

    def test_run_conversation_filters_leading_whitespace_snapshot_from_history(self, agent):
        """Snapshot filtering should match gateway startswith semantics after lstrip()."""
        snapshot = "  \n[CURRENT SESSION TODO STATE — NOT A USER REQUEST]\n- [>] stale. stale old task (in_progress)"
        history = [
            {"role": "user", "content": "real prior question"},
            {"role": "user", "content": snapshot},
        ]

        ok_resp = _mock_response(content="clean", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("latest real ask", conversation_history=history)

        sent_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
        user_contents = [m.get("content", "") for m in sent_messages if m.get("role") == "user"]
        assert "real prior question" in user_contents
        assert not any("CURRENT SESSION TODO STATE" in str(c) for c in user_contents)
        assert result["completed"] is True


    def test_run_conversation_hydrates_todo_store_from_filtered_history(self, agent):
        """Hydration should use the same sanitized history that is sent to the model."""
        snapshot = "[Your active task list was preserved across context compression]\n- [>] stale. stale old task (in_progress)"
        history = [
            {"role": "user", "content": "real prior question"},
            {"role": "user", "content": snapshot},
        ]

        ok_resp = _mock_response(content="clean", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_hydrate_todo_store") as hydrate,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("latest real ask", conversation_history=history)

        hydrated_histories = [call.args[0] for call in hydrate.call_args_list]
        assert hydrated_histories, "expected run_conversation to hydrate the todo store"
        assert [{"role": "user", "content": "real prior question"}] in hydrated_histories
        for hydrated_history in hydrated_histories:
            assert not any(
                "active task list was preserved" in str(msg.get("content", ""))
                for msg in hydrated_history
            )
        assert result["completed"] is True

class TestPreflightCompression:
    """Preflight compression should compress history before the first API call."""

    def test_preflight_compresses_oversized_history(self, agent):
        """When loaded history exceeds the model's context threshold, compress before API call."""
        agent.compression_enabled = True
        # Set a small context so the history is "oversized", but large enough
        # that the compressed result (2 short messages) fits in a single pass.
        agent.context_compressor.context_length = 2000
        agent.context_compressor.threshold_tokens = 200

        # Build a history that will be large enough to trigger preflight
        # (each message ~50 chars ≈ 13 tokens, 40 messages ≈ 520 tokens > 200 threshold)
        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message number {i} with some extra text padding"})
            big_history.append({"role": "assistant", "content": f"Response number {i} with extra padding here"})

        ok_resp = _mock_response(content="After preflight", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # Simulate compression reducing messages to a small set that fits
            mock_compress.return_value = (
                [
                    {"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"},
                    {"role": "user", "content": "hello"},
                ],
                "new system prompt",
            )
            result = agent.run_conversation("hello", conversation_history=big_history)

        # Preflight compression is a multi-pass loop (up to 3 passes for very
        # large sessions, breaking when no further reduction is possible).
        # First pass must have received the full oversized history.
        assert mock_compress.call_count >= 1, "Preflight compression never ran"
        first_call_messages = mock_compress.call_args_list[0].args[0]
        assert len(first_call_messages) >= 40, (
            f"First preflight pass should see the full history, got "
            f"{len(first_call_messages)} messages"
        )
        assert result["completed"] is True
        assert result["final_response"] == "After preflight"

    def test_no_preflight_when_under_threshold(self, agent):
        """When history fits within context, no preflight compression needed."""
        agent.compression_enabled = True
        # Large context — history easily fits
        agent.context_compressor.context_length = 1000000
        agent.context_compressor.threshold_tokens = 850000

        small_history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

        ok_resp = _mock_response(content="No compression needed", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=small_history)

        mock_compress.assert_not_called()
        assert result["completed"] is True

    def test_no_preflight_when_compression_disabled(self, agent):
        """Preflight should not run when compression is disabled."""
        agent.compression_enabled = False
        agent.context_compressor.context_length = 100
        agent.context_compressor.threshold_tokens = 85

        big_history = [
            {"role": "user", "content": "x" * 1000},
            {"role": "assistant", "content": "y" * 1000},
        ] * 10

        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=big_history)

        mock_compress.assert_not_called()


class TestToolResultPreflightCompression:
    """Compression should trigger when tool results push context past the threshold."""

    def test_large_tool_results_trigger_compression(self, agent):
        """When tool results push estimated tokens past threshold, compress before next call."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 130_000  # below the 135k reported usage
        agent.context_compressor.last_prompt_tokens = 130_000
        agent.context_compressor.last_completion_tokens = 5_000

        tc = SimpleNamespace(
            id="tc1", type="function",
            function=SimpleNamespace(name="web_search", arguments='{"query":"test"}'),
        )
        tool_resp = _mock_response(
            content=None, finish_reason="stop", tool_calls=[tc],
            usage={"prompt_tokens": 130_000, "completion_tokens": 5_000, "total_tokens": 135_000},
        )
        ok_resp = _mock_response(
            content="Done after compression", finish_reason="stop",
            usage={"prompt_tokens": 50_000, "completion_tokens": 100, "total_tokens": 50_100},
        )
        agent.client.chat.completions.create.side_effect = [tool_resp, ok_resp]
        large_result = "x" * 100_000

        with (
            patch("run_agent.handle_function_call", return_value=large_result),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}], "compressed prompt",
            )
            result = agent.run_conversation("hello")

        mock_compress.assert_called_once()
        assert result["completed"] is True

    def test_anthropic_prompt_too_long_safety_net(self, agent):
        """Anthropic 'prompt is too long' error triggers compression as safety net."""
        err_400 = Exception(
            "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
            "'message': 'prompt is too long: 233153 tokens > 200000 maximum'}}"
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="Recovered", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_400, ok_resp]
        prefill = [
            {"role": "user", "content": "previous"},
            {"role": "assistant", "content": "answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}], "compressed",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True
