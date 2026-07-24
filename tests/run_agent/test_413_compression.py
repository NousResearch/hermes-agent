"""Tests for payload/context-length → compression retry logic in AIAgent.

Verifies that:
- HTTP 413 errors trigger history compression and retry
- HTTP 400 context-length errors trigger compression (not generic 4xx abort)
- Preflight compression proactively compresses oversized sessions before API calls
"""

import pytest
#pytestmark = pytest.mark.skip(reason="Hangs in non-interactive environments")



from types import SimpleNamespace
from unittest.mock import MagicMock, patch


from agent.context_compressor import SUMMARY_PREFIX
from agent.conversation_compression import COMPACTION_DONE_STATUS, COMPACTION_STATUS
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
        # Default matches production (`compression.enabled` defaults to True).
        # Overflow-recovery tests below verify that 413 / context-overflow
        # errors DO trigger compression; the disabled-path behavior is
        # covered explicitly by TestOverflowWithCompactionDisabled.
        a.compression_enabled = True
        a.save_trajectories = False
        return a


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_current_user_turn_is_persisted_before_provider_call(agent):
    """The inbound user turn is flushed before provider/tool work can crash."""
    observed = []

    def _record_persist(messages, conversation_history):
        observed.append(("persist", list(messages), list(conversation_history or [])))

    def _provider_crash(*_args, **_kwargs):
        observed.append(("provider", [], []))
        raise RuntimeError("provider died after turn-start persistence")

    agent.client.chat.completions.create.side_effect = _provider_crash

    with (
        patch.object(agent, "_persist_session", side_effect=_record_persist),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "new message that must survive a crash",
            conversation_history=[{"role": "user", "content": "old message"}],
        )

    assert result.get("failed") is True
    assert observed[0][0] == "persist"
    assert observed[1][0] == "provider"
    persisted_messages = observed[0][1]
    assert persisted_messages[-1] == {
        "role": "user",
        "content": "new message that must survive a crash",
    }


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

    def test_413_strips_vision_payloads_when_compression_cannot_reduce_messages(self, agent):
        """If compression leaves image payloads behind, strip them and retry.

        Browser vision tool results can contain base64 image parts. A 413 can
        persist even after summarisation when the remaining recent tool result
        still carries binary data; Hermes should evict the image payload and
        keep the text/placeholder context instead of failing immediately.
        """
        err_413 = _make_413_error()
        ok_resp = _mock_response(content="Recovered after image eviction", finish_reason="stop")
        request_payloads = []

        def _side_effect(**kwargs):
            request_payloads.append(kwargs)
            if len(request_payloads) == 1:
                raise err_413
            return ok_resp

        agent.client.chat.completions.create.side_effect = _side_effect

        image_part = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + ("a" * 2000)},
        }
        prefill = [
            {"role": "user", "content": "please inspect this page"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_vision",
                        "type": "function",
                        "function": {"name": "browser_vision", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_vision",
                "name": "browser_vision",
                "content": [
                    {"type": "text", "text": "Screenshot of the dashboard"},
                    image_part,
                ],
            },
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # Simulate the bad production case: compression ran, but the
            # recent vision tool message survived so message count did not drop.
            mock_compress.side_effect = lambda msgs, *_a, **_k: (msgs, "compressed prompt")
            result = agent.run_conversation("continue", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True
        assert result["final_response"] == "Recovered after image eviction"
        assert len(request_payloads) == 2
        first_tool = next(m for m in request_payloads[0]["messages"] if m.get("role") == "tool")
        retried_tool = next(m for m in request_payloads[1]["messages"] if m.get("role") == "tool")
        assert "Screenshot of the dashboard" in str(first_tool["content"])
        assert "data:image" not in str(retried_tool["content"])
        assert "Screenshot of the dashboard" in str(retried_tool["content"])
        assert not getattr(agent, "_no_list_tool_content_models", set())

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
                side_effect=lambda msgs, hist: persist_calls.append((list(msgs), hist)),
            ),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "summary"}],
                "compressed prompt",
            )
            agent.run_conversation("hello", conversation_history=big_history)

        assert any(hist is None for _msgs, hist in persist_calls), (
            "Expected at least one post-compression _persist_session call "
            "with conversation_history=None"
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
                side_effect=lambda msgs, hist: persist_calls.append((list(msgs), hist)),
            ),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "summary"}],
                "compressed prompt",
            )
            agent.run_conversation("hello", conversation_history=big_history)

        assert any(hist is None for _msgs, hist in persist_calls)

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

    def test_413_retries_on_token_only_compression(self, agent):
        """Same message COUNT but fewer TOKENS must count as progress and retry.

        Regression for #39550/#23767: tool-result pruning / in-place
        summarization can shrink request size without dropping the message
        count. The old gate (len(messages) < original_len) treated that as
        'cannot compress further' and aborted; the fix re-estimates tokens and
        retries when they drop materially.
        """
        err_413 = _make_413_error()
        ok_resp = _mock_response(content="OK after token-only compaction", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_413, ok_resp]

        # 3 large messages in, 3 much smaller messages out (same count, far
        # fewer tokens) — exactly the token-only-progress case.
        prefill = [
            {"role": "user", "content": "x" * 4000},
            {"role": "assistant", "content": "y" * 4000},
            {"role": "user", "content": "z" * 4000},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # Same message count (3) but ~10x smaller content → token drop.
            mock_compress.return_value = (
                [
                    {"role": "user", "content": "x" * 300},
                    {"role": "assistant", "content": "y" * 300},
                    {"role": "user", "content": "z" * 300},
                ],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True
        assert result["final_response"] == "OK after token-only compaction"


class TestPreflightCompression:
    """Preflight compression should compress history before the first API call."""

    def test_compress_context_emits_lifecycle_status_before_work(self, agent):
        """Direct context compression should tell gateway users why the turn paused."""
        # This test calls _compress_context directly and asserts the FIRST
        # status event is the lifecycle "Compacting context" message. With
        # compaction enabled the lazy feasibility probe would emit an
        # aux-provider warning first (no aux key in the hermetic test env),
        # displacing events[0]. The flag value is irrelevant to what this
        # test asserts, so disable it to suppress the probe.
        agent.compression_enabled = False
        events = []
        agent.status_callback = lambda ev, msg: events.append((ev, msg))

        def _fake_compress(
            messages,
            current_tokens=None,
            focus_topic=None,
            force=False,
            memory_context="",
        ):
            events.append(("compress", "started"))
            return [{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}]

        with (
            patch.object(agent.context_compressor, "compress", side_effect=_fake_compress),
            patch.object(agent, "_build_system_prompt", return_value="new system prompt") as build_prompt,
            patch("run_agent.estimate_request_tokens_rough", return_value=42),
        ):
            compressed, new_system_prompt = agent._compress_context(
                [{"role": "user", "content": "hello"}],
                "system prompt",
                approx_tokens=1234,
            )

        # The compressor returned only the user-role summary; a summary
        # message no longer satisfies the human-anchor check, so the real
        # user turn is restored after it (repair merges the pair
        # summary-first before the next API call).
        assert compressed == [
            {"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"},
            {"role": "user", "content": "hello"},
        ]
        assert new_system_prompt == "You are helpful."
        build_prompt.assert_not_called()
        assert events == [
            ("lifecycle", COMPACTION_STATUS),
            ("compress", "started"),
            ("compacted", COMPACTION_DONE_STATUS),
        ]

    def test_compress_context_emits_one_terminal_status_when_lock_is_unavailable(self, agent):
        """A rejected lock must retire the started desktop compaction phase."""
        agent.compression_enabled = False
        agent.session_id = "session-with-contended-lock"
        agent._session_db = SimpleNamespace(
            get_compression_lock_holder=lambda _session_id: "other-agent",
            try_acquire_compression_lock=lambda *_args, **_kwargs: False,
        )
        events = []
        agent.status_callback = lambda event, message: events.append((event, message))
        messages = [{"role": "user", "content": "hello"}]

        compressed, prompt = agent._compress_context(messages, "system prompt", force=True)

        assert compressed is messages
        assert prompt == "You are helpful."
        assert [event for event, _ in events] == ["lifecycle", "warn", "compacted"]
        assert events[-1] == ("compacted", COMPACTION_DONE_STATUS)

    def test_compress_context_emits_one_terminal_status_after_an_abort(self, agent):
        """An aborted summary must retire the started desktop compaction phase."""
        agent.compression_enabled = False
        events = []
        agent.status_callback = lambda event, message: events.append((event, message))
        messages = [{"role": "user", "content": "hello"}]

        def _abort_compression(current_messages, **_kwargs):
            agent.context_compressor._last_compress_aborted = True
            agent.context_compressor._last_summary_error = "auxiliary model unavailable"
            return current_messages

        with patch.object(agent.context_compressor, "compress", side_effect=_abort_compression):
            compressed, prompt = agent._compress_context(messages, "system prompt", force=True)

        assert compressed is messages
        assert prompt == "You are helpful."
        assert [event for event, _ in events] == ["lifecycle", "warn", "compacted"]
        assert events[-1] == ("compacted", COMPACTION_DONE_STATUS)

    def test_compression_reuses_cached_prompt_when_memory_snapshot_is_unchanged(self, agent):
        """A memory reload without new injected text must keep the cache prefix."""
        agent.compression_enabled = False
        agent._memory_enabled = True
        agent._user_profile_enabled = False
        agent._memory_manager = None
        agent._cached_system_prompt = (
            "cached system prompt\n\n<memory>same facts</memory>"
        )
        memory_store = MagicMock()
        memory_store.format_for_system_prompt.return_value = "<memory>same facts</memory>"
        agent._memory_store = memory_store

        with (
            patch.object(
                agent.context_compressor,
                "compress",
                return_value=[{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}],
            ),
            patch.object(agent, "_build_system_prompt") as build_prompt,
        ):
            _, new_system_prompt = agent._compress_context(
                [{"role": "user", "content": "hello"}],
                "system prompt",
                approx_tokens=1234,
            )

        assert new_system_prompt is agent._cached_system_prompt
        assert new_system_prompt == "cached system prompt\n\n<memory>same facts</memory>"
        build_prompt.assert_not_called()
        memory_store.load_from_disk.assert_called_once()

    def test_compression_rebuilds_prompt_when_memory_snapshot_changes(self, agent):
        """A changed memory block must be reflected in the next model request."""
        agent.compression_enabled = False
        agent._memory_enabled = True
        agent._user_profile_enabled = False
        agent._memory_manager = None
        agent._cached_system_prompt = (
            "cached system prompt\n\n<memory>old facts</memory>"
        )
        memory_store = MagicMock()
        memory_store.format_for_system_prompt.return_value = "<memory>new facts</memory>"
        agent._memory_store = memory_store

        with (
            patch.object(
                agent.context_compressor,
                "compress",
                return_value=[{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}],
            ),
            patch.object(agent, "_build_system_prompt", return_value="rebuilt system prompt") as build_prompt,
        ):
            _, new_system_prompt = agent._compress_context(
                [{"role": "user", "content": "hello"}],
                "system prompt",
                approx_tokens=1234,
            )

        assert new_system_prompt == "rebuilt system prompt"
        build_prompt.assert_called_once_with("system prompt")
        memory_store.load_from_disk.assert_called_once()

    def test_compression_rebuilds_when_restored_prompt_predates_memory_write(self, agent):
        """Gateway fresh-agent path: a session-DB-restored prompt built with OLD
        memory must be rebuilt even though the in-memory snapshot is identical
        before and after the disk reload (the fresh MemoryStore already
        absorbed the mid-session write at init). Guards the containment check
        against regressing to before/after snapshot equality."""
        agent.compression_enabled = False
        agent._memory_enabled = True
        agent._user_profile_enabled = False
        agent._memory_manager = None
        # Restored from SessionDB in an earlier process — built with fact A only.
        agent._cached_system_prompt = "system prompt\n\n<memory>fact A</memory>"
        memory_store = MagicMock()
        # Fresh store loaded fact A + fact B at agent init; stable across reload.
        memory_store.format_for_system_prompt.return_value = "<memory>fact A\nfact B</memory>"
        agent._memory_store = memory_store

        with (
            patch.object(
                agent.context_compressor,
                "compress",
                return_value=[{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}],
            ),
            patch.object(agent, "_build_system_prompt", return_value="rebuilt with fact B") as build_prompt,
        ):
            _, new_system_prompt = agent._compress_context(
                [{"role": "user", "content": "hello"}],
                "system prompt",
                approx_tokens=1234,
            )

        assert new_system_prompt == "rebuilt with fact B"
        build_prompt.assert_called_once_with("system prompt")

    def test_compression_rebuilds_when_prompt_has_leftover_block_for_emptied_memory(self, agent):
        """A prompt still carrying a memory block after all entries were
        removed must be rebuilt — empty current blocks are vacuously
        'contained', so the leftover-header check has to catch this."""
        agent.compression_enabled = False
        agent._memory_enabled = True
        agent._user_profile_enabled = False
        agent._memory_manager = None
        agent._cached_system_prompt = (
            "system prompt\n\nMEMORY (your personal notes) [1% — 10/2,200 chars]\nold fact"
        )
        memory_store = MagicMock()
        memory_store.format_for_system_prompt.return_value = None  # emptied
        agent._memory_store = memory_store

        with (
            patch.object(
                agent.context_compressor,
                "compress",
                return_value=[{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}],
            ),
            patch.object(agent, "_build_system_prompt", return_value="rebuilt without memory") as build_prompt,
        ):
            _, new_system_prompt = agent._compress_context(
                [{"role": "user", "content": "hello"}],
                "system prompt",
                approx_tokens=1234,
            )

        assert new_system_prompt == "rebuilt without memory"
        build_prompt.assert_called_once_with("system prompt")

    def test_compress_context_suppresses_automatic_status_when_engine_opts_out(self, agent):
        """Plugin engines can make successful automatic compaction silent."""
        # Keep this isolated from the lazy aux-provider feasibility warning,
        # which is unrelated to automatic compaction lifecycle status.
        agent.compression_enabled = False
        events = []
        agent.status_callback = lambda ev, msg: events.append((ev, msg))
        agent.context_compressor.emit_automatic_compaction_status = False

        def _fake_compress(
            messages,
            current_tokens=None,
            focus_topic=None,
            force=False,
            memory_context="",
        ):
            events.append(("compress", "started"))
            return [{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}]

        with (
            patch.object(agent.context_compressor, "compress", side_effect=_fake_compress),
            patch.object(agent, "_build_system_prompt", return_value="new system prompt"),
            patch("run_agent.estimate_request_tokens_rough", return_value=42),
        ):
            compressed, new_system_prompt = agent._compress_context(
                [{"role": "user", "content": "hello"}],
                "system prompt",
                approx_tokens=1234,
            )

        # The compressor returned only the user-role summary; the human-anchor
        # repair restores the real user turn after it (same contract as
        # test_compress_context_emits_lifecycle_status_before_work above).
        assert compressed == [
            {"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"},
            {"role": "user", "content": "hello"},
        ]
        # Memory snapshot unchanged → the cached prompt is retained (same
        # contract as test_compress_context_emits_lifecycle_status_before_work).
        assert new_system_prompt == "You are helpful."
        assert events == [("compress", "started")]

    def test_compress_context_force_keeps_manual_status_when_engine_opts_out(self, agent):
        """Manual /compress remains visible even for quiet automatic engines."""
        # Keep this isolated from the lazy aux-provider feasibility warning,
        # which is unrelated to manual compression lifecycle status.
        agent.compression_enabled = False
        events = []
        agent.status_callback = lambda ev, msg: events.append((ev, msg))
        agent.context_compressor.emit_automatic_compaction_status = False

        def _fake_compress(messages, current_tokens=None, focus_topic=None, force=False):
            events.append(("compress", "started"))
            return [{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}]

        with (
            patch.object(agent.context_compressor, "compress", side_effect=_fake_compress),
            patch.object(agent, "_build_system_prompt", return_value="new system prompt"),
            patch("run_agent.estimate_request_tokens_rough", return_value=42),
        ):
            agent._compress_context(
                [{"role": "user", "content": "hello"}],
                "system prompt",
                approx_tokens=1234,
                force=True,
            )

        assert events[0][0] == "lifecycle"
        assert "Compacting context" in events[0][1]
        assert events[1] == ("compress", "started")

    def test_compress_context_abort_warning_is_never_suppressed(self, agent):
        """Failure warnings stay visible even when a quiet engine suppresses
        routine automatic status — only ROUTINE lifecycle lines are quiet."""
        agent.compression_enabled = False
        agent.context_compressor.emit_automatic_compaction_status = False
        events = []
        agent.status_callback = lambda event, message: events.append((event, message))
        messages = [{"role": "user", "content": "hello"}]

        def _abort_compression(current_messages, **_kwargs):
            agent.context_compressor._last_compress_aborted = True
            agent.context_compressor._last_summary_error = "auxiliary model unavailable"
            return current_messages

        with patch.object(agent.context_compressor, "compress", side_effect=_abort_compression):
            compressed, prompt = agent._compress_context(messages, "system prompt")

        assert compressed is messages
        assert prompt == "You are helpful."
        # Routine lifecycle + structured compacted edges are suppressed (the
        # quiet engine opened no visible phase), but the abort warning fires.
        assert [event for event, _ in events] == ["warn"]
        assert "Compression aborted" in events[0][1]

    def test_pre_api_compression_status_suppressed_when_engine_opts_out(self, agent):
        """The mid-turn pre-API pressure emit routes through the resolver too.

        Regression guard for the #35191 review gap: with
        ``emit_automatic_compaction_status = False`` the 📦 Pre-API line must
        not leak even though compaction itself still runs.
        """
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 130_000
        agent.context_compressor.emit_automatic_compaction_status = False

        history = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        ok_resp = _mock_response(content="After quiet pre-API", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]
        status_messages = []
        agent.status_callback = lambda ev, msg: status_messages.append((ev, msg))

        with (
            # Keep the turn-prologue preflight quiet-by-size so only the
            # in-loop pre-API pressure gate fires.
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=10_000),
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=144_669),
            patch(
                "agent.conversation_loop.estimate_messages_tokens_rough",
                return_value=144_669,
            ),
            patch.object(
                agent,
                "_compress_context",
                side_effect=lambda msgs, *a, **k: (msgs, agent._cached_system_prompt),
            ) as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=history)

        assert result["completed"] is True
        assert mock_compress.call_count >= 1, "pre-API compression never ran"
        assert not any(
            "Pre-API compression" in msg for _ev, msg in status_messages
        )

    def test_pre_api_compression_status_emitted_by_default(self, agent):
        """Control: the default (non-quiet) engine keeps the pre-API line."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 130_000

        history = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        ok_resp = _mock_response(content="After pre-API", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]
        status_messages = []
        agent.status_callback = lambda ev, msg: status_messages.append((ev, msg))

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=10_000),
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=144_669),
            patch(
                "agent.conversation_loop.estimate_messages_tokens_rough",
                return_value=144_669,
            ),
            patch.object(
                agent,
                "_compress_context",
                side_effect=lambda msgs, *a, **k: (msgs, agent._cached_system_prompt),
            ) as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=history)

        assert result["completed"] is True
        assert mock_compress.call_count >= 1
        assert any(
            ev == "lifecycle" and "Pre-API compression" in msg
            for ev, msg in status_messages
        )

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
        status_messages = []
        agent.status_callback = lambda ev, msg: status_messages.append((ev, msg))

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
        assert any(
            ev == "lifecycle" and "Preflight compression" in msg
            for ev, msg in status_messages
        )

    def test_preflight_suppresses_status_when_context_engine_opts_out(self, agent):
        """LCM-style engines can keep routine automatic preflight maintenance silent."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 100_000
        agent.context_compressor.emit_automatic_compaction_status = False

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded"})

        ok_resp = _mock_response(content="After quiet preflight", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]
        status_messages = []
        agent.status_callback = lambda ev, msg: status_messages.append((ev, msg))

        _rough_calls = {"n": 0}

        def _rough_estimate(*_args, **_kwargs):
            _rough_calls["n"] += 1
            return 114_000 if _rough_calls["n"] == 1 else 40_000

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", side_effect=_rough_estimate),
            patch("agent.conversation_loop.estimate_request_tokens_rough", side_effect=_rough_estimate),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}],
                "new system prompt",
            )
            result = agent.run_conversation("hello", conversation_history=big_history)

        mock_compress.assert_called_once()
        assert result["completed"] is True
        assert not any(
            ev == "lifecycle" and "Preflight compression" in msg
            for ev, msg in status_messages
        )

    def test_preflight_uses_context_engine_custom_status_message(self, agent):
        """Plugin engines can replace generic built-in-compressor wording."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 100_000

        def _custom_status(**kwargs):
            assert kwargs["phase"] == "preflight"
            assert kwargs["approx_tokens"] == 114_000
            assert kwargs["threshold_tokens"] == 100_000
            return "🔧 LCM context maintenance: preparing compacted context."

        agent.context_compressor.get_automatic_compaction_status_message = _custom_status

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded"})

        ok_resp = _mock_response(content="After custom preflight", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]
        status_messages = []
        agent.status_callback = lambda ev, msg: status_messages.append((ev, msg))

        _rough_calls = {"n": 0}

        def _rough_estimate(*_args, **_kwargs):
            _rough_calls["n"] += 1
            return 114_000 if _rough_calls["n"] == 1 else 40_000

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", side_effect=_rough_estimate),
            patch("agent.conversation_loop.estimate_request_tokens_rough", side_effect=_rough_estimate),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}],
                "new system prompt",
            )
            result = agent.run_conversation("hello", conversation_history=big_history)

        mock_compress.assert_called_once()
        assert result["completed"] is True
        lifecycle_messages = [msg for ev, msg in status_messages if ev == "lifecycle"]
        assert "🔧 LCM context maintenance: preparing compacted context." in lifecycle_messages
        assert not any("Preflight compression" in msg for msg in lifecycle_messages)

    def test_preflight_defers_when_recent_real_usage_fit(self, agent):
        """A noisy rough estimate should not re-compact a recently fitting request."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 100_000
        agent.context_compressor.last_prompt_tokens = 58_000
        agent.context_compressor.last_real_prompt_tokens = 58_000
        agent.context_compressor.last_rough_tokens_when_real_prompt_fit = 113_000

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded"})

        ok_resp = _mock_response(
            content="Used real fit",
            finish_reason="stop",
            usage={"prompt_tokens": 59_000, "completion_tokens": 100, "total_tokens": 59_100},
        )
        agent.client.chat.completions.create.side_effect = [ok_resp]
        status_messages = []
        agent.status_callback = lambda ev, msg: status_messages.append((ev, msg))

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=114_000),
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=114_000),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=big_history)

        mock_compress.assert_not_called()
        assert result["completed"] is True
        assert result["final_response"] == "Used real fit"
        assert not any(
            ev == "lifecycle" and "Preflight compression" in msg
            for ev, msg in status_messages
        )

    def test_preflight_compresses_when_rough_growth_after_fit_is_large(self, agent):
        """Large rough growth after a fitting request still triggers preflight."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 100_000
        agent.context_compressor.last_prompt_tokens = 58_000
        agent.context_compressor.last_real_prompt_tokens = 58_000
        agent.context_compressor.last_rough_tokens_when_real_prompt_fit = 113_000

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded"})

        ok_resp = _mock_response(
            content="Compressed after growth",
            finish_reason="stop",
            usage={"prompt_tokens": 50_000, "completion_tokens": 100, "total_tokens": 50_100},
        )
        agent.client.chat.completions.create.side_effect = [ok_resp]

        # First rough estimate must clear the threshold so preflight fires
        # (rough growth since the last fitting request is large, so the
        # deferral path is NOT taken). Every estimate after compaction is
        # sub-threshold. Use a callable side_effect rather than a fixed list
        # so we don't have to predict how many times the loop re-estimates —
        # the post-response real-token estimate is an extra call that a
        # 2-element list would exhaust (StopIteration).
        _rough_calls = {"n": 0}

        def _rough_estimate(*_args, **_kwargs):
            _rough_calls["n"] += 1
            return 125_000 if _rough_calls["n"] == 1 else 40_000

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", side_effect=_rough_estimate),
            patch("agent.conversation_loop.estimate_request_tokens_rough", side_effect=_rough_estimate),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}],
                "new system prompt",
            )
            result = agent.run_conversation("hello", conversation_history=big_history)

        mock_compress.assert_called_once()
        assert result["completed"] is True

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

    def test_preflight_respects_anti_thrash(self, agent):
        """Preflight must call ``should_compress()`` so anti-thrash applies.

        Regression for #29335 — preflight used to bypass ``should_compress()``
        and re-trigger every turn even when the prior two passes each saved
        <10% (the canonical infinite-compression-loop signal).
        """
        agent.compression_enabled = True
        agent.context_compressor.context_length = 2000
        agent.context_compressor.threshold_tokens = 200

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded"})

        ok_resp = _mock_response(content="No preflight", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent.context_compressor, "should_compress", return_value=False) as mock_should,
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=big_history)

        # The gate consulted should_compress — anti-thrash had a chance to vote.
        mock_should.assert_called()
        # And vetoed: even though tokens >= threshold, no compression ran.
        mock_compress.assert_not_called()
        assert result["completed"] is True

    def test_preflight_seeds_display_tokens_when_compression_aborts(self, agent):
        """Display must reflect the real context size even when compression no-ops.

        Regression: the CLI status bar reads ``last_prompt_tokens``, which only
        updated from a *successful* API response. When the loaded history was
        oversized but compression failed to reduce it (e.g. the auxiliary
        summary model timed out), the bar stayed stuck at the old, smaller
        value while the preflight estimate reported a much larger number —
        looking permanently out of sync.
        """
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 130_000
        # Simulate a stale display value from an earlier, smaller turn.
        agent.context_compressor.last_prompt_tokens = 74_400

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded text"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded text"})

        ok_resp = _mock_response(content="After preflight", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=144_669),
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=144_669),
            patch(
                "agent.conversation_loop.estimate_messages_tokens_rough",
                return_value=144_669,
            ),
            # Compression no-ops (returns input unchanged) — mirrors an aux
            # summary-model timeout where the messages can't be reduced.
            patch.object(
                agent,
                "_compress_context",
                side_effect=lambda msgs, *a, **k: (msgs, agent._cached_system_prompt),
            ) as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=big_history)

        assert result["completed"] is True
        # A no-op pass cannot become more effective by immediately summarizing
        # the same request twice more. Proceed to the provider/recovery path
        # after one attempt instead of spending the full three-pass budget.
        assert mock_compress.call_count == 1
        # The display token count was revised up to the fresh preflight estimate,
        # not left at the stale 74_400.
        assert agent.context_compressor.last_prompt_tokens == 144_669

    def test_preflight_seed_only_revises_upward(self, agent):
        """A larger tracked value must not be clobbered by a smaller estimate."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 130_000
        # A real, larger usage figure is already tracked.
        agent.context_compressor.last_prompt_tokens = 160_000

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded text"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded text"})

        ok_resp = _mock_response(content="After preflight", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=144_669),
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=144_669),
            patch.object(agent, "_compress_context", side_effect=lambda msgs, *a, **k: (msgs, agent._cached_system_prompt)),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation("hello", conversation_history=big_history)

        # Smaller estimate must not overwrite the larger tracked value.
        assert agent.context_compressor.last_prompt_tokens == 160_000

    def test_preflight_stops_after_marginal_compression(self, agent):
        """Do not spend three summary calls removing one row per pass."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 130_000

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded text"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded text"})

        ok_resp = _mock_response(content="After marginal preflight", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        def _drop_one_row(messages, *_args, **_kwargs):
            return messages[:-1], agent._cached_system_prompt

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=144_669),
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=144_669),
            patch(
                "agent.conversation_loop.estimate_messages_tokens_rough",
                return_value=144_669,
            ),
            patch.object(
                agent, "_compress_context", side_effect=_drop_one_row
            ) as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=big_history)

        assert result["completed"] is True
        assert result["final_response"] == "After marginal preflight"
        assert mock_compress.call_count == 1

    @pytest.mark.parametrize(
        "rows_removed",
        [pytest.param(0, id="no-op"), pytest.param(1, id="marginal")],
    )
    def test_provider_overflow_recovers_after_blocked_turn_start_preflight(
        self, agent, rows_removed
    ):
        """The proactive retry block must not consume provider-overflow recovery."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 130_000

        big_history = []
        for i in range(20):
            big_history.append(
                {"role": "user", "content": f"Message {i} padded text"}
            )
            big_history.append(
                {"role": "assistant", "content": f"Response {i} padded text"}
            )

        agent.client.chat.completions.create.side_effect = [
            _make_413_error(),
            _mock_response(content="Recovered after overflow", finish_reason="stop"),
        ]

        compress_calls = 0

        def _compress(messages, *_args, **_kwargs):
            nonlocal compress_calls
            compress_calls += 1
            if compress_calls == 1:
                kept = messages[:-rows_removed] if rows_removed else messages
                return kept, agent._cached_system_prompt
            return (
                [{"role": "user", "content": "hello"}],
                "compressed after provider overflow",
            )

        with (
            patch(
                "agent.turn_context.estimate_request_tokens_rough",
                return_value=144_669,
            ),
            patch(
                "agent.conversation_loop.estimate_request_tokens_rough",
                return_value=144_669,
            ),
            patch(
                "agent.conversation_loop.estimate_messages_tokens_rough",
                return_value=144_669,
            ),
            patch.object(
                agent, "_compress_context", side_effect=_compress
            ) as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "hello", conversation_history=big_history
            )

        assert result["completed"] is True
        assert result["final_response"] == "Recovered after overflow"
        assert mock_compress.call_count == 2


    def test_interrupt_before_first_provider_call_restores_preflight_display_seed(self, agent):
        """Interrupted turns must not keep a speculative preflight display seed.

        Preflight runs before the main loop checks ``_interrupt_requested``.
        If the user interrupts during that window, no provider usage ever
        arrives to validate the rough estimate, so the old display token count
        must be restored instead of leaking the speculative value forward.
        """
        agent.compression_enabled = True
        agent._interrupt_requested = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 130_000
        agent.context_compressor.last_prompt_tokens = 74_400

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded text"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded text"})

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=144_669),
            patch.object(agent.context_compressor, "should_compress", return_value=False),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=big_history)

        assert result["interrupted"] is True
        assert agent.client.chat.completions.create.call_count == 0
        assert agent.context_compressor.last_prompt_tokens == 74_400

    def test_usage_less_provider_response_prevents_display_seed_rollback(self, agent):
        """A successful response counts even when the provider omits usage."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 130_000
        agent.context_compressor.last_prompt_tokens = 74_400

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded text"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded text"})

        tool_call = SimpleNamespace(
            id="tc1",
            type="function",
            function=SimpleNamespace(name="web_search", arguments='{"query":"test"}'),
        )
        agent.client.chat.completions.create.side_effect = [
            _mock_response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[tool_call],
                usage=None,
            )
        ]

        def _interrupt_after_tool(*_args, **_kwargs):
            agent._interrupt_requested = True

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=144_669),
            patch.object(agent.context_compressor, "should_compress", return_value=False),
            patch.object(agent, "_execute_tool_calls", side_effect=_interrupt_after_tool),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=big_history)

        assert result["interrupted"] is True
        assert agent.client.chat.completions.create.call_count == 1
        assert agent.context_compressor.last_prompt_tokens == 144_669

    def test_interrupt_keeps_post_compression_state(self, agent):
        """Display rollback must not restore real post-compaction state.

        A completed preflight compaction still leaves the conversation in the
        post-compression ``-1`` sentinel state. Its anti-thrashing verdict also
        remains owned by the completed compaction boundary rather than the
        speculative display snapshot.
        """
        agent.compression_enabled = True
        agent._interrupt_requested = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 130_000
        agent.context_compressor.last_prompt_tokens = 74_400
        agent.context_compressor._ineffective_compression_count = 1

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded text"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded text"})

        def _fake_preflight_compress(msgs, *_args, **_kwargs):
            agent.context_compressor.last_prompt_tokens = -1
            agent.context_compressor.awaiting_real_usage_after_compression = True
            agent.context_compressor.compression_count += 1
            agent.context_compressor._ineffective_compression_count = 2
            agent.context_compressor._last_compression_savings_pct = 0.0
            return msgs, agent._cached_system_prompt

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=144_669),
            patch.object(agent.context_compressor, "should_compress", return_value=True),
            patch.object(agent, "_compress_context", side_effect=_fake_preflight_compress),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=big_history)

        assert result["interrupted"] is True
        assert agent.client.chat.completions.create.call_count == 0
        assert agent.context_compressor.last_prompt_tokens == -1
        assert agent.context_compressor.awaiting_real_usage_after_compression is True
        assert agent.context_compressor._ineffective_compression_count == 2
        assert agent.context_compressor._last_compression_savings_pct == 0.0


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

    def test_mid_turn_retry_compares_fully_assembled_requests(self, agent):
        """API-only context must not make marginal compression look effective."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 130_000

        tc = SimpleNamespace(
            id="tc1", type="function",
            function=SimpleNamespace(name="web_search", arguments='{"query":"test"}'),
        )
        tool_resp = _mock_response(
            content="",
            finish_reason="stop",
            tool_calls=[tc],
        )
        ok_resp = _mock_response(
            content="Done after one compression", finish_reason="stop"
        )
        agent.client.chat.completions.create.side_effect = [tool_resp, ok_resp]

        # First provider request is small. The tool result pushes the fully
        # assembled request over threshold; rebuilding after compression only
        # trims it from 150K to 148K. Raw-message estimation is much smaller,
        # which previously made the no-op pass look successful and allowed two
        # more immediate summaries.
        assembled_estimates = iter(
            [1_000, 150_000, 148_000, 148_000, 148_000]
        )

        with (
            patch(
                "agent.conversation_loop.estimate_messages_tokens_rough",
                side_effect=lambda *_a, **_k: next(assembled_estimates),
            ),
            patch("run_agent.handle_function_call", return_value="x" * 100_000),
            patch.object(
                agent,
                "_compress_context",
                side_effect=lambda msgs, *_a, **_k: (
                    msgs,
                    agent._cached_system_prompt,
                ),
            ) as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "Done after one compression"
        assert mock_compress.call_count == 1

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


# ---------------------------------------------------------------------------
# Disabled auto-compaction on overflow (port of anomalyco/opencode#30749)
# ---------------------------------------------------------------------------

class TestOverflowWithCompactionDisabled:
    """When ``compression.enabled`` is False, NO automatic compaction may
    fire — including the provider/request-size overflow recovery paths.

    Ported from anomalyco/opencode#30749: the proactive token-threshold
    path already honoured the setting, but provider overflow errors
    (413 payload-too-large, context-overflow, long-context-tier 429) still
    silently compressed + rotated the session. The fix surfaces a terminal
    error so the user can compact manually, start fresh, or switch models.
    """

    @staticmethod
    def _prefill():
        return [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

    def test_413_does_not_compress_when_disabled(self, agent):
        """413 must NOT call _compress_context when compaction is disabled."""
        agent.compression_enabled = False
        err_413 = _make_413_error()
        # If the guard fails, a second (success) response would be consumed.
        agent.client.chat.completions.create.side_effect = [err_413, _mock_response()]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session") as mock_persist,
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=self._prefill())

        mock_compress.assert_not_called()
        mock_persist.assert_called()
        assert result.get("failed") is True
        assert result.get("compaction_disabled") is True
        assert "auto-compaction is disabled" in result["error"]

    def test_context_overflow_does_not_compress_when_disabled(self, agent):
        """400 'prompt is too long' must NOT compress when compaction disabled."""
        agent.compression_enabled = False
        err_400 = Exception(
            "Error code: 400 - {'type': 'error', 'error': {'type': "
            "'invalid_request_error', 'message': 'prompt is too long: "
            "233153 tokens > 200000 maximum'}}"
        )
        err_400.status_code = 400
        agent.client.chat.completions.create.side_effect = [err_400, _mock_response()]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=self._prefill())

        mock_compress.assert_not_called()
        assert result.get("compaction_disabled") is True

    def test_413_still_compresses_when_enabled(self, agent):
        """Control: with compaction enabled, 413 still triggers compression.

        Guards against the disabled-path guard accidentally swallowing the
        enabled path.
        """
        agent.compression_enabled = True
        err_413 = _make_413_error()
        ok_resp = _mock_response(content="Recovered", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_413, ok_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}], "compressed",
            )
            result = agent.run_conversation("hello", conversation_history=self._prefill())

        mock_compress.assert_called_once()
        assert result["completed"] is True
        assert result.get("compaction_disabled") is not True
