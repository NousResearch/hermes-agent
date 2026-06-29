"""Unit tests for run_agent.py (AIAgent) — run_conversation loop, max-iterations, retry exhaustion, todo hydration.

Split out of the former monolithic ``tests/run_agent/test_run_agent.py`` (which
outgrew the per-file CI wall-clock cap). Shared fixtures live in ``conftest.py``;
mock-builders in ``_run_agent_helpers.py``.
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.run_agent._run_agent_helpers import (
    _mock_response,
    _mock_tool_call,
)


class TestHydrateTodoStore:
    def test_no_todo_in_history(self, agent):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        with patch("run_agent._set_interrupt"):
            agent._hydrate_todo_store(history)
        assert not agent._todo_store.has_items()

    def test_recovers_from_history(self, agent):
        todos = [{"id": "1", "content": "do thing", "status": "pending"}]
        history = [
            {"role": "user", "content": "plan"},
            {"role": "assistant", "content": "ok"},
            {
                "role": "tool",
                "content": json.dumps({"todos": todos}),
                "tool_call_id": "c1",
            },
        ]
        with patch("run_agent._set_interrupt"):
            agent._hydrate_todo_store(history)
        assert agent._todo_store.has_items()

    def test_skips_non_todo_tools(self, agent):
        history = [
            {
                "role": "tool",
                "content": '{"result": "search done"}',
                "tool_call_id": "c1",
            },
        ]
        with patch("run_agent._set_interrupt"):
            agent._hydrate_todo_store(history)
        assert not agent._todo_store.has_items()

    def test_invalid_json_skipped(self, agent):
        history = [
            {
                "role": "tool",
                "content": 'not valid json "todos" oops',
                "tool_call_id": "c1",
            },
        ]
        with patch("run_agent._set_interrupt"):
            agent._hydrate_todo_store(history)
        assert not agent._todo_store.has_items()


class TestHandleMaxIterations:
    def test_returns_summary(self, agent):
        resp = _mock_response(content="Here is a summary of what I did.")
        agent.client.chat.completions.create.return_value = resp
        agent._cached_system_prompt = "You are helpful."
        messages = [{"role": "user", "content": "do stuff"}]
        result = agent._handle_max_iterations(messages, 60)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "summary" in result.lower()

    def test_api_failure_returns_error(self, agent):
        agent.client.chat.completions.create.side_effect = Exception("API down")
        agent._cached_system_prompt = "You are helpful."
        messages = [{"role": "user", "content": "do stuff"}]
        result = agent._handle_max_iterations(messages, 60)
        assert isinstance(result, str)
        assert "error" in result.lower()
        assert "API down" in result

    def test_summary_skips_reasoning_for_unsupported_openrouter_model(self, agent):
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.model = "minimax/minimax-m2.5"
        resp = _mock_response(content="Summary")
        agent.client.chat.completions.create.return_value = resp
        agent._cached_system_prompt = "You are helpful."
        messages = [{"role": "user", "content": "do stuff"}]

        result = agent._handle_max_iterations(messages, 60)

        assert result == "Summary"
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert "reasoning" not in kwargs.get("extra_body", {})

    def test_summary_request_removes_orphan_tool_result(self, agent):
        """Regression: max-iterations summary request must NOT contain
        orphan tool results (tool_call_id with no matching assistant tool_call)."""
        resp = _mock_response(content="Summary of work done.")
        agent.client.chat.completions.create.return_value = resp
        agent._cached_system_prompt = "You are helpful."
        messages = [
            {"role": "user", "content": "Analyze finance-data-router"},
            {"role": "assistant", "content": "[Session Arc Summary] ..."},
            {"role": "tool", "tool_call_id": "call_cfedFhJjGmu1RvRc1OUC38j8", "content": "file content here"},
            {"role": "assistant", "tool_calls": [{"id": "call_8fXBXsT592Vpvm7wnW4obPEu", "function": {"name": "patch", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_8fXBXsT592Vpvm7wnW4obPEu", "content": "patch result"},
            {"role": "assistant", "content": "Done."},
        ]

        result = agent._handle_max_iterations(messages, 120)

        assert result == "Summary of work done."
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        sent_msgs = kwargs.get("messages", [])
        orphan_ids = [
            m.get("tool_call_id") for m in sent_msgs
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_cfedFhJjGmu1RvRc1OUC38j8"
        ]
        assert len(orphan_ids) == 0, f"Orphan tool result still present: {orphan_ids}"

    def test_summary_request_inserts_stub_for_missing_tool_result(self, agent):
        """If an assistant tool_call has no matching tool result in the
        summary request, a stub must be inserted to satisfy the API contract."""
        resp = _mock_response(content="Summary")
        agent.client.chat.completions.create.return_value = resp
        agent._cached_system_prompt = "You are helpful."
        messages = [
            {"role": "user", "content": "do stuff"},
            {"role": "assistant", "tool_calls": [{"id": "call_no_result", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "assistant", "content": "Continuing..."},
        ]

        result = agent._handle_max_iterations(messages, 60)

        assert result == "Summary"
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        sent_msgs = kwargs.get("messages", [])
        stub_ids = [
            m.get("tool_call_id") for m in sent_msgs
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_no_result"
        ]
        assert len(stub_ids) >= 1, f"No stub result for assistant tool_call: {stub_ids}"

    def test_summary_strips_strict_schema_foreign_fields(self, agent):
        """Regression: the max-iterations summary request must NOT carry
        Chat-Completions-schema-foreign keys — tool_name (SQLite FTS
        bookkeeping), codex_* reasoning carriers, or internal _-prefixed
        scaffolding. Strict gateways (Fireworks-backed OpenCode Go, Mistral,
        Kimi) reject these with 'Extra inputs are not permitted, field:
        messages[N].tool_name'. The transport's convert_messages() strips
        them on the main loop; this hand-built summary path must mirror it."""
        agent.client.chat.completions.create.return_value = _mock_response(content="Summary")
        agent._cached_system_prompt = "You are helpful."
        messages = [
            {"role": "user", "content": "do stuff"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "function": {"name": "execute_code", "arguments": "{}"}}],
                "codex_reasoning_items": [{"id": "rs_1"}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result", "tool_name": "execute_code"},
            {"role": "assistant", "content": "Done.", "_empty_recovery_synthetic": True},
        ]

        result = agent._handle_max_iterations(messages, 60)

        assert result == "Summary"
        sent_msgs = agent.client.chat.completions.create.call_args.kwargs.get("messages", [])
        for m in sent_msgs:
            assert "tool_name" not in m, m
            assert "codex_reasoning_items" not in m, m
            assert "codex_message_items" not in m, m
            assert not any(isinstance(k, str) and k.startswith("_") for k in m), m
        # Internal history is untouched — the path copies each message.
        assert messages[2]["tool_name"] == "execute_code"
        assert messages[1]["codex_reasoning_items"] == [{"id": "rs_1"}]

    def test_summary_omits_provider_preferences_for_non_openrouter(self, agent):
        agent.base_url = "https://api.openai.com/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.provider = "openai"
        agent.providers_allowed = ["Anthropic"]
        agent.client.chat.completions.create.return_value = _mock_response(content="Summary")
        agent._cached_system_prompt = "You are helpful."

        result = agent._handle_max_iterations([{"role": "user", "content": "do stuff"}], 60)

        assert result == "Summary"
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert "provider" not in kwargs.get("extra_body", {})

    def test_summary_keeps_provider_preferences_for_openrouter(self, agent):
        agent.base_url = "https://openrouter.ai/api/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.provider = "openrouter"
        agent.providers_allowed = ["Anthropic"]
        agent.client.chat.completions.create.return_value = _mock_response(content="Summary")
        agent._cached_system_prompt = "You are helpful."

        result = agent._handle_max_iterations([{"role": "user", "content": "do stuff"}], 60)

        assert result == "Summary"
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["provider"]["only"] == ["Anthropic"]

    def test_codex_summary_sanitizes_orphan_tool_results(self, agent):
        agent.api_mode = "codex_responses"
        agent.provider = "openai-codex"
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent._base_url_lower = agent.base_url.lower()
        agent._base_url_hostname = "chatgpt.com"
        agent.model = "gpt-5.5"
        agent._cached_system_prompt = "You are helpful."
        captured = {}

        def fake_run_codex_stream(kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                status="completed",
                output=[
                    SimpleNamespace(
                        type="message",
                        status="completed",
                        content=[SimpleNamespace(type="output_text", text="Summary")],
                    )
                ],
            )

        messages = [
            {"role": "user", "content": "do stuff"},
            {
                "role": "tool",
                "tool_call_id": "call_orphan",
                "content": "orphaned result from compressed history",
            },
        ]

        with patch.object(agent, "_run_codex_stream", side_effect=fake_run_codex_stream):
            result = agent._handle_max_iterations(messages, 90)

        assert result == "Summary"
        input_items = captured["input"]
        assert not any(
            item.get("type") == "function_call_output"
            and item.get("call_id") == "call_orphan"
            for item in input_items
        )

    def test_api_sanitizer_matches_responses_call_id_when_id_differs(self, agent):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "fc_123",
                        "call_id": "call_123",
                        "response_item_id": "fc_123",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "result"},
        ]

        sanitized = agent._sanitize_api_messages(messages)

        assert [m.get("tool_call_id") for m in sanitized if m.get("role") == "tool"] == [
            "call_123"
        ]


class TestRunConversation:
    """Tests for the main run_conversation method.

    Each test mocks client.chat.completions.create to return controlled
    responses, exercising different code paths without real API calls.
    """

    def _setup_agent(self, agent):
        """Common setup for run_conversation tests."""
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False

    def test_stop_finish_reason_returns_response(self, agent):
        self._setup_agent(agent)
        resp = _mock_response(content="Final answer", finish_reason="stop")
        agent.client.chat.completions.create.return_value = resp
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")
        assert result["final_response"] == "Final answer"
        assert result["completed"] is True

    def test_ollama_small_runtime_context_fails_before_api_call(self, agent, caplog):
        self._setup_agent(agent)
        agent.model = "qwen3.5:9b"
        agent.provider = "custom"
        agent.base_url = "http://host.docker.internal:11434/v1"
        agent._ollama_num_ctx = 4096

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            caplog.at_level(logging.WARNING, logger="agent.conversation_loop"),
        ):
            result = agent.run_conversation("Call ps -aux")

        assert result["failed"] is True
        assert result["completed"] is False
        assert result["api_calls"] == 0
        assert result["turn_exit_reason"] == "ollama_runtime_context_too_small"
        assert "Ollama loaded `qwen3.5:9b` with only 4,096 tokens" in result["final_response"]
        assert "model.ollama_num_ctx: 65536" in result["final_response"]
        assert not agent.client.chat.completions.create.called
        assert "Ollama runtime context too small for Hermes tool use" in caplog.text
        assert "runtime_context=4096" in caplog.text

    def test_tool_calls_then_stop(self, agent):
        self._setup_agent(agent)
        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
        resp2 = _mock_response(content="Done searching", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [resp1, resp2]
        with (
            patch("run_agent.handle_function_call", return_value="search result") as mock_handle_function_call,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("search something")
        assert result["final_response"] == "Done searching"
        assert result["api_calls"] == 2
        assert mock_handle_function_call.call_args.kwargs["tool_call_id"] == "c1"
        assert mock_handle_function_call.call_args.kwargs["session_id"] == agent.session_id

    def test_request_scoped_api_hooks_fire_for_each_api_call(self, agent):
        self._setup_agent(agent)
        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
        resp2 = _mock_response(content="Done searching", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [resp1, resp2]

        hook_calls = []

        def _record_hook(name, **kwargs):
            hook_calls.append((name, kwargs))
            return []

        with (
            patch("run_agent.handle_function_call", return_value="search result"),
            patch(
                "hermes_cli.plugins.has_hook",
                side_effect=lambda name: name in {"pre_api_request", "post_api_request"},
            ),
            patch("hermes_cli.plugins.invoke_hook", side_effect=_record_hook),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("search something")

        assert result["final_response"] == "Done searching"
        pre_request_calls = [kw for name, kw in hook_calls if name == "pre_api_request"]
        post_request_calls = [kw for name, kw in hook_calls if name == "post_api_request"]
        assert len(pre_request_calls) == 2
        assert len(post_request_calls) == 2
        assert [call["api_call_count"] for call in pre_request_calls] == [1, 2]
        assert [call["api_call_count"] for call in post_request_calls] == [1, 2]
        assert all(call["session_id"] == agent.session_id for call in pre_request_calls)
        assert all(call["turn_id"] == pre_request_calls[0]["turn_id"] for call in pre_request_calls + post_request_calls)
        assert [call["api_request_id"] for call in pre_request_calls] == [
            call["api_request_id"] for call in post_request_calls
        ]
        assert all("message_count" in c and isinstance(c.get("request_messages"), list) for c in pre_request_calls)
        assert all("request" in c and "messages" in c["request"]["body"] for c in pre_request_calls)
        assert any(msg.get("role") == "user" and msg.get("content") == "search something" for msg in pre_request_calls[0]["request_messages"])
        assert all("usage" in c and "response" in c for c in post_request_calls)
        assert all("assistant_message" in c["response"] for c in post_request_calls)

    def test_api_request_error_hook_skips_payload_work_without_listener(self, agent, monkeypatch):
        payload_built = False
        hook_called = False

        def _payload_for_hook(_api_kwargs):
            nonlocal payload_built
            payload_built = True
            return {}

        def _invoke_hook(_name, **_kwargs):
            nonlocal hook_called
            hook_called = True
            return []

        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke_hook)
        monkeypatch.setattr(agent, "_api_request_payload_for_hook", _payload_for_hook)

        agent._invoke_api_request_error_hook(
            task_id="task-1",
            turn_id="turn-1",
            api_request_id="api-1",
            api_call_count=1,
            api_start_time=0.0,
            api_kwargs={"messages": [{"role": "user", "content": "hi"}]},
            error_type="RuntimeError",
            error_message="boom",
        )

        assert payload_built is False
        assert hook_called is False

    def test_request_scoped_api_hooks_skip_payload_work_without_listeners(self, agent, monkeypatch):
        self._setup_agent(agent)
        agent.client.chat.completions.create.return_value = _mock_response(
            content="No listeners",
            finish_reason="stop",
        )
        hook_checks = {"pre_api_request": 0, "post_api_request": 0}
        payload_counts = {"request": 0, "response": 0}

        def _has_hook(name):
            if name in hook_checks:
                hook_checks[name] += 1
            return False

        def _request_payload(_api_kwargs):
            payload_counts["request"] += 1
            return {}

        def _response_payload(_response, _assistant_message, *, finish_reason):
            payload_counts["response"] += 1
            return {}

        monkeypatch.setattr("hermes_cli.plugins.has_hook", _has_hook)
        monkeypatch.setattr(agent, "_api_request_payload_for_hook", _request_payload)
        monkeypatch.setattr(agent, "_api_response_payload_for_hook", _response_payload)

        with (
            patch("hermes_cli.plugins.invoke_hook", return_value=[]),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["final_response"] == "No listeners"
        assert hook_checks == {"pre_api_request": 1, "post_api_request": 1}
        assert payload_counts == {"request": 0, "response": 0}

    def test_content_with_tool_calls_stays_silent_for_non_cli_quiet_mode(self, agent):
        self._setup_agent(agent)
        agent.platform = None
        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        resp1 = _mock_response(
            content="I'll search for that.",
            finish_reason="tool_calls",
            tool_calls=[tc],
        )
        resp2 = _mock_response(content="Done searching", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [resp1, resp2]

        with (
            patch("run_agent.handle_function_call", return_value="search result"),
            patch.object(agent, "_safe_print") as mock_print,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("search something")

        assert result["final_response"] == "Done searching"
        mock_print.assert_not_called()

    def test_interrupt_breaks_loop(self, agent):
        self._setup_agent(agent)

        def interrupt_side_effect(api_kwargs):
            agent._interrupt_requested = True
            raise InterruptedError("Agent interrupted during API call")

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent._set_interrupt"),
            patch.object(
                agent, "_interruptible_api_call", side_effect=interrupt_side_effect
            ),
        ):
            result = agent.run_conversation("hello")
        assert result["interrupted"] is True

    def test_invalid_tool_name_retry(self, agent):
        """Model hallucinates an invalid tool name, agent retries and succeeds."""
        self._setup_agent(agent)
        bad_tc = _mock_tool_call(name="nonexistent_tool", arguments="{}", call_id="c1")
        resp_bad = _mock_response(
            content="", finish_reason="tool_calls", tool_calls=[bad_tc]
        )
        resp_good = _mock_response(content="Got it", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [resp_bad, resp_good]
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("do something")
        assert result["final_response"] == "Got it"
        assert result["completed"] is True
        assert result["api_calls"] == 2

    def test_reasoning_only_local_resumed_no_compression_triggered(self, agent):
        """Reasoning-only responses no longer trigger compression — prefill then accepted."""
        self._setup_agent(agent)
        agent.base_url = "http://127.0.0.1:1234/v1"
        agent.compression_enabled = True
        empty_resp = _mock_response(
            content=None,
            finish_reason="stop",
            reasoning_content="reasoning only",
        )
        prefill = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]

        # 6 responses: original + 2 prefill + 3 retries after prefill exhaustion
        with (
            patch.object(agent, "_interruptible_api_call", side_effect=[empty_resp] * 6),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_not_called()  # no compression triggered
        assert result["completed"] is True
        # #34452: the bare "(empty)" sentinel is now replaced by a
        # user-visible end-of-turn explanation so the failure isn't silent.
        assert result["final_response"] != "(empty)"
        assert "No reply:" in result["final_response"]
        assert result["turn_exit_reason"] == "empty_response_exhausted"
        assert result["api_calls"] == 6  # 1 original + 2 prefill + 3 retries

    def test_reasoning_only_response_prefill_then_empty(self, agent):
        """Structured reasoning-only triggers prefill (2), then retries (3), then (empty)."""
        self._setup_agent(agent)
        empty_resp = _mock_response(
            content=None,
            finish_reason="stop",
            reasoning_content="structured reasoning answer",
        )
        # 6 responses: 1 original + 2 prefill + 3 retries after prefill exhaustion
        agent.client.chat.completions.create.side_effect = [empty_resp] * 6
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")
        assert result["completed"] is True
        # #34452: explanation replaces the bare "(empty)" sentinel.
        assert result["final_response"] != "(empty)"
        assert "No reply:" in result["final_response"]
        assert result["api_calls"] == 6  # 1 original + 2 prefill + 3 retries

    def test_reasoning_only_prefill_succeeds_on_continuation(self, agent):
        """When prefill continuation produces content, it becomes the final response."""
        self._setup_agent(agent)
        empty_resp = _mock_response(
            content=None,
            finish_reason="stop",
            reasoning_content="structured reasoning answer",
        )
        content_resp = _mock_response(
            content="Here is the actual answer.",
            finish_reason="stop",
        )
        agent.client.chat.completions.create.side_effect = [empty_resp, content_resp]
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")
        assert result["completed"] is True
        assert result["final_response"] == "Here is the actual answer."
        assert result["api_calls"] == 2  # 1 original + 1 prefill continuation
        # Prefill message should be cleaned up — no consecutive assistant messages
        roles = [m.get("role") for m in result["messages"]]
        for i in range(len(roles) - 1):
            if roles[i] == "assistant" and roles[i + 1] == "assistant":
                raise AssertionError("Consecutive assistant messages found in history")

    def test_truly_empty_response_retries_3_times_then_empty(self, agent):
        """Truly empty response (no content, no reasoning) retries 3 times then falls through to (empty)."""
        self._setup_agent(agent)
        agent.base_url = "http://127.0.0.1:1234/v1"
        empty_resp = _mock_response(content=None, finish_reason="stop")
        # 4 responses: 1 original + 3 nudge retries, all empty
        agent.client.chat.completions.create.side_effect = [
            empty_resp, empty_resp, empty_resp, empty_resp,
        ]
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")
        assert result["completed"] is True
        # #34452: explanation replaces the bare "(empty)" sentinel.
        assert result["final_response"] != "(empty)"
        assert "No reply:" in result["final_response"]
        assert result["api_calls"] == 4  # 1 original + 3 retries

    def test_truly_empty_response_succeeds_on_nudge(self, agent):
        """Model produces content after being nudged for empty response."""
        self._setup_agent(agent)
        agent.base_url = "http://127.0.0.1:1234/v1"
        empty_resp = _mock_response(content=None, finish_reason="stop")
        content_resp = _mock_response(
            content="Here is the actual answer.",
            finish_reason="stop",
        )
        # 1 empty response, then model produces content on nudge
        agent.client.chat.completions.create.side_effect = [empty_resp, content_resp]
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")
        assert result["completed"] is True
        assert result["final_response"] == "Here is the actual answer."
        assert result["api_calls"] == 2  # 1 original + 1 nudge retry

    def test_empty_response_triggers_fallback_provider(self, agent):
        """After 3 empty retries, fallback provider is activated and produces content."""
        self._setup_agent(agent)
        agent.base_url = "http://127.0.0.1:1234/v1"
        # Configure a fallback chain
        agent._fallback_chain = [{"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}]
        agent._fallback_index = 0
        agent._fallback_activated = False

        empty_resp = _mock_response(content=None, finish_reason="stop")
        content_resp = _mock_response(content="Fallback answer.", finish_reason="stop")
        # 4 empty (1 orig + 3 retries), then fallback model answers
        agent.client.chat.completions.create.side_effect = [
            empty_resp, empty_resp, empty_resp, empty_resp, content_resp,
        ]

        fallback_called = {"called": False}

        def _mock_fallback():
            fallback_called["called"] = True
            # Simulate what _try_activate_fallback does: just advance the
            # index and set the flag (the client is already mocked).
            agent._fallback_index = 1
            agent._fallback_activated = True
            agent.model = "anthropic/claude-sonnet-4"
            agent.provider = "openrouter"
            return True

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_try_activate_fallback", side_effect=_mock_fallback),
        ):
            result = agent.run_conversation("answer me")
        assert fallback_called["called"], "Fallback should have been triggered"
        assert result["completed"] is True
        assert result["final_response"] == "Fallback answer."

    def test_empty_response_fallback_also_empty_returns_empty(self, agent):
        """If fallback also returns empty, final response is (empty)."""
        self._setup_agent(agent)
        agent.base_url = "http://127.0.0.1:1234/v1"
        agent._fallback_chain = [{"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}]
        agent._fallback_index = 0
        agent._fallback_activated = False

        empty_resp = _mock_response(content=None, finish_reason="stop")
        # 4 empty from primary (1 + 3 retries), fallback activated,
        # then 4 more empty from fallback (1 + 3 retries), no more fallbacks
        agent.client.chat.completions.create.side_effect = [
            empty_resp, empty_resp, empty_resp, empty_resp,  # primary exhausted
            empty_resp, empty_resp, empty_resp, empty_resp,  # fallback exhausted
        ]

        def _mock_fallback():
            if agent._fallback_index >= len(agent._fallback_chain):
                return False
            agent._fallback_index += 1
            agent._fallback_activated = True
            agent.model = "anthropic/claude-sonnet-4"
            agent.provider = "openrouter"
            return True

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_try_activate_fallback", side_effect=_mock_fallback),
        ):
            result = agent.run_conversation("answer me")
        assert result["completed"] is True
        # #34452: explanation replaces the bare "(empty)" sentinel.
        assert result["final_response"] != "(empty)"
        assert "No reply:" in result["final_response"]

    def test_empty_response_emits_status_for_gateway(self, agent):
        """_emit_status is called during empty retries so gateway users see feedback."""
        self._setup_agent(agent)
        agent.base_url = "http://127.0.0.1:1234/v1"

        empty_resp = _mock_response(content=None, finish_reason="stop")
        # 4 empty: 1 original + 3 retries, all empty, no fallback
        agent.client.chat.completions.create.side_effect = [
            empty_resp, empty_resp, empty_resp, empty_resp,
        ]

        status_messages = []

        def _capture_status(msg):
            status_messages.append(msg)

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_emit_status", side_effect=_capture_status),
        ):
            result = agent.run_conversation("answer me")

        # #34452: explanation replaces the bare "(empty)" sentinel, but the
        # status emissions during retries are unchanged.
        assert result["final_response"] != "(empty)"
        assert "No reply:" in result["final_response"]
        # Should have emitted retry statuses (3 retries) + final failure
        retry_msgs = [m for m in status_messages if "retrying" in m.lower()]
        assert len(retry_msgs) == 3, f"Expected 3 retry status messages, got {len(retry_msgs)}: {status_messages}"
        failure_msgs = [m for m in status_messages if "no content" in m.lower() or "no fallback" in m.lower()]
        assert len(failure_msgs) >= 1, f"Expected at least 1 failure status, got: {status_messages}"

    def test_partial_stream_recovery_uses_streamed_content(self, agent):
        """When streaming fails after partial delivery, recovered partial content becomes final response."""
        self._setup_agent(agent)
        # Simulate a partial-stream-stub response: content recovered from streaming
        partial_resp = _mock_response(
            content="Here is the partial answer that was stream",
            finish_reason="stop",
        )
        agent.client.chat.completions.create.return_value = partial_resp
        # Simulate that streaming had already delivered this text
        agent._current_streamed_assistant_text = "Here is the partial answer that was stream"
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("explain something")
        # The partial content should be used as-is (not empty, not retried)
        assert result["completed"] is True
        assert result["final_response"] == "Here is the partial answer that was stream"
        assert result["api_calls"] == 1  # No retries

    def test_partial_stream_recovery_on_empty_stub(self, agent):
        """When stub response has no content but text was streamed, use streamed text."""
        self._setup_agent(agent)
        # Stub response with no content (old behavior before fix)
        empty_stub = _mock_response(content=None, finish_reason="stop")

        def _fake_api_call(api_kwargs):
            # Simulate what streaming does: accumulate text before returning
            # a stub with no content (connection died mid-stream)
            agent._current_streamed_assistant_text = "The answer to your question is that"
            return empty_stub

        status_messages = []

        def _capture_status(msg):
            status_messages.append(msg)

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_emit_status", side_effect=_capture_status),
        ):
            result = agent.run_conversation("ask me")
        # Should recover partial streamed content, not fall through to (empty)
        assert result["completed"] is True
        # 2026-06-29 upstream merge: partial_stream_recovery now appends a
        # user-facing "send `continue`" notice. The load-bearing check is that the
        # partial content was recovered (not lost / not "(empty)").
        assert result["final_response"].startswith("The answer to your question is that")
        assert "streaming stopped early" in result["final_response"]
        assert result["api_calls"] == 1  # No wasted retries
        # Should emit the stream-interrupted status, NOT the empty-retry status
        recovery_msgs = [m for m in status_messages if "stream interrupted" in m.lower()]
        assert len(recovery_msgs) >= 1, f"Expected stream recovery status, got: {status_messages}"
        # Should NOT have retry statuses
        retry_msgs = [m for m in status_messages if "retrying" in m.lower()]
        assert len(retry_msgs) == 0, f"Should not retry when stream content exists: {status_messages}"

    def test_partial_stream_recovery_preempts_prior_turn_fallback(self, agent):
        """Partial streamed content takes priority over _last_content_with_tools fallback."""
        self._setup_agent(agent)
        # Set up the prior-turn fallback content (from a previous turn with tool calls)
        agent._last_content_with_tools = "Old content from prior turn with tools"
        # Stub response with no content
        empty_stub = _mock_response(content=None, finish_reason="stop")

        def _fake_api_call(api_kwargs):
            # Simulate partial streaming before connection death
            agent._current_streamed_assistant_text = "Fresh partial content from this turn"
            return empty_stub

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("question")
        # Should use the streamed content, not the old prior-turn fallback
        assert result["final_response"].startswith("Fresh partial content from this turn")
        assert "streaming stopped early" in result["final_response"]
        assert result["api_calls"] == 1

    def test_nous_401_refreshes_after_remint_and_retries(self, agent):
        self._setup_agent(agent)
        agent.provider = "nous"
        agent.api_mode = "chat_completions"

        calls = {"api": 0, "refresh": 0}

        class _UnauthorizedError(RuntimeError):
            def __init__(self):
                super().__init__("Error code: 401 - unauthorized")
                self.status_code = 401

        def _fake_api_call(api_kwargs):
            calls["api"] += 1
            if calls["api"] == 1:
                raise _UnauthorizedError()
            return _mock_response(
                content="Recovered after remint", finish_reason="stop"
            )

        def _fake_refresh(*, force=True):
            calls["refresh"] += 1
            assert force is True
            return True

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),
            patch.object(
                agent, "_try_refresh_nous_client_credentials", side_effect=_fake_refresh
            ),
        ):
            result = agent.run_conversation("hello")

        assert calls["api"] == 2
        assert calls["refresh"] == 1
        assert result["completed"] is True
        assert result["final_response"] == "Recovered after remint"

    def test_context_compression_triggered(self, agent):
        """When compressor says should_compress, compression runs."""
        self._setup_agent(agent)
        agent.compression_enabled = True

        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
        resp2 = _mock_response(content="All done", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [resp1, resp2]

        with (
            patch("run_agent.handle_function_call", return_value="result"),
            patch.object(
                agent.context_compressor, "should_compress", return_value=True
            ),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # _compress_context should return (messages, system_prompt)
            mock_compress.return_value = (
                [{"role": "user", "content": "search something"}],
                "compressed system prompt",
            )
            result = agent.run_conversation("search something")
        mock_compress.assert_called_once()
        assert result["final_response"] == "All done"
        assert result["completed"] is True

    def test_glm_prompt_exceeds_max_length_triggers_compression(self, agent):
        """GLM/Z.AI uses 'Prompt exceeds max length' for context overflow."""
        self._setup_agent(agent)
        agent.compression_enabled = True  # this test verifies overflow→compression fires
        err_400 = Exception(
            "Error code: 400 - {'error': {'code': '1261', 'message': 'Prompt exceeds max length'}}"
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
                "compressed system prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["final_response"] == "Recovered after compression"
        assert result["completed"] is True

    def test_minimax_delta_overflow_keeps_known_context_length(self, agent):
        """MiniMax reports overflow deltas like 'limit (2013)' without the real window.

        Keep the known 204,800-token window and compress instead of probing down
        to the generic 128K fallback tier.
        """
        self._setup_agent(agent)
        agent.compression_enabled = True  # this test verifies overflow→compression fires
        agent.provider = "minimax"
        agent.model = "MiniMax-M2.7-highspeed"
        agent.base_url = "https://api.minimax.io/anthropic"
        agent.context_compressor.context_length = 204_800
        agent.context_compressor.threshold_tokens = int(
            agent.context_compressor.context_length * agent.context_compressor.threshold_percent
        )

        err_400 = Exception(
            "HTTP 400: invalid params, context window exceeds limit (2013)"
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
                "compressed system prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert agent.context_compressor.context_length == 204_800
        assert agent.context_compressor._context_probed is False
        assert result["final_response"] == "Recovered after compression"
        assert result["completed"] is True

    def test_non_minimax_overflow_without_provider_limit_keeps_context(self, agent):
        """Generic overflow without a provider-reported max must NOT probe-step down.

        Previously a 200K configured window would silently drop to the 128K probe
        tier on a generic overflow error.  Now we keep the configured window and
        rely on compression — see #33669 / PR #33826.
        """
        self._setup_agent(agent)
        agent.compression_enabled = True  # this test verifies overflow→compression fires
        agent.provider = "openrouter"
        agent.model = "some/unknown-model"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = int(
            agent.context_compressor.context_length * agent.context_compressor.threshold_percent
        )

        err_400 = Exception(
            "HTTP 400: invalid params, context window exceeds limit (2013)"
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
                "compressed system prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        # Context length preserved — no guessed probe-tier step-down.
        assert agent.context_compressor.context_length == 200_000
        assert result["final_response"] == "Recovered after compression"
        assert result["completed"] is True

    def test_length_finish_reason_requests_continuation(self, agent):
        """Normal truncation (partial real content) triggers continuation."""
        self._setup_agent(agent)
        first = _mock_response(content="Part 1 ", finish_reason="length")
        second = _mock_response(content="Part 2", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [first, second]

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["api_calls"] == 2
        assert result["final_response"] == "Part 1 Part 2"

        second_call_messages = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
        assert second_call_messages[-1]["role"] == "user"
        assert "truncated by the output length limit" in second_call_messages[-1]["content"]

    def test_length_continuation_preserves_large_provider_default_output_cap(self, agent):
        """Continuation retries must not shrink a higher provider default cap."""
        self._setup_agent(agent)
        agent.max_tokens = None
        requested_caps = []

        def _fake_build_api_kwargs(api_messages):
            ephemeral = getattr(agent, "_ephemeral_max_output_tokens", None)
            if ephemeral is not None:
                agent._ephemeral_max_output_tokens = None
            cap = ephemeral if ephemeral is not None else 65536
            requested_caps.append(cap)
            return {"model": agent.model, "messages": api_messages, "max_tokens": cap}

        first = _mock_response(content="Part 1 ", finish_reason="length")
        second = _mock_response(content="Part 2", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [first, second]

        with (
            patch.object(agent, "_build_api_kwargs", side_effect=_fake_build_api_kwargs),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "Part 1 Part 2"
        assert requested_caps == [65536, 65536]

    def test_ollama_glm_stop_after_tools_without_terminal_boundary_requests_continuation(self, agent):
        """Ollama-hosted GLM responses can misreport truncated output as stop."""
        self._setup_agent(agent)
        agent.base_url = "http://localhost:11434/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = "glm-5.1:cloud"

        tool_turn = _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id="c1")],
        )
        misreported_stop = _mock_response(
            content="Based on the search results, the best next",
            finish_reason="stop",
        )
        continued = _mock_response(
            content=" step is to update the config.",
            finish_reason="stop",
        )
        agent.client.chat.completions.create.side_effect = [
            tool_turn,
            misreported_stop,
            continued,
        ]

        with (
            patch("run_agent.handle_function_call", return_value="search result"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["api_calls"] == 3
        assert (
            result["final_response"]
            == "Based on the search results, the best next step is to update the config."
        )

        third_call_messages = agent.client.chat.completions.create.call_args_list[2].kwargs["messages"]
        assert third_call_messages[-1]["role"] == "user"
        assert "truncated by the output length limit" in third_call_messages[-1]["content"]

    def test_ollama_glm_stop_with_terminal_boundary_does_not_continue(self, agent):
        """Complete Ollama/GLM responses should not be reclassified as truncated."""
        self._setup_agent(agent)
        agent.base_url = "http://localhost:11434/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = "glm-5.1:cloud"

        tool_turn = _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id="c1")],
        )
        complete_stop = _mock_response(
            content="Based on the search results, the best next step is to update the config.",
            finish_reason="stop",
        )
        agent.client.chat.completions.create.side_effect = [tool_turn, complete_stop]

        with (
            patch("run_agent.handle_function_call", return_value="search result"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["api_calls"] == 2
        assert (
            result["final_response"]
            == "Based on the search results, the best next step is to update the config."
        )

    def test_non_ollama_stop_without_terminal_boundary_does_not_continue(self, agent):
        """The stop->length workaround should stay scoped to Ollama/GLM backends."""
        self._setup_agent(agent)
        agent.base_url = "https://api.openai.com/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = "gpt-4o-mini"

        tool_turn = _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id="c1")],
        )
        normal_stop = _mock_response(
            content="Based on the search results, the best next",
            finish_reason="stop",
        )
        agent.client.chat.completions.create.side_effect = [tool_turn, normal_stop]

        with (
            patch("run_agent.handle_function_call", return_value="search result"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["api_calls"] == 2
        assert result["final_response"] == "Based on the search results, the best next"

    def test_length_thinking_exhausted_skips_continuation(self, agent):
        """When finish_reason='length' but content is only thinking, skip retries."""
        self._setup_agent(agent)
        resp = _mock_response(
            content="<think>internal reasoning</think>",
            finish_reason="length",
        )
        agent.client.chat.completions.create.return_value = resp

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        # Should return immediately — no continuation, only 1 API call
        assert result["completed"] is False
        assert result["api_calls"] == 1
        assert "reasoning" in result["error"].lower()
        assert "output tokens" in result["error"].lower()
        # Should have a user-friendly response (not None)
        assert result["final_response"] is not None
        assert "Thinking Budget Exhausted" in result["final_response"]
        assert "/thinkon" in result["final_response"]

    def test_length_empty_content_without_think_tags_retries_normally(self, agent):
        """When finish_reason='length' and content is None but no think tags,
        fall through to normal continuation retry (not thinking-exhaustion)."""
        self._setup_agent(agent)
        resp = _mock_response(content=None, finish_reason="length")
        agent.client.chat.completions.create.return_value = resp

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        # Without think tags, the agent should attempt continuation retries
        # (up to 3), not immediately fire thinking-exhaustion.
        assert result["api_calls"] == 3
        assert result["completed"] is False

    def test_length_with_tool_calls_returns_partial_without_executing_tools(self, agent):
        self._setup_agent(agent)
        bad_tc = _mock_tool_call(
            name="write_file",
            arguments='{"path":"report.md","content":"partial',
            call_id="c1",
        )
        resp = _mock_response(content="", finish_reason="length", tool_calls=[bad_tc])
        agent.client.chat.completions.create.return_value = resp

        with (
            patch("run_agent.handle_function_call") as mock_handle_function_call,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("write the report")

        assert result["completed"] is False
        assert result["partial"] is True
        assert "truncated due to output length limit" in result["error"]
        mock_handle_function_call.assert_not_called()

    def test_truncated_tool_call_retries_once_before_refusing(self, agent):
        """When tool call args are truncated, the agent retries the API call
        (up to 3 times). If a retry succeeds (valid JSON args), tool execution
        proceeds."""
        self._setup_agent(agent)
        agent.valid_tool_names.add("write_file")
        bad_tc = _mock_tool_call(
            name="write_file",
            arguments='{"path":"report.md","content":"partial',
            call_id="c1",
        )
        truncated_resp = _mock_response(
            content="", finish_reason="length", tool_calls=[bad_tc],
        )
        good_tc = _mock_tool_call(
            name="write_file",
            arguments='{"path":"report.md","content":"full content"}',
            call_id="c2",
        )
        good_resp = _mock_response(
            content="", finish_reason="stop", tool_calls=[good_tc],
        )
        with (
            patch("run_agent.handle_function_call", return_value='{"success":true}') as mock_hfc,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # First call: truncated → retry. Second: valid → execute tool.
            # Third: final text response.
            final_resp = _mock_response(content="Done!", finish_reason="stop")
            agent.client.chat.completions.create.side_effect = [
                truncated_resp, good_resp, final_resp,
            ]
            result = agent.run_conversation("write the report")

        # Tool was executed on the retry (good_resp)
        mock_hfc.assert_called_once()
        assert result["final_response"] == "Done!"

    def test_stub_stall_mid_tool_call_recovers_within_3_retries(self, agent):
        """A network stream stall mid tool-call (PARTIAL_STREAM_STUB_ID) must
        retry up to 3 times rather than hard-failing after one — and recover
        if a retry produces a complete tool call. Regression for the false
        'model hit max output tokens' on Opus when the stream simply dropped."""
        from hermes_constants import PARTIAL_STREAM_STUB_ID

        self._setup_agent(agent)
        agent.valid_tool_names.add("write_file")
        bad_tc = _mock_tool_call(
            name="write_file",
            arguments='{"path":"report.md","content":"partial',
            call_id="c1",
        )
        # Two consecutive stub-stall responses, then a clean tool call.
        stall1 = _mock_response(content="", finish_reason="length", tool_calls=[bad_tc])
        stall1.id = PARTIAL_STREAM_STUB_ID
        stall2 = _mock_response(content="", finish_reason="length", tool_calls=[bad_tc])
        stall2.id = PARTIAL_STREAM_STUB_ID
        good_tc = _mock_tool_call(
            name="write_file",
            arguments='{"path":"report.md","content":"full content"}',
            call_id="c2",
        )
        good_resp = _mock_response(content="", finish_reason="stop", tool_calls=[good_tc])
        final_resp = _mock_response(content="Done!", finish_reason="stop")

        with (
            patch("run_agent.handle_function_call", return_value='{"success":true}') as mock_hfc,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.client.chat.completions.create.side_effect = [
                stall1, stall2, good_resp, final_resp,
            ]
            result = agent.run_conversation("write the report")

        # Recovered on the 3rd attempt instead of refusing after the 1st.
        mock_hfc.assert_called_once()
        assert result["final_response"] == "Done!"

    def test_truncated_tool_args_detected_when_finish_reason_not_length(self, agent):
        """When a router rewrites finish_reason from 'length' to 'tool_calls',
        truncated JSON arguments should still be detected and refused rather
        than wasting 3 retry attempts."""
        self._setup_agent(agent)
        agent.valid_tool_names.add("write_file")
        bad_tc = _mock_tool_call(
            name="write_file",
            arguments='{"path":"report.md","content":"partial',
            call_id="c1",
        )
        resp = _mock_response(
            content="", finish_reason="tool_calls", tool_calls=[bad_tc],
        )
        agent.client.chat.completions.create.return_value = resp

        with (
            patch("run_agent.handle_function_call") as mock_handle_function_call,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("write the report")

        assert result["completed"] is False
        assert result["partial"] is True
        assert "truncated due to output length limit" in result["error"]
        mock_handle_function_call.assert_not_called()

    def test_kanban_block_called_on_iteration_exhaustion(self, agent, monkeypatch):
        """Regression: kanban worker must signal the dispatcher when its
        iteration budget is exhausted, otherwise the task silently re-runs
        forever without ever tripping the failure_limit circuit breaker
        (issue #23216 / #29747 gap 2).

        As of #29747, the exhaustion path routes through
        ``kanban_db._record_task_failure(outcome="timed_out")`` so the
        ``consecutive_failures`` counter increments and the dispatcher's
        ``failure_limit`` breaker eventually trips. The legacy
        ``kanban_block`` call was replaced because blocked-outcome runs
        bypass the failure counter.
        """
        self._setup_agent(agent)
        agent.max_iterations = 2

        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test_task_123")

        # Return a tool call for every iteration to exhaust the budget.
        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        tool_resp = _mock_response(
            content="", finish_reason="tool_calls", tool_calls=[tc],
        )
        # Final summary response from _handle_max_iterations.
        summary_resp = _mock_response(
            content="Could not finish — budget exhausted.", finish_reason="stop",
        )
        agent.client.chat.completions.create.side_effect = [
            tool_resp, tool_resp, summary_resp,
        ]

        mock_record_failure = MagicMock(return_value=False)
        mock_connect = MagicMock(return_value=MagicMock())

        with (
            patch("run_agent.handle_function_call", return_value="ok"),
            patch("hermes_cli.kanban_db._record_task_failure",
                  mock_record_failure),
            patch("hermes_cli.kanban_db.connect", mock_connect),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("do the kanban work")

        # The agent should have reported the task as not completed.
        assert result["completed"] is False

        # _record_task_failure should have been called exactly once for
        # the exhaustion event, with outcome="timed_out".
        assert mock_record_failure.call_count == 1, (
            f"Expected exactly 1 _record_task_failure call, "
            f"got {mock_record_failure.call_count}. "
            f"Calls: {mock_record_failure.call_args_list}"
        )
        call = mock_record_failure.call_args_list[0]
        # Positional: (conn, task_id, ...)
        assert call.args[1] == "t_test_task_123"
        assert call.kwargs.get("outcome") == "timed_out"
        assert call.kwargs.get("release_claim") is True
        assert call.kwargs.get("end_run") is True
        assert "Iteration budget exhausted" in call.kwargs.get("error", "")

    def test_no_kanban_block_when_not_in_kanban_mode(self, agent, monkeypatch):
        """The exhaustion bridge must NOT fire when HERMES_KANBAN_TASK
        is unset (non-kanban runs are unaffected by #29747 gap 2)."""
        self._setup_agent(agent)
        agent.max_iterations = 2

        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        tool_resp = _mock_response(
            content="", finish_reason="tool_calls", tool_calls=[tc],
        )
        summary_resp = _mock_response(
            content="Summary.", finish_reason="stop",
        )
        agent.client.chat.completions.create.side_effect = [
            tool_resp, tool_resp, summary_resp,
        ]

        mock_record_failure = MagicMock(return_value=False)

        with (
            patch("run_agent.handle_function_call", return_value="ok"),
            patch("hermes_cli.kanban_db._record_task_failure",
                  mock_record_failure),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation("do stuff")

        assert mock_record_failure.call_count == 0, (
            "_record_task_failure should not be called outside kanban mode"
        )


class TestRetryExhaustion:
    """Regression: retry_count > max_retries was dead code (off-by-one).

    When retries were exhausted the condition never triggered, causing
    the loop to exit and fall through to response.choices[0] on an
    invalid response, raising IndexError.
    """

    def _setup_agent(self, agent):
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False

    @staticmethod
    def _make_fast_time_mock():
        """Return a mock time module where sleep loops exit instantly."""
        mock_time = MagicMock()
        _t = [1000.0]

        def _advancing_time():
            _t[0] += 500.0  # jump 500s per call so sleep_end is always in the past
            return _t[0]

        mock_time.time.side_effect = _advancing_time
        mock_time.sleep = MagicMock()  # no-op
        mock_time.monotonic.return_value = 12345.0
        return mock_time

    def test_invalid_response_returns_error_not_crash(self, agent):
        """Exhausted retries on invalid (empty choices) response must not IndexError."""
        self._setup_agent(agent)
        # Return response with empty choices every time
        bad_resp = SimpleNamespace(
            choices=[],
            model="test/model",
            usage=None,
        )
        agent.client.chat.completions.create.return_value = bad_resp
        # The conversation loop was extracted out of run_agent.py and pulls
        # in time/jittered_backoff at module level — patch BOTH so the
        # retry waits don't burn 18+ seconds of real wall-clock time here.
        from agent import conversation_loop as _conv_loop
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.time", self._make_fast_time_mock()),
            patch.object(_conv_loop, "time", self._make_fast_time_mock()),
            patch.object(_conv_loop, "jittered_backoff", lambda *a, **k: 0.0),
        ):
            result = agent.run_conversation("hello")
        assert result.get("completed") is False, (
            f"Expected completed=False, got: {result}"
        )
        assert result.get("failed") is True
        assert "error" in result
        assert "Invalid API response" in result["error"]

    def test_content_filter_refusal_surfaced_not_retried(self, agent):
        """A model refusal must be surfaced immediately, NOT laundered into
        the empty-response retry loop and reported as "rate limited" / "no
        content after retries".

        Regression: running a Claude refusal through an OpenAI-compatible
        portal (Nous Portal fronting Anthropic) returns ``message.refusal``
        with empty content. The transport now promotes that to a
        ``content_filter`` finish reason and the loop surfaces it as a terminal
        ``content_policy_blocked`` result instead of retrying a deterministic
        refusal three times.
        """
        self._setup_agent(agent)
        refusal_resp = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content=None, tool_calls=None, reasoning=None,
                    reasoning_content=None, refusal="I won't help with that.",
                ),
                finish_reason="stop",
            )],
            model="test/model",
            usage=None,
            id="resp_1",
        )
        agent.client.chat.completions.create.return_value = refusal_resp
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("please do something disallowed")
        assert result.get("completed") is False
        assert result.get("failed") is True
        assert "content_policy_blocked" in result.get("error", "")
        # The model's refusal text is surfaced to the user, not swallowed.
        assert "I won't help with that." in (result.get("final_response") or "")
        # Crucial regression guard: a deterministic refusal is NOT retried —
        # exactly one API call, no empty-response retry loop.
        assert agent.client.chat.completions.create.call_count == 1

    def test_api_error_returns_gracefully_after_retries(self, agent):
        """Exhausted retries on API errors must return error result, not crash."""
        self._setup_agent(agent)
        agent.client.chat.completions.create.side_effect = RuntimeError("rate limited")
        from agent import conversation_loop as _conv_loop
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.time", self._make_fast_time_mock()),
            patch.object(_conv_loop, "time", self._make_fast_time_mock()),
            patch.object(_conv_loop, "jittered_backoff", lambda *a, **k: 0.0),
        ):
            result = agent.run_conversation("hello")
        assert result.get("completed") is False
        assert result.get("failed") is True
        assert "error" in result
        assert "rate limited" in result["error"]

    def test_build_api_kwargs_error_no_unbound_local(self, agent):
        """When _build_api_kwargs raises, except handler must not crash with UnboundLocalError.

        Regression: _dump_api_request_debug(api_kwargs, ...) in the except block
        referenced api_kwargs before it was assigned when _build_api_kwargs threw.
        """
        self._setup_agent(agent)
        with (
            patch.object(agent, "_build_api_kwargs", side_effect=ValueError("bad messages")),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.time", self._make_fast_time_mock()),
        ):
            result = agent.run_conversation("hello")
        # Must surface the real error, not UnboundLocalError
        assert result.get("completed") is False
        assert result.get("failed") is True
        assert "error" in result
        assert "UnboundLocalError" not in result.get("error", "")
        assert "bad messages" in result["error"]


class TestConversationHistoryNotMutated:
    """run_conversation must not mutate the caller's conversation_history list."""

    def test_caller_list_unchanged_after_run(self, agent):
        """Passing conversation_history should not modify the original list."""
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        original_len = len(history)

        resp = _mock_response(content="new answer", finish_reason="stop")
        agent.client.chat.completions.create.return_value = resp

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "new question", conversation_history=history
            )

        # Caller's list must be untouched
        assert len(history) == original_len, (
            f"conversation_history was mutated: expected {original_len} items, got {len(history)}"
        )
        # Result should have more messages than the original history
        assert len(result["messages"]) > original_len


class TestBudgetPressure:
    """Budget exhaustion grace call system."""

    def test_grace_call_flags_initialized(self, agent):
        """Agent should have budget grace call flags."""
        assert agent._budget_exhausted_injected is False
        assert agent._budget_grace_call is False


class TestDeadRetryCode:
    """Unreachable retry_count >= max_retries after raise must not exist."""

    def test_no_unreachable_max_retries_after_backoff(self):
        import inspect
        from agent.conversation_loop import run_conversation as _rc
        source = inspect.getsource(_rc)
        occurrences = source.count("if retry_count >= max_retries:")
        assert occurrences == 2, (
            f"Expected 2 occurrences of 'if retry_count >= max_retries:' "
            f"but found {occurrences}"
        )
