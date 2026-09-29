"""test_run_agent_conversation: split from the former tests/agent/test_run_agent.py monolith (#99982)."""
import json
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from agent.error_classifier import FailoverReason
from tui_gateway import server as tui_server
from tests.agent._run_agent_helpers import _mock_response, _mock_tool_call


class TestHydrateTodoStore:
    @staticmethod
    def _assistant_todo_call(call_id="c1", name="todo", arguments="{}"):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }

    @pytest.mark.parametrize(
        "name,arguments",
        [
            ("todo_list", "{}"),
            ("tool_call", json.dumps({"calls": [{"name": "todo_list", "arguments": {}}]})),
        ],
        ids=["direct", "bridged"],
    )
    def test_todo_list_name_hydrates(self, agent, name, arguments):
        """Regression for #124960: the current name and its tool_call-bridged form pair like legacy ``todo``."""
        todos = [{"id": "t", "content": "Task", "status": "pending"}]
        history = [
            self._assistant_todo_call(name=name, arguments=arguments),
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"todos": todos, "revision": 3})},
        ]

        with patch("run_agent._set_interrupt"), patch("agent.interrupt_control._set_interrupt"):
            agent._hydrate_todo_store(history)

        assert agent._todo_store.snapshot() == {"todos": todos, "revision": 3}
        # The TUI resume path (no AIAgent yet) must pair the same call via the same predicate.
        assert tui_server._todo_state_from_history(history)["todos"] == todos

    def test_no_todo_in_history(self, agent):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        with patch("run_agent._set_interrupt"), patch("agent.interrupt_control._set_interrupt"):
            agent._hydrate_todo_store(history)
        assert not agent._todo_store.has_items()

    def test_newer_live_revision_wins_over_history(self, agent):
        agent._todo_store.restore(
            [{"id": "db", "content": "Current", "status": "in_progress"}],
            revision=5,
        )
        history = [
            self._assistant_todo_call(),
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": json.dumps(
                    {
                        "todos": [
                            {"id": "old", "content": "Old", "status": "pending"}
                        ],
                        "revision": 4,
                    }
                ),
            },
        ]

        with patch("run_agent._set_interrupt"), patch("agent.interrupt_control._set_interrupt"):
            agent._hydrate_todo_store(history)

        assert agent._todo_store.snapshot()["revision"] == 5
        assert agent._todo_store.read()[0]["id"] == "db"

    def test_history_recovers_newer_snapshot(self, agent):
        history = [
            self._assistant_todo_call(),
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": json.dumps(
                    {
                        "todos": [
                            {"id": "new", "content": "Recovered", "status": "pending"}
                        ],
                        "revision": 2,
                    }
                ),
            },
        ]

        with patch("run_agent._set_interrupt"), patch("agent.interrupt_control._set_interrupt"):
            agent._hydrate_todo_store(history)

        assert agent._todo_store.snapshot()["revision"] == 2
        assert agent._todo_store.read()[0]["id"] == "new"

class TestRetryAfterCap:
    """The loop honors provider cooldowns up to a 600-second ceiling.

    This covers rate-limit headers (#26293) and retryable 5xx responses.
    """

    @staticmethod
    def _retryable_error(status_code, headers, body=None):
        """A provider error carrying optional Retry-After surfaces."""
        message = (
            "Error code: 429 - Rate limit exceeded."
            if status_code == 429
            else f"Error code: {status_code} - origin response timeout"
        )

        class _ProviderError(Exception):
            def __init__(self):
                super().__init__(message)
                self.status_code = status_code
                self.response = SimpleNamespace(headers=headers)
                if body is not None:
                    self.body = body

        return _ProviderError()

    def _drive_once(self, agent, error, status_marker):
        """Raise ``error`` from the API call and capture the backoff status the
        loop chose. Interrupt during the backoff sleep so the test doesn't
        actually wait, and return the status string reporting the wait."""

        def _fake_api_call(api_kwargs):
            raise error

        agent._interruptible_api_call = _fake_api_call
        agent._persist_session = lambda *args, **kwargs: None
        agent._save_trajectory = lambda *args, **kwargs: None

        captured = []
        original_buffer = agent._buffer_status
        original_emit = agent._emit_status

        def _capture_status(msg, *args, **kwargs):
            captured.append((msg, "buffer"))
            # Break out of the backoff sleep immediately rather than blocking
            # for the full Retry-After window.
            if status_marker in msg:
                agent._interrupt_requested = True
            return original_buffer(msg, *args, **kwargs)

        def _capture_emit(msg):
            captured.append((msg, "emit"))
            if status_marker in msg:
                agent._interrupt_requested = True
            return original_emit(msg)

        agent._buffer_status = _capture_status
        agent._emit_status = _capture_emit
        agent.run_conversation("hello")
        return next(((m, s) for m, s in captured if status_marker in m), ("", ""))

    def test_retry_after_under_cap_is_honored(self, agent):
        # 300s > old 120s cap but < new 600s cap → used verbatim.
        error = self._retryable_error(429, {"retry-after": "300"})
        status, _ = self._drive_once(agent, error, "Waiting")
        assert "Waiting 300.0s" in status

    @pytest.mark.parametrize(
        ("headers", "body", "expected_wait", "expected_surface"),
        [
            # Long cooldowns (> 60s) surface immediately...
            ({"Retry-After": "120"}, {}, "120.0", "emit"),
            ({}, {"status": 524, "retry_after": 120}, "120.0", "emit"),
            ({}, {"status": 524, "error": {"retry_after": 120}}, "120.0", "emit"),
            # Above the 600s ceiling → capped, never used verbatim.
            ({"Retry-After": "3600"}, {}, "600.0", "emit"),
            # ...short cooldowns keep the buffered status line.
            ({"Retry-After": "30"}, {}, "30.0", "buffer"),
            # No cooldown on header or body → falls through to jittered
            # backoff (patched to 0.0 by the conftest fixture), no crash.
            ({}, {"status": 524}, "0.0", "buffer"),
        ],
        ids=(
            "header",
            "problem-detail-body",
            "nested-problem-detail-body",
            "over-cap-is-capped",
            "short-cooldown-is-buffered",
            "no-cooldown-falls-back",
        ),
    )
    def test_retry_after_on_cloudflare_524_is_honored(
        self, agent, headers, body, expected_wait, expected_surface
    ):
        """A retryable 5xx must not bypass the provider's cooldown."""
        error = self._retryable_error(524, headers, body)
        status, surface = self._drive_once(agent, error, "Retrying in")
        assert f"Retrying in {expected_wait}s" in status
        assert surface == expected_surface

class TestHandleMaxIterations:
    @pytest.mark.parametrize("api_mode,platform", [
        ("chat_completions", "cli"), ("chat_completions", "cron"),
        ("anthropic_messages", "cli"),
    ])
    def test_summary_interrupt_aborts_only_its_request(self, agent, monkeypatch, api_mode, platform):
        agent.api_mode = api_mode
        agent.platform = platform
        agent._cached_system_prompt = "You are helpful."
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        request_client = MagicMock()
        aborted = []

        def blocked(*args, **kwargs):
            entered.set()
            release.wait(10)
            raise OSError("fixture request stopped")

        def abort(client, **kwargs):
            aborted.append(client)
            release.set()

        agent.client.chat.completions.create.side_effect = blocked
        request_client.chat.completions.create.side_effect = blocked
        monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: request_client)
        monkeypatch.setattr(agent, "_create_request_anthropic_client", lambda **kw: request_client)
        monkeypatch.setattr(agent, "_abort_request_openai_client", abort)
        monkeypatch.setattr(agent, "_abort_request_anthropic_client", abort)
        monkeypatch.setattr(agent, "_close_request_openai_client", lambda *a, **kw: None)
        monkeypatch.setattr(agent, "_close_request_anthropic_client", lambda *a, **kw: None)
        if api_mode == "anthropic_messages":
            agent._is_anthropic_oauth = False
            transport = SimpleNamespace(build_kwargs=lambda **kw: {"model": "fixture", "messages": kw["messages"]})
            monkeypatch.setattr(agent, "_get_transport", lambda: transport)
            monkeypatch.setattr(agent, "_anthropic_messages_create", blocked)

        raised = []

        def summarize():
            try:
                agent._handle_max_iterations([{"role": "user", "content": "work"}], 1)
            except InterruptedError as exc:
                raised.append(exc)
            finally:
                finished.set()

        worker = threading.Thread(target=summarize)
        worker.start()
        try:
            assert entered.wait(5), "summary did not reach provider fixture"
            agent.interrupt()
            assert finished.wait(4), "summary ignored interrupt while provider was blocked"
            assert aborted == [request_client]
            assert len(raised) == 1, "summary cancellation must propagate, not become a fallback"
            agent.client.close.assert_not_called()
        finally:
            release.set()
            worker.join(12)

    def test_interrupted_summary_ends_turn_interrupted_and_keeps_pending_message(self, agent, monkeypatch):
        from agent.context_compressor import MAX_ITERATIONS_SUMMARY_REQUEST
        from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent.max_iterations = 1
        tool_resp = _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id="c1")],
        )
        release = threading.Event()
        calls = []

        def provider(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return tool_resp
            # The summary request: a new user message arrives while it is in flight.
            agent.interrupt("follow-up message")
            release.wait(10)
            raise OSError("fixture request stopped")

        request_client = MagicMock()
        request_client.chat.completions.create.side_effect = provider
        agent.client.chat.completions.create.side_effect = provider
        monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: request_client)
        monkeypatch.setattr(agent, "_abort_request_openai_client", lambda *a, **kw: release.set())
        monkeypatch.setattr(agent, "_close_request_openai_client", lambda *a, **kw: None)

        try:
            with (
                patch("model_tools.handle_function_call", return_value="ok"),
                patch.object(agent, "_persist_session"),
                patch.object(agent, "_save_trajectory"),
                patch.object(agent, "_cleanup_task_resources"),
            ):
                result = agent.run_conversation("do the work")
        finally:
            release.set()

        assert len(calls) == 2, "summary request never reached the provider fixture"
        assert result["interrupted"] is True
        assert result["completed"] is False
        assert result["interrupt_message"] == "follow-up message"
        assert result["turn_exit_reason"].startswith("interrupted_during_api_call")
        assert result["final_response"].startswith(INTERRUPT_WAITING_FOR_MODEL_PREFIX)
        assert "couldn't produce a summary" not in result["final_response"]
        assert all(m.get("content") != MAX_ITERATIONS_SUMMARY_REQUEST for m in result["messages"])

    def test_summary_notice_uses_safe_print(self, agent):
        agent._print_fn = lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("closed"))
        agent.client.chat.completions.create.return_value = _mock_response(content="Summary")
        agent._cached_system_prompt = "You are helpful."

        assert agent._handle_max_iterations([{"role": "user", "content": "do stuff"}], 60) == "Summary"


    def test_summary_retries_share_relay_identity(self, agent):
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content=""),
            _mock_response(content="Summary"),
        ]
        agent._cached_system_prompt = "You are helpful."
        relay_calls = []

        def execute_current(request, callback, **kwargs):
            relay_calls.append(kwargs)
            return callback(request)

        with (
            patch("agent.relay_llm.execute_current", side_effect=execute_current),
            patch("agent.relay_llm.complete_logical_call") as complete_logical,
        ):
            result = agent._handle_max_iterations(
                [{"role": "user", "content": "do stuff"}],
                60,
            )

        assert result == "Summary"
        assert [call["metadata"]["retry_count"] for call in relay_calls] == [0, 1]
        assert relay_calls[0]["metadata"]["api_request_id"] == (
            relay_calls[1]["metadata"]["api_request_id"]
        )
        assert relay_calls[0]["metadata"]["call_role"] == "iteration_summary"
        assert all(call["defer_logical_completion"] is True for call in relay_calls)
        complete_logical.assert_called_once_with(
            relay_calls[0]["metadata"]["api_request_id"],
            outcome="success",
        )

    def test_suppress_status_output_keeps_iteration_warning_off_stdout(self, agent, capsys):
        """Machine-readable mode (-Q/oneshot) must not contaminate stdout (#26155)."""
        resp = _mock_response(content="Summary")
        agent.client.chat.completions.create.return_value = resp
        agent._cached_system_prompt = "You are helpful."
        agent.suppress_status_output = True

        result = agent._handle_max_iterations(
            [{"role": "user", "content": "do stuff"}],
            1,
        )

        captured = capsys.readouterr()
        assert result == "Summary"
        assert "Reached maximum iterations" not in captured.out

    def test_plain_quiet_mode_still_prints_iteration_warning(self, agent, capsys):
        """Interactive CLI runs quiet_mode=True by default — the warning must
        still show there; only suppress_status_output gates it (#26155)."""
        resp = _mock_response(content="Summary")
        agent.client.chat.completions.create.return_value = resp
        agent._cached_system_prompt = "You are helpful."
        agent.quiet_mode = True
        agent.suppress_status_output = False
        printed = []
        agent._print_fn = lambda *a, **k: printed.append(" ".join(str(x) for x in a))

        result = agent._handle_max_iterations(
            [{"role": "user", "content": "do stuff"}],
            1,
        )

        assert result == "Summary"
        combined = "\n".join(printed) + capsys.readouterr().out
        assert combined.strip()

    def test_api_failure_returns_error(self, agent):
        agent.client.chat.completions.create.side_effect = Exception("API down")
        agent._cached_system_prompt = "You are helpful."
        messages = [{"role": "user", "content": "do stuff"}]
        with patch("agent.relay_llm.complete_logical_call") as complete_logical:
            result = agent._handle_max_iterations(messages, 60)
        assert isinstance(result, str)
        # Plain what-now for the user; the raw exception stays in the log, not the reply.
        assert result.strip()
        assert "API down" not in result
        complete_logical.assert_called_once()
        assert complete_logical.call_args.kwargs == {"outcome": "failed"}

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

    def test_summary_uses_ordinary_tools_and_prompt_cache_key(self, agent):
        """The terminal summary follows the ordinary request's cache lineage."""
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content=""),
            _mock_response(content="Summary"),
        ]
        agent.base_url = "https://api.openai.com/v1"
        agent.provider = "openai"
        agent._cached_system_prompt = "You are helpful."
        messages = [{"role": "user", "content": "do stuff"}]
        ordinary = agent._build_api_kwargs(
            [{"role": "system", "content": agent._cached_system_prompt}, *messages]
        )

        result = agent._handle_max_iterations(messages, 60)

        summaries = [call.kwargs for call in agent.client.chat.completions.create.call_args_list]
        assert result == "Summary"
        assert len(summaries) == 2
        assert all(summary["tools"] == ordinary["tools"] for summary in summaries)
        assert all(summary["prompt_cache_key"] == ordinary["prompt_cache_key"] for summary in summaries)
        assert all(summary.get("tool_choice") == ordinary.get("tool_choice") for summary in summaries)
        assert all(summary["model"] == ordinary["model"] for summary in summaries)
        assert all(summary["messages"][0] == ordinary["messages"][0] for summary in summaries)

    def test_summary_request_scrubs_surrogates_in_tool_schema(self, agent):
        """The summary rides the same outbound surrogate chokepoint as the main loop (#50959 class)."""
        agent.client.chat.completions.create.return_value = _mock_response(content="Summary")
        agent._cached_system_prompt = "You are helpful."
        agent.tools = [{"type": "function", "function": {
            "name": "web_search", "description": "lone surrogate \ud83d here",
            "parameters": {"type": "object", "properties": {}},
        }}]

        result = agent._handle_max_iterations([{"role": "user", "content": "do stuff"}], 60)

        assert result == "Summary"
        sent = agent.client.chat.completions.create.call_args.kwargs
        description = sent["tools"][0]["function"]["description"]
        assert "\ud83d" not in description
        description.encode("utf-8")  # a provider serializes this; lone surrogates raise here

    def test_summary_tool_call_only_response_retries_once(self, agent, caplog):
        """A tool-only summary is never executed: it is logged, reads as empty, and gets one retry."""
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="", tool_calls=[_mock_tool_call()]),
            _mock_response(content="Summary"),
        ]
        agent._cached_system_prompt = "You are helpful."

        with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
            result = agent._handle_max_iterations(
                [{"role": "user", "content": "do stuff"}], 60,
            )

        assert result == "Summary"
        assert agent.client.chat.completions.create.call_count == 2

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
            {"role": "user", "content": "do stuff", "name": "sylvain"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "function": {"name": "execute_code", "arguments": "{}"}}],
                "codex_reasoning_items": [{"id": "rs_1"}],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "result",
                "tool_name": "execute_code",
                "name": "execute_code",
            },
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
            # ``name`` is schema-foreign on tool results only (aki.io rejects
            # it with "contains item with unknown key name"); it stays valid
            # on user/assistant messages.
            if m.get("role") == "tool":
                assert "name" not in m, m
        assert [m for m in sent_msgs if m.get("role") == "user"][0]["name"] == "sylvain"
        # Internal history is untouched — the path copies each message.
        assert messages[2]["tool_name"] == "execute_code"
        assert messages[2]["name"] == "execute_code"
        assert messages[1]["codex_reasoning_items"] == [{"id": "rs_1"}]


    def test_codex_summary_uses_interruptible_request_path(self, agent):
        """Max-iteration Codex summaries must retain request watchdogs.

        A direct ``_run_codex_stream`` call bypasses the absolute stale timeout,
        interrupt handling, and request-local client cleanup. In unattended cron
        sessions that turns a wedged summary stream into a job that never returns
        to cron's completion/error delivery lifecycle (#70943).
        """
        agent.api_mode = "codex_responses"
        agent.provider = "xai-oauth"
        agent.base_url = "https://api.x.ai/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent._base_url_hostname = "api.x.ai"
        agent.model = "grok-4.5"
        agent.platform = "cron"
        agent._cached_system_prompt = "You are helpful."
        response = SimpleNamespace(
            status="completed",
            output=[
                SimpleNamespace(
                    type="message",
                    status="completed",
                    content=[SimpleNamespace(type="output_text", text="Summary")],
                )
            ],
        )

        with patch.object(
            agent, "_interruptible_api_call", return_value=response
        ) as guarded_call, patch.object(
            agent,
            "_run_codex_stream",
            side_effect=AssertionError("summary bypassed request watchdogs"),
        ):
            result = agent._handle_max_iterations(
                [{"role": "user", "content": "do stuff"}], 4
            )

        assert result == "Summary"
        guarded_call.assert_called_once()

    def test_codex_summary_retry_uses_interruptible_request_path(self, agent):
        """The empty-summary retry must use the same bounded request seam."""
        agent.api_mode = "codex_responses"
        agent.provider = "xai-oauth"
        agent.base_url = "https://api.x.ai/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent._base_url_hostname = "api.x.ai"
        agent.model = "grok-4.5"
        agent.platform = "cron"
        agent._cached_system_prompt = "You are helpful."

        def codex_response(text):
            return SimpleNamespace(
                status="completed",
                output=[
                    SimpleNamespace(
                        type="message",
                        status="completed",
                        content=[SimpleNamespace(type="output_text", text=text)],
                    )
                ],
            )

        with patch.object(
            agent,
            "_interruptible_api_call",
            side_effect=[codex_response(""), codex_response("Summary after retry")],
        ) as guarded_call, patch.object(
            agent,
            "_run_codex_stream",
            side_effect=AssertionError("summary retry bypassed request watchdogs"),
        ):
            result = agent._handle_max_iterations(
                [{"role": "user", "content": "do stuff"}], 4
            )

        assert result == "Summary after retry"
        assert guarded_call.call_count == 2

    def test_codex_summary_sanitizes_orphan_tool_results(self, agent):
        agent.api_mode = "codex_responses"
        agent.provider = "openai-codex"
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent._base_url_lower = agent.base_url.lower()
        agent._base_url_hostname = "chatgpt.com"
        agent.model = "gpt-5.5"
        agent._cached_system_prompt = "You are helpful."
        captured = {}

        def fake_run_codex_stream(kwargs, client=None, on_first_delta=None):
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

    def test_codex_summary_strips_tool_controls_on_every_attempt(self, agent):
        """Iteration-limit summaries retry once on an empty answer; both attempts share one
        ``_attempt`` closure, and both must go out without ``tools``, ``tool_choice`` and
        ``parallel_tool_calls`` — the transport emits the three as one block, and strict
        Responses backends 400 on ``tool_choice`` without ``tools``.
        """
        agent.api_mode = "codex_responses"
        agent.provider = "openai-codex"
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent._base_url_lower = agent.base_url.lower()
        agent._base_url_hostname = "chatgpt.com"
        agent.model = "gpt-5.5"
        agent._cached_system_prompt = "You are helpful."
        leaked_controls = {"tools", "tool_choice", "parallel_tool_calls"}
        # Precondition against the real transport: the main-loop request carries all three.
        assert leaked_controls <= agent._build_api_kwargs([{"role": "user", "content": "do stuff"}]).keys()
        bodies = []

        def fake_run_codex_stream(kwargs):
            bodies.append(dict(kwargs))
            text = "" if len(bodies) == 1 else "Summary"
            return SimpleNamespace(
                status="completed",
                output=[
                    SimpleNamespace(
                        type="message",
                        status="completed",
                        content=[SimpleNamespace(type="output_text", text=text)],
                    )
                ],
            )

        with patch.object(agent, "_interruptible_api_call", side_effect=fake_run_codex_stream):
            result = agent._handle_max_iterations([{"role": "user", "content": "do stuff"}], 90)

        assert result == "Summary"
        assert len(bodies) == 2, f"expected one retry after the empty summary, got {len(bodies)} attempts"
        for attempt_index, sent in enumerate(bodies):
            assert not leaked_controls & sent.keys(), f"attempt {attempt_index}: {sorted(leaked_controls & sent.keys())} leaked"

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

    def test_api_sanitizer_matches_responses_id_when_result_keyed_on_id(self, agent):
        """Inverse of the call_id case: a tool_call carries BOTH ``id`` (fc_...)
        and a distinct ``call_id``, but the matching result is keyed on ``id``.
        The sanitizer preferred ``call_id`` only, so it treated the valid
        result as orphaned, dropped it, and injected a bogus
        '[Result unavailable ...]' stub — silently eating a real tool result
        (e.g. mnemosyne_recall / cronjob list). The result must survive intact.
        (#55626)"""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "fc_456",
                        "call_id": "call_456",
                        "type": "function",
                        "function": {"name": "mnemosyne_recall", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "fc_456", "content": '{"results": [1, 2]}'},
        ]

        sanitized = agent._sanitize_api_messages(messages)

        tool_msgs = [m for m in sanitized if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "fc_456"
        assert tool_msgs[0]["content"] == '{"results": [1, 2]}'
        assert "Result unavailable" not in tool_msgs[0]["content"]

    def test_api_sanitizer_still_drops_genuinely_orphaned_result(self, agent):
        """The id-variant matching must not weaken orphan removal: a tool result
        whose tool_call_id matches NO assistant tool_call (neither call_id nor
        id) is still dropped. (#55626 regression guard)"""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "call_nomatch", "content": "orphan"},
        ]

        sanitized = agent._sanitize_api_messages(messages)

        assert all(m.get("role") != "tool" for m in sanitized)

    def test_api_sanitizer_repairs_tool_call_with_empty_function_name(self, agent):
        """A tool_call with id but empty function.name makes the Responses-API
        adapter drop the function_call while keeping its function_call_output,
        causing the gateway's HTTP 400 'No tool call found for function call
        output ...'. The sanitizer renames the blank name to a non-empty
        sentinel so the call and its result stay PAIRED (no orphaned output,
        no 400) while the result content is preserved — it must NOT drop the
        call, because hermes' dispatch loop keeps empty-name calls paired with
        an anti-priming result for self-correction (#47967). (#12807)"""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_good",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    },
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {"name": "", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_good", "content": "ok"},
            {"role": "tool", "tool_call_id": "call_bad", "content": "orphan"},
        ]

        sanitized = agent._sanitize_api_messages(messages)

        # The good call is untouched; the malformed call is repaired in place
        # (renamed to a non-empty sentinel) rather than dropped.
        assistant = next(m for m in sanitized if m.get("role") == "assistant")
        names = [tc["function"]["name"] for tc in assistant["tool_calls"]]
        assert names == ["web_search", "invalid_tool_call"]
        # Both calls now have non-empty names, so neither output is orphaned
        # and both tool results survive — this is what prevents the 400.
        tool_ids = [m.get("tool_call_id") for m in sanitized if m.get("role") == "tool"]
        assert tool_ids == ["call_good", "call_bad"]

class TestRunConversation:
    """Tests for the main run_conversation method.

    Each test mocks client.chat.completions.create to return controlled
    responses, exercising different code paths without real API calls.
    """

    def _setup_agent(self, agent):
        """Common setup for run_conversation tests."""
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False

    def test_task_start_failure_closes_relay_turn_and_lease(self, agent):
        relay_lease = SimpleNamespace(
            parent_session_id="",
            profile_key="/profile",
            session_id=agent.session_id or "",
        )
        relay_turn = object()
        coordinator = MagicMock()
        coordinator.acquire_conversation.return_value = relay_lease
        coordinator.begin_turn.return_value = relay_turn
        start_error = RuntimeError("task metrics start failed")

        with (
            patch("agent.relay_runtime.SESSION_COORDINATOR", coordinator),
            patch(
                "agent.relay_runtime.current_profile_key",
                return_value="/profile",
            ),
            patch(
                "hermes_cli.observability.relay_shared_metrics.start_task_run",
                side_effect=start_error,
            ),
            patch(
                "hermes_cli.observability.relay_shared_metrics.finish_task_run"
            ) as finish_task_run,
            patch("agent.conversation_loop.run_conversation") as run_conversation,
        ):
            with pytest.raises(RuntimeError) as caught:
                agent.run_conversation("hello", task_id="task-1")

        assert caught.value is start_error
        run_conversation.assert_not_called()
        finish_task_run.assert_not_called()
        coordinator.finish_logical_calls.assert_called_once_with(
            relay_turn,
            outcome="failed",
        )
        coordinator.end_turn.assert_called_once_with(
            relay_turn,
            outcome="failed",
        )
        coordinator.release_conversation.assert_called_once_with(relay_lease)
        assert agent._relay_pending_turn_id is None

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

    def test_prompt_cache_marks_static_system_prefix_on_wire(self, agent):
        self._setup_agent(agent)
        agent._cached_system_prompt = "stable instructions\n\nsession context"
        agent._cached_system_prompt_static = "stable instructions"
        agent._use_prompt_caching = True
        agent._use_native_cache_layout = False
        agent._cache_ttl = "5m"
        agent.client.chat.completions.create.return_value = _mock_response(
            content="Final answer",
            finish_reason="stop",
        )

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        system = agent.client.chat.completions.create.call_args.kwargs["messages"][0]
        assert system["role"] == "system"
        assert system["content"] == [
            {
                "type": "text",
                "text": "stable instructions",
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": "\n\nsession context",
                "cache_control": {"type": "ephemeral"},
            },
        ]

    def test_codex_content_filter_incomplete_routes_to_policy_fallback(self, agent):
        self._setup_agent(agent)
        agent.api_mode = "codex_responses"
        agent.provider = "openai-codex"
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent._base_url_lower = agent.base_url.lower()
        agent._base_url_hostname = "chatgpt.com"
        agent.model = "gpt-5.5"
        agent._fallback_chain = [
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.7"},
        ]
        agent._fallback_index = 0

        content_filter_response = SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="content_filter"),
            output=[],
            output_text="",
            model="gpt-5.5",
            usage=None,
        )
        fallback_response = SimpleNamespace(
            status="completed",
            incomplete_details=None,
            output=[
                SimpleNamespace(
                    type="message",
                    status="completed",
                    content=[SimpleNamespace(type="output_text", text="Recovered on fallback")],
                )
            ],
            model="fallback/model",
            usage=None,
        )
        hook_events = []
        logical_completions = []

        def _fake_activate(reason=None):
            agent._fallback_index = len(agent._fallback_chain)
            return True

        with (
            patch.object(agent, "_create_request_openai_client", return_value=MagicMock()),
            patch.object(agent, "_close_request_openai_client"),
            patch.object(agent, "_run_codex_stream", side_effect=[content_filter_response, fallback_response]) as mock_run_codex_stream,
            patch.object(agent, "_try_activate_fallback", side_effect=_fake_activate) as mock_try_activate_fallback,
            patch.object(agent, "_invoke_api_request_error_hook", side_effect=lambda **kw: hook_events.append(kw)),
            patch(
                "agent.relay_llm.complete_logical_call",
                side_effect=lambda request_id, *, outcome: logical_completions.append(
                    (request_id, outcome)
                ),
            ),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("summarize this large Slack thread")

        assert result["final_response"] == "Recovered on fallback"
        assert result["completed"] is True
        mock_try_activate_fallback.assert_called_once_with()
        assert mock_run_codex_stream.call_count == 2
        assert hook_events[0]["error_type"] == "ContentPolicyBlocked"
        assert hook_events[0]["retryable"] is False
        assert hook_events[0]["reason"] == FailoverReason.content_policy_blocked.value
        assert logical_completions == [
            (hook_events[0]["api_request_id"], "success")
        ]

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
        assert "ollama_num_ctx" in result["final_response"]
        assert not agent.client.chat.completions.create.called

    def test_tool_calls_then_stop(self, agent):
        self._setup_agent(agent)
        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
        resp2 = _mock_response(content="Done searching", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [resp1, resp2]
        with (
            patch("model_tools.handle_function_call", return_value="search result") as mock_handle_function_call,
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
            patch("model_tools.handle_function_call", return_value="search result"),
            patch(
                "hermes_cli.lifecycle.has_hook",
                side_effect=lambda name: name in {"pre_api_request", "post_api_request"},
            ),
            patch("hermes_cli.lifecycle.invoke_hook", side_effect=_record_hook),
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
        assert [call["retry_count"] for call in pre_request_calls] == [0, 0]
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

    def test_terminal_task_closes_logical_calls_before_metrics_scope(self, agent):
        from agent import relay_runtime

        order = []
        failed_result = {
            "final_response": "provider failed",
            "messages": [],
            "completed": False,
            "failed": True,
            "interrupted": False,
        }

        with (
            patch(
                "agent.conversation_loop.run_conversation",
                return_value=failed_result,
            ),
            patch(
                "hermes_cli.observability.relay_shared_metrics.start_task_run",
            ),
            patch(
                "hermes_cli.observability.relay_shared_metrics.finish_task_run",
                side_effect=lambda **_kwargs: order.append("metrics"),
            ),
            patch.object(
                relay_runtime.SESSION_COORDINATOR,
                "finish_logical_calls",
                side_effect=lambda *_args, **_kwargs: order.append("logical"),
            ),
        ):
            result = agent.run_conversation("private prompt")

        assert result is failed_result
        assert order == ["logical", "metrics"]


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

        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", _has_hook)
        monkeypatch.setattr(agent, "_api_request_payload_for_hook", _request_payload)
        monkeypatch.setattr(agent, "_api_response_payload_for_hook", _response_payload)

        with (
            patch("hermes_cli.lifecycle.invoke_hook", return_value=[]),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["final_response"] == "No listeners"
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
            patch("model_tools.handle_function_call", return_value="search result"),
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
            patch("agent.interrupt_control._set_interrupt"),
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

    def test_reasoning_only_local_clean_stop_returns_immediately(self, agent):
        """A clean-stop reasoning answer returns without compression or recovery."""
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
        assert result["final_response"] == "reasoning only"
        assert result["api_calls"] == 1


    def test_truly_empty_response_stops_after_repeated_empty(self, agent):
        """Repeated empty responses stop after one retry and return an explanation."""
        self._setup_agent(agent)
        agent.base_url = "http://127.0.0.1:1234/v1"
        empty_resp = _mock_response(content=None, finish_reason="stop")
        # Extra responses prove the guard stops consuming after repetition.
        agent.client.chat.completions.create.side_effect = [
            empty_resp, empty_resp, empty_resp, empty_resp,
        ]
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")
        # Empty after retries keeps the pre-existing status (not a failed turn: cron stays silent,
        # the transcript keeps the text) and only gains the descriptor code for Desktop/TUI.
        assert result["failed"] is False and result["completed"] is True
        assert result["failure_reason"] == "empty_response"
        # #34452: explanation replaces the bare "(empty)" sentinel.
        assert result["final_response"] != "(empty)"
        assert agent.session_api_calls == 2
        assert result["api_calls"] == 2  # 1 original + 1 retry

    def test_deterministic_empty_stops_retries_early(self, agent):
        """NS-503: consecutive zero-output-token empties with identical
        model/provider/finish_reason are deterministic (unsignaled refusal)
        — the loop must stop re-billing the full input after the second
        attempt instead of burning the whole retry budget."""
        self._setup_agent(agent)
        agent.base_url = "http://127.0.0.1:1234/v1"
        zero_usage = {
            "prompt_tokens": 25_900,
            "completion_tokens": 0,
            "total_tokens": 25_900,
        }
        empty_resp = _mock_response(
            content=None, finish_reason="stop", usage=zero_usage
        )
        # Provide plenty of responses; guard should stop consuming early.
        agent.client.chat.completions.create.side_effect = [empty_resp] * 6
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")
        # Empty after retries keeps the pre-existing status (not a failed turn: cron stays silent,
        # the transcript keeps the text) and only gains the descriptor code for Desktop/TUI.
        assert result["failed"] is False and result["completed"] is True
        assert result["failure_reason"] == "empty_response"
        assert result["final_response"] != "(empty)"
        # 1 original + 1 retry: the second identical zero-output empty
        # proves determinism, remaining retries are skipped.
        assert result["api_calls"] == 2

    def test_guard_disabled_via_config_restores_legacy_retries(self, agent):
        """NS-503: agent.empty_response_guard.enabled: false in config.yaml
        (resolved to _empty_guard_enabled at init) restores the legacy
        fixed 3-retry behaviour even for deterministic empties."""
        self._setup_agent(agent)
        agent.base_url = "http://127.0.0.1:1234/v1"
        agent._empty_guard_enabled = False  # as set by agent_init from config
        zero_usage = {
            "prompt_tokens": 25_900,
            "completion_tokens": 0,
            "total_tokens": 25_900,
        }
        empty_resp = _mock_response(
            content=None, finish_reason="stop", usage=zero_usage
        )
        agent.client.chat.completions.create.side_effect = [empty_resp] * 6
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")
        # Empty after retries keeps the pre-existing status (not a failed turn: cron stays silent,
        # the transcript keeps the text) and only gains the descriptor code for Desktop/TUI.
        assert result["failed"] is False and result["completed"] is True
        assert result["failure_reason"] == "empty_response"
        assert result["api_calls"] == 4  # legacy: 1 original + 3 retries


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

    def test_openrouter_empty_retry_bypasses_response_cache(self, agent, monkeypatch):
        """An OpenRouter empty retry must not replay the cached empty response."""
        self._setup_agent(agent)
        empty_resp = _mock_response(content=None, finish_reason="stop")
        content_resp = _mock_response(
            content="Fresh provider response.",
            finish_reason="stop",
        )
        responses = iter([empty_resp, content_resp])
        request_kwargs = []

        def _create(**kwargs):
            request_kwargs.append(kwargs)
            return next(responses)

        original_build_api_kwargs = agent._build_api_kwargs

        def _build_api_kwargs(*args, **kwargs):
            built = original_build_api_kwargs(*args, **kwargs)
            built["extra_headers"] = {"X-Custom-Header": "preserved"}
            return built

        agent.client.chat.completions.create.side_effect = _create
        monkeypatch.setattr(agent, "_build_api_kwargs", _build_api_kwargs)
        monkeypatch.setattr(
            "agent.retry_utils.jittered_backoff",
            lambda *args, **kwargs: 0.0,
        )

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")

        assert result["final_response"] == "Fresh provider response."
        assert "X-OpenRouter-Cache" not in request_kwargs[0].get(
            "extra_headers", {}
        )
        assert request_kwargs[1]["extra_headers"]["X-Custom-Header"] == "preserved"
        assert request_kwargs[1]["extra_headers"]["X-OpenRouter-Cache"] == "false"

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
        # Empty after retries keeps the pre-existing status (not a failed turn: cron stays silent,
        # the transcript keeps the text) and only gains the descriptor code for Desktop/TUI.
        assert result["failed"] is False and result["completed"] is True
        assert result["failure_reason"] == "empty_response"
        # #34452: explanation replaces the bare "(empty)" sentinel.
        assert result["final_response"] != "(empty)"


    def test_empty_response_retry_backoff_interrupted(self, agent, monkeypatch):
        """If an interrupt is requested during the empty response retry wait, we abort."""
        self._setup_agent(agent)
        agent.base_url = "http://127.0.0.1:1234/v1"
        empty_resp = _mock_response(content=None, finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [empty_resp, empty_resp]

        from agent import retry_utils as _retry_utils

        # Make backoff return 10.0 seconds
        monkeypatch.setattr(_retry_utils, "jittered_backoff", lambda *a, **k: 10.0)

        # Trigger the interrupt on the first sleep call inside the wait loop
        original_sleep = time.sleep
        sleep_called = []

        def _mock_sleep(seconds):
            sleep_called.append(seconds)
            if seconds == 0.2:
                agent._interrupt_requested = True
            else:
                original_sleep(seconds)

        monkeypatch.setattr(time, "sleep", _mock_sleep)

        with (
            patch.object(agent, "_persist_session") as mock_persist,
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")

        assert result["interrupted"] is True
        mock_persist.assert_called()

    def test_empty_response_retry_backoff_status(self, agent, monkeypatch):
        """Empty response retry wait updates the agent's status with wait time and sleeps."""
        self._setup_agent(agent)
        agent.base_url = "http://127.0.0.1:1234/v1"

        # Two responses: first empty, second succeeds so it doesn't run forever
        empty_resp = _mock_response(content=None, finish_reason="stop")
        ok_resp = _mock_response(content="Final ok response.", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [empty_resp, ok_resp]

        from agent import conversation_loop as _conv_loop
        from agent import retry_utils as _retry_utils

        monkeypatch.setattr(_retry_utils, "jittered_backoff", lambda *a, **k: 7.5)

        # Fake clock: the retry loop gates on real time.time() < sleep_end, so
        # a no-op sleep alone busy-spins 7.5 wall-clock seconds. Advance a fake
        # clock by each sleep amount instead (established pattern:
        # test_session_activity_persist.py patches run_agent.time.time).
        clock = {"t": time.time()}
        monkeypatch.setattr(_conv_loop.time, "time", lambda: clock["t"])

        sleep_calls = []

        def _fake_sleep(secs):
            sleep_calls.append(secs)
            clock["t"] += secs

        monkeypatch.setattr(time, "sleep", _fake_sleep)
        monkeypatch.setattr(_conv_loop.time, "sleep", _fake_sleep)

        status_messages = []
        monkeypatch.setattr(agent, "_buffer_status", lambda status: status_messages.append(status))

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")

        assert result["completed"] is True
        assert result["final_response"] == "Final ok response."

        assert sum(sleep_calls) >= 7.5


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
        assert result["final_response"].startswith("The answer to your question is that")
        assert result["response_previewed"] is False
        assert result["api_calls"] == 1  # No wasted retries
        # Should emit the stream-interrupted status, NOT the empty-retry status
        # Should NOT have retry statuses


    def test_interrupt_during_stream_preserves_partial_assistant_text(self, agent):
        """Stopping mid-response keeps the streamed reply in history (not 'forgotten')."""
        self._setup_agent(agent)

        def _fake_api_call(api_kwargs):
            # Model streamed some visible text, then the user hit stop.
            agent._current_streamed_assistant_text = "Sure, here's how to do it: first"
            raise InterruptedError("Agent interrupted during streaming API call")

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("how do I do X")

        assert result["interrupted"] is True
        # Partial reply is surfaced and persisted as an assistant turn so the
        # next turn remembers what the model said.
        assert result["final_response"] == "Sure, here's how to do it: first"
        assert result["messages"][-1]["role"] == "assistant"
        assert (
            result["messages"][-1]["content"]
            == "Sure, here's how to do it: first"
        )
        assert isinstance(result["messages"][-1]["timestamp"], float)

    def test_redirect_during_thinking_retries_same_turn_with_context(self, agent):
        """A corrective follow-up does not end the turn, and displayed reasoning
        never re-enters the transcript (classifier-poisoning guard)."""
        self._setup_agent(agent)
        agent.reasoning_callback = lambda _text: None
        final = _mock_response(content="Using Postgres instead.", finish_reason="stop")
        requests = []
        persisted = []

        def _fake_api_call(api_kwargs):
            requests.append(api_kwargs)
            if len(requests) == 1:
                agent._fire_reasoning_delta("I should implement this with SQLite.")
                assert agent.redirect("No, use Postgres instead.") is True
                raise InterruptedError("redirect cancelled the first request")
            return final

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),
            patch.object(
                agent,
                "_persist_session",
                side_effect=lambda messages, *_a, **_k: persisted.append(
                    [dict(message) for message in messages]
                ),
            ),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("Choose a database and implement it.")

        assert result["completed"] is True
        assert result["interrupted"] is False
        assert result["final_response"] == "Using Postgres instead."
        assert len(requests) == 2

        replay = requests[1]["messages"]
        assert [m["role"] for m in replay[-3:]] == [
            "user",
            "assistant",
            "user",
        ]
        # Scaffold rides on the user correction (api_content → content), never
        # as the assistant placeholder's own reply (#81841).
        placeholder = replay[-2]["content"]
        correction = replay[-1]["content"]
        assert "interrupted by a user correction" not in (placeholder or "")
        assert "interrupted by a user correction" in correction
        assert correction.endswith("No, use Postgres instead.")
        # Displayed chain-of-thought must NOT be replayed: an assistant turn
        # inlining its own reasoning trips Anthropic's output classifier and
        # bricks the session with deterministic empty responses (July 2026).
        assert "I should implement this with SQLite." not in correction
        assert "Reasoning shown before the interruption" not in correction
        assert agent._pending_redirect is None
        assert any(
            snapshot[-1].get("content") == "No, use Postgres instead."
            and snapshot[-2].get("role") == "assistant"
            for snapshot in persisted
            if len(snapshot) >= 2
        )

    def test_redirect_wins_race_with_response_completion(self, agent):
        """If the provider returns as redirect lands, discard the stale answer."""
        self._setup_agent(agent)
        stale = _mock_response(content="Using SQLite.", finish_reason="stop")
        corrected = _mock_response(content="Using Postgres.", finish_reason="stop")
        calls = 0

        def _fake_api_call(_api_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                assert agent.redirect("Use Postgres instead.") is True
                return stale
            return corrected

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("Choose a database.")

        assert calls == 2
        assert result["final_response"] == "Using Postgres."
        assert all(
            message.get("content") != "Using SQLite."
            for message in result["messages"]
        )

    def test_redirect_from_input_thread_cancels_live_model_request(self, agent):
        """Exercise the real cross-thread path used by CLI and gateways."""
        self._setup_agent(agent)
        agent.reasoning_callback = lambda _text: None
        entered = threading.Event()
        results = {}
        calls = 0
        final = _mock_response(content="Corrected answer.", finish_reason="stop")

        def _fake_api_call(_api_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                agent._fire_reasoning_delta("Following the original approach.")
                entered.set()
                deadline = time.time() + 2
                while not agent._interrupt_requested and time.time() < deadline:
                    time.sleep(0.01)
                raise InterruptedError("request cancelled by redirect")
            return final

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            worker = threading.Thread(
                target=lambda: results.update(
                    result=agent.run_conversation("Take the original approach.")
                )
            )
            worker.start()
            assert entered.wait(timeout=2)
            assert agent.redirect("Use the corrected approach.") is True
            worker.join(timeout=5)

        assert worker.is_alive() is False
        assert calls == 2
        assert results["result"]["completed"] is True
        assert results["result"]["final_response"] == "Corrected answer."
        placeholder = results["result"]["messages"][-3]
        correction = results["result"]["messages"][-2]
        assert placeholder["role"] == "assistant"
        assert "interrupted by a user correction" not in (
            placeholder.get("content") or ""
        )
        assert "interrupted by a user correction" in (
            correction.get("api_content") or ""
        )
        # Displayed reasoning is display-only — replaying it as assistant
        # content trips Anthropic's output classifier (July 2026 brickings).
        assert "Following the original approach." not in (
            correction.get("api_content") or ""
        )
        assert correction["content"] == "Use the corrected approach."

    def test_legacy_interrupt_scaffold_ghost_dropped_from_api_replay(self, agent):
        """Pre-#81841 hidden assistant rows with the interrupt scaffold must
        not be replayed to the provider — that is what made the model echo
        them into a self-replicating ghost loop."""
        self._setup_agent(agent)
        scaffold = "[This response was interrupted by a user correction.]"
        history = [
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": scaffold,
                "api_content": scaffold,
                "display_kind": "hidden",
            },
            {"role": "user", "content": "real follow-up"},
            {"role": "assistant", "content": "ok"},
        ]
        requests = []

        def _fake_api_call(api_kwargs):
            requests.append(api_kwargs)
            return _mock_response(content="done", finish_reason="stop")

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "next turn", conversation_history=history
            )

        assert result["completed"] is True
        assert len(requests) == 1
        replayed = requests[0]["messages"]
        assert not any(
            isinstance(m.get("content"), str) and m["content"].strip() == scaffold
            for m in replayed
            if m.get("role") == "assistant"
        )
        # Real history around the ghost still reaches the provider.
        # The two consecutive user messages ("first" + "real follow-up")
        # may be merged by repair_message_sequence, so check for the
        # content as a substring rather than exact match.
        assert any(
            m.get("role") == "user"
            and "real follow-up" in str(m.get("content", ""))
            for m in replayed
        )
        assert any(
            m.get("role") == "assistant" and m.get("content") == "ok"
            for m in replayed
        )

    def test_invalid_stored_tool_call_names_are_coerced_on_the_wire(self, agent):
        """A stored ``multi_tool_use.parallel`` / shell-command / empty function.name must reach the
        provider as ``^[A-Za-z0-9_-]{1,64}$`` on every request, and the persisted history must keep
        the original bytes (#51944)."""
        self._setup_agent(agent)
        long_name = 'gbrain query "x" 2>/dev/null | head -40; ' + "y" * 340
        history = [
            {"role": "user", "content": "do two things"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "multi_tool_use.parallel", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": long_name, "arguments": "{}"}},
                {"id": "c3", "type": "function", "function": {"name": "", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "multi_tool_use.parallel", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "name": long_name, "content": "r2"},
            {"role": "tool", "tool_call_id": "c3", "name": "", "content": "r3"},
            {"role": "assistant", "content": "done"},
        ]
        requests = []

        def _fake_api_call(api_kwargs):
            requests.append(api_kwargs)
            return _mock_response(content="ok", finish_reason="stop")

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation("continue", conversation_history=history)

        wire_names = [
            tc["function"]["name"]
            for m in requests[0]["messages"] if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
        ]
        assert len(wire_names) == 3
        assert all(len(n) <= 64 and n.replace("_", "").replace("-", "").isalnum() for n in wire_names)
        assert [tc["function"]["name"] for tc in history[1]["tool_calls"]] == ["multi_tool_use.parallel", long_name, ""]

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
            patch("model_tools.handle_function_call", return_value="result"),
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

    def test_engine_preflight_fires_below_threshold(self, agent):
        """Sub-threshold ContextEngine.should_compress_preflight() routes to compress().

        Regression test for #20316: when running below the threshold_tokens
        cutoff, run_conversation must still consult the engine's
        should_compress_preflight() hook so engines like hermes-lcm can
        perform incremental maintenance (e.g. leaf-chunk compaction)
        without waiting for the 75% context fill threshold.
        """
        self._setup_agent(agent)
        agent.compression_enabled = True

        # Build a conversation history long enough to clear the
        # protect_first_n + protect_last_n + 1 guard so the preflight
        # block actually executes.
        protect_first = agent.context_compressor.protect_first_n
        protect_last = agent.context_compressor.protect_last_n
        prefill = []
        for _i in range((protect_first + protect_last + 4)):
            prefill.append({"role": "user", "content": f"q{_i}"})
            prefill.append({"role": "assistant", "content": f"a{_i}"})

        # Force the preflight estimator far below the threshold so the
        # legacy ``>= threshold_tokens`` branch does NOT fire — only the
        # new engine-driven elif branch should be exercised.
        agent.context_compressor.threshold_tokens = 10**9

        ok_resp = _mock_response(content="Done", finish_reason="stop")
        agent.client.chat.completions.create.return_value = ok_resp

        # Engine-style hook: returns True so the elif branch should
        # invoke _compress_context once for sub-threshold maintenance.
        with (
            patch.object(
                agent.context_compressor,
                "should_compress_preflight",
                return_value=True,
                create=True,
            ) as mock_preflight,
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

        mock_preflight.assert_called_once()
        mock_compress.assert_called_once()
        assert result["final_response"] == "Done"
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
            patch("agent.turn_overflow.time.sleep"),
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

    def test_length_continuation_preserves_large_provider_default_output_cap(self, agent):
        """Continuation retries must not shrink a higher provider default cap — and must
        raise it, since re-sending the same cap just truncates again (#72770)."""
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
        assert requested_caps == [65536, 131072]

    def test_ollama_glm_stop_after_tools_without_terminal_boundary_requests_continuation(self, agent):
        """Local Ollama-hosted GLM (no :cloud suffix) misreports truncated output as stop."""
        self._setup_agent(agent)
        agent.base_url = "http://localhost:11434/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = "glm-4-9b"  # local GLM — no :cloud suffix

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
            patch("model_tools.handle_function_call", return_value="search result"),
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

    @pytest.mark.parametrize("base_url, model", [
        ("https://ollama.com/v1", "glm-5.3-flash"),      # Ollama Cloud host (#72316)
        ("http://localhost:11434/v1", "glm-5.1:cloud"),  # :cloud via local proxy (#98406)
    ])
    def test_ollama_cloud_glm_stop_is_never_rewritten(self, agent, base_url, model):
        """Ollama Cloud reports finish_reason faithfully — an unpunctuated stop stays stop."""
        self._setup_agent(agent)
        agent.base_url = base_url
        agent._base_url_lower = base_url.lower()
        agent.model = model
        unpunctuated = SimpleNamespace(content="Based on the results the best next step is to update the config", tool_calls=None)
        assert agent._should_treat_stop_as_truncated("stop", unpunctuated, [{"role": "tool", "content": "r"}]) is False

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
        # Should have a user-friendly response (not None)
        assert result["final_response"] is not None
        assert "/reasoning" in result["final_response"]


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
            patch("model_tools.handle_function_call") as mock_handle_function_call,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("write the report")

        assert result["completed"] is False
        assert result["partial"] is True
        assert result["failure_reason"] == "truncated"
        mock_handle_function_call.assert_not_called()

    def test_clean_eof_stub_gets_distinct_truncation_message(self, agent):
        """#102766: a clean-EOF partial-stream stub (stream ended with no
        transport exception and no finish_reason) must not print the same
        'stream ended before completion' wording used for a genuine
        network drop — that wording sends the user chasing a network
        problem a stream_diag log with finish_reason_seen=False would
        already have ruled out."""
        from hermes_constants import PARTIAL_STREAM_STUB_ID

        self._setup_agent(agent)
        bad_tc = _mock_tool_call(
            name="write_file",
            arguments='{"path":"report.md","content":"partial',
            call_id="c1",
        )
        resp = _mock_response(content="", finish_reason="length", tool_calls=[bad_tc])
        resp.id = PARTIAL_STREAM_STUB_ID
        resp._clean_eof = True
        agent.client.chat.completions.create.return_value = resp

        printed = []
        agent._print_fn = lambda *a, **k: printed.append(" ".join(str(x) for x in a))

        with (
            patch("model_tools.handle_function_call"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("write the report")

        text = " ".join(printed)
        assert "server ended the stream without ever sending finish_reason" in text
        assert "stream ended before completion" not in text
        # Retries exhausted: the final copy/reason must not blame the network either.
        assert "Check your network" not in result["final_response"]
        assert "kept closing the stream" in result["final_response"]
        assert result["failure_reason"] == "truncated"

    def test_transport_drop_stub_keeps_original_truncation_message(self, agent):
        """Companion to the clean-EOF test above: a stub NOT tagged
        _clean_eof (the shape built after a real transport exception) must
        keep printing the original 'stream ended before completion'
        wording — issue #102766 asks that this case's existing wording
        stay as-is, only the clean-EOF case gets new wording."""
        from hermes_constants import PARTIAL_STREAM_STUB_ID

        self._setup_agent(agent)
        bad_tc = _mock_tool_call(
            name="write_file",
            arguments='{"path":"report.md","content":"partial',
            call_id="c1",
        )
        resp = _mock_response(content="", finish_reason="length", tool_calls=[bad_tc])
        resp.id = PARTIAL_STREAM_STUB_ID
        agent.client.chat.completions.create.return_value = resp

        printed = []
        agent._print_fn = lambda *a, **k: printed.append(" ".join(str(x) for x in a))

        with (
            patch("model_tools.handle_function_call"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation("write the report")

        text = " ".join(printed)
        assert "stream ended before completion" in text

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
            patch("model_tools.handle_function_call", return_value='{"success":true}') as mock_hfc,
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
            patch("model_tools.handle_function_call", return_value='{"success":true}') as mock_hfc,
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

    def test_zero_byte_tool_args_stub_recovers_within_retries(self, agent):
        """#80498: a stream that dies before a single argument byte arrives
        (name-only tool call) produces a stub with tool_calls=None and
        _dropped_tool_names set — the real shape _build_partial_stream_stub
        returns, distinct from the truncated-JSON stub above (which still
        carries a tool_calls list). Confirms the zero-byte trigger is wired
        end-to-end through the retry loop, not just detected at the
        chat_completion_helpers unit level."""
        from hermes_constants import PARTIAL_STREAM_STUB_ID

        self._setup_agent(agent)
        agent.valid_tool_names.add("write_file")

        stall = _mock_response(content="", finish_reason="length", tool_calls=None)
        stall.id = PARTIAL_STREAM_STUB_ID
        stall._dropped_tool_names = ["write_file"]

        good_tc = _mock_tool_call(
            name="write_file",
            arguments='{"path":"report.md","content":"full content"}',
            call_id="c2",
        )
        good_resp = _mock_response(content="", finish_reason="stop", tool_calls=[good_tc])
        final_resp = _mock_response(content="Done!", finish_reason="stop")

        with (
            patch("model_tools.handle_function_call", return_value='{"success":true}') as mock_hfc,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.client.chat.completions.create.side_effect = [
                stall, good_resp, final_resp,
            ]
            result = agent.run_conversation("write the report")

        # The zero-byte stub must trigger a retry, not silently execute
        # write_file with coerced empty arguments (the #80498 regression).
        mock_hfc.assert_called_once()
        assert result["final_response"] == "Done!"


    def test_truncated_tool_json_after_tool_batch_closes_tool_tail(self, agent):
        """finish_reason=tool_calls + truncated args after a real tool must close tool→user."""
        self._setup_agent(agent)
        agent.valid_tool_names.add("write_file")
        good_tc = _mock_tool_call(
            name="write_file",
            arguments='{"path":"ok.md","content":"x"}',
            call_id="c_ok",
        )
        good_resp = _mock_response(
            content="", finish_reason="tool_calls", tool_calls=[good_tc],
        )
        bad_tc = _mock_tool_call(
            name="write_file",
            arguments='{"path":"report.md","content":"partial',
            call_id="c_bad",
        )
        bad_resp = _mock_response(
            content="", finish_reason="tool_calls", tool_calls=[bad_tc],
        )
        agent.client.chat.completions.create.side_effect = [good_resp, bad_resp]

        with (
            patch("model_tools.handle_function_call", return_value='{"success":true}'),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("write then truncate")

        assert result.get("partial") is True
        msgs = result.get("messages") or []
        assert msgs[-1].get("role") == "assistant"
        assert any(isinstance(m, dict) and m.get("role") == "tool" for m in msgs)


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
            patch("model_tools.handle_function_call", return_value="ok"),
            patch("hermes_cli.kanban_db_dispatch._record_task_failure",
                  mock_record_failure),
            patch("hermes_cli.kanban_db_connect.connect", mock_connect),
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
            patch("model_tools.handle_function_call", return_value="ok"),
            patch("hermes_cli.kanban_db_dispatch._record_task_failure",
                  mock_record_failure),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation("do stuff")

        assert mock_record_failure.call_count == 0, (
            "_record_task_failure should not be called outside kanban mode"
        )

    # ── Output-cap retry: safe_out uses provider available_out + request estimate ──

    def test_output_cap_retry_uses_provider_available_out(self, agent):
        """run_conversation retries an output-cap error with max_tokens <=
        available_out - 64, and does NOT halve context_length or trigger
        compression.
        """
        self._setup_agent(agent)
        agent.api_mode = "chat_completions"
        agent.provider = "openrouter"
        agent.model = "some/model"
        agent.max_tokens = 65_536
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.should_compress = MagicMock(return_value=False)

        error_msg = (
            "max_tokens: 65536 > context_window: 200000 "
            "- input_tokens: 199000 = available_tokens: 1000"
        )
        exc = Exception(error_msg)
        exc.status_code = 400
        exc.code = 400

        ok_resp = _mock_response(content="done", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [exc, ok_resp]

        mock_compress = MagicMock(return_value=(
            [{"role": "user", "content": "hello"}],
            "You are helpful.",
        ))
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent.context_compressor, "update_model"),
            patch.object(agent, "_compress_context", mock_compress),
        ):
            result = agent.run_conversation("hello")

        second_call = agent.client.chat.completions.create.call_args_list[1].kwargs
        assert result["completed"] is True
        assert second_call["max_tokens"] <= 936
        assert agent.context_compressor.context_length == 200_000
        mock_compress.assert_called_once()

    def test_output_cap_retry_before_generic_retry_exhaustion(self, agent):
        """Provider max-output-cap 400s clamp via the output-cap handler, not
        the generic retry loop ("failed after 3 retries").
        """
        self._setup_agent(agent)
        agent.api_mode = "chat_completions"
        agent.provider = "deepseek"
        agent.base_url = "https://api.deepseek.com/v1"
        agent.model = "deepseek-v4-flash"
        agent.max_tokens = 98_304
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.should_compress = MagicMock(return_value=False)

        error_msg = (
            "[400]: max_tokens (98304) exceeds model's maximum output tokens "
            "(65536) for model deepseek-v4-flash "
            "(ref: 7735422e-9cb4-4075-a779-dfecb3204a0e)"
        )
        exc = Exception(error_msg)
        exc.status_code = 400
        exc.code = 400

        ok_resp = _mock_response(content="done", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [exc, ok_resp]

        mock_compress = MagicMock(return_value=(
            [{"role": "user", "content": "hello"}],
            "You are helpful.",
        ))
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent.context_compressor, "update_model"),
            patch.object(agent, "_compress_context", mock_compress),
        ):
            result = agent.run_conversation("hello")

        assert len(agent.client.chat.completions.create.call_args_list) == 2
        second_call = agent.client.chat.completions.create.call_args_list[1].kwargs
        assert result["completed"] is True
        assert second_call["max_tokens"] <= 65_472
        assert agent.context_compressor.context_length == 200_000


    def test_wrapped_output_cap_429_not_consumed_by_eager_fallback(self, agent):
        """With a NON-EMPTY fallback chain, the eager rate-limit fallback must
        NOT consume the wrapped output-cap 429 — the failure is a deterministic
        request-shape problem the clamp fixes in one retry; switching provider
        burns a fallback slot for nothing (#72281 ordering guard).
        """
        self._run_wrapped_429_output_cap(
            agent,
            fallback_chain=[{"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}],
        )

    def _run_wrapped_429_output_cap(self, agent, *, fallback_chain):
        self._setup_agent(agent)
        agent.api_mode = "chat_completions"
        agent.provider = "custom"
        agent.base_url = "http://192.168.1.254:20128/v1"
        agent.model = "deepseekv4flash"
        agent.max_tokens = 98_304
        agent.compression_enabled = True
        agent._fallback_chain = fallback_chain
        agent._fallback_index = 0
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.should_compress = MagicMock(return_value=False)

        error_msg = (
            "Error code: 429 - {'error': {'message': \"[400]: max_tokens "
            "(98304) exceeds model's maximum output tokens (65536) for model "
            "deepseek-v4-flash (ref: 37bde60f-44e7-44e2-b995-4af17fba6d6b)\", "
            "'type': 'rate_limit_error', 'code': 'rate_limit_exceeded'}}"
        )
        exc = Exception(error_msg)
        exc.status_code = 429
        exc.code = "rate_limit_exceeded"

        ok_resp = _mock_response(content="done", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [exc, ok_resp]

        mock_compress = MagicMock(return_value=(
            [{"role": "user", "content": "hello"}],
            "You are helpful.",
        ))
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent.context_compressor, "update_model"),
            patch.object(agent, "_compress_context", mock_compress),
        ):
            result = agent.run_conversation("hello")

        assert len(agent.client.chat.completions.create.call_args_list) == 2
        second_call = agent.client.chat.completions.create.call_args_list[1].kwargs
        assert result["completed"] is True
        assert second_call["max_tokens"] <= 65_472
        assert agent.context_compressor.context_length == 200_000
        # The clamp, not provider failover, must have recovered: no fallback
        # slot consumed and the model unchanged.
        assert agent._fallback_index == 0
        assert agent.model == "deepseekv4flash"

    def test_output_cap_retry_with_large_api_only_content(self, agent):
        """When a large system prompt makes api_messages huge while persisted
        messages stay tiny, the retry cap must still respect provider
        available_tokens — not blow up to the full context window.
        """
        self._setup_agent(agent)
        agent.api_mode = "chat_completions"
        agent.provider = "openrouter"
        agent.model = "some/model"
        agent.max_tokens = 65_536
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.should_compress = MagicMock(return_value=False)

        # Huge API-only system prompt; persisted messages are tiny.
        agent._cached_system_prompt = "S" * 796_000

        error_msg = (
            "max_tokens: 65536 > context_window: 200000 "
            "- input_tokens: 199000 = available_tokens: 1000"
        )
        exc = Exception(error_msg)
        exc.status_code = 400
        exc.code = 400

        ok_resp = _mock_response(content="done", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [exc, ok_resp]

        mock_compress = MagicMock(return_value=(
            [{"role": "user", "content": "hello"}],
            "You are helpful.",
        ))
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent.context_compressor, "update_model"),
            patch.object(agent, "_compress_context", mock_compress),
        ):
            result = agent.run_conversation("hello")

        second_call = agent.client.chat.completions.create.call_args_list[1].kwargs
        assert result["completed"] is True
        # The current branch (messages-only estimate) would send max_tokens
        # near 199927 — this test fails on it.
        assert second_call["max_tokens"] <= 936
        assert agent.context_compressor.context_length == 200_000
        mock_compress.assert_called_once()

    def test_output_cap_retry_triggers_compression_and_recovers(self, agent):
        """Regression for the output-cap death-loop (#55546 / #61761).

        When the provider reports an output-cap error on a near-full context
        window, the retry must NOT just shrink max_tokens by a tiny amount and
        spin forever. It must fire _compress_context() to actually free tokens
        so the session recovers instead of exhausting compression_attempts.

        This locks in the fix: previously the output-cap path set
        restart_with_compressed_messages without ever calling the compressor.
        """
        self._setup_agent(agent)
        agent.api_mode = "chat_completions"
        agent.provider = "openrouter"
        agent.model = "some/model"
        agent.max_tokens = 65_536
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        # Context is essentially full -> compressor would want to run.
        agent.context_compressor.should_compress = MagicMock(return_value=True)

        error_msg = (
            "max_tokens: 65536 > context_window: 200000 "
            "- input_tokens: 199000 = available_tokens: 1000"
        )
        exc = Exception(error_msg)
        exc.status_code = 400
        exc.code = 400

        ok_resp = _mock_response(content="done", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [exc, ok_resp]

        # Compress drops the huge history (15 msgs -> 1), freeing tokens.
        mock_compress = MagicMock(return_value=(
            [{"role": "user", "content": "hello"}],
            "You are helpful.",
        ))
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent.context_compressor, "update_model"),
            patch.object(agent, "_compress_context", mock_compress),
        ):
            result = agent.run_conversation("hello")

        # Compression fired exactly once, on the output-cap retry.
        mock_compress.assert_called_once()
        # The compressed messages were re-sent and the call succeeded.
        assert result["completed"] is True
        assert result["final_response"] == "done"
        # The retry honored the reduced max_tokens (available_out - 64).
        second_call = agent.client.chat.completions.create.call_args_list[1].kwargs
        assert second_call["max_tokens"] <= 936
        # LOCK IN THE FIX: the retry must actually SEND the compressed history
        # (the 1-message payload from _compress_context + its new system
        # prompt), not the original multi-message window. Without this, the
        # output-cap retry would call the compressor but re-transmit the same
        # oversized request forever.
        second_messages = second_call.get("messages", [])
        assert second_messages[-1].get("content") == "hello"
        assert len(second_messages) == 2
        assert second_messages[0]["role"] == "system"
        # context_length was NOT mutated by an output-cap error.
        assert agent.context_compressor.context_length == 200_000

    def test_output_cap_retry_compression_no_progress_terminates_bounded(self, agent):
        """Regression: when the compressor cannot reduce the request (zero
        progress AND no images to strip), the output-cap retry must terminate
        via the max-attempts guard instead of spinning forever.

        The compressor is injected to return the input unchanged (same list
        object, no lock-defer — just zero progress), and the provider keeps
        rejecting, so the only correct outcome is a bounded
        ``compression_exhausted`` failure, not an unbounded loop.
        """
        self._setup_agent(agent)
        agent.api_mode = "chat_completions"
        agent.provider = "openrouter"
        agent.model = "some/model"
        agent.max_tokens = 65_536
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.should_compress = MagicMock(return_value=True)

        error_msg = (
            "max_tokens: 65536 > context_window: 200000 "
            "- input_tokens: 199000 = available_tokens: 1000"
        )

        def _rejecting(*args, **kwargs):
            exc = Exception(error_msg)
            exc.status_code = 400
            exc.code = 400
            raise exc

        # The provider never recovers (side effect raises on every call).
        agent.client.chat.completions.create.side_effect = _rejecting

        def _no_progress(messages, system_message, **kwargs):
            # Compressor runs but cannot shrink the request: no-op, same list.
            return messages, system_message

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent.context_compressor, "update_model"),
            patch.object(agent, "_compress_context", side_effect=_no_progress),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is False
        assert result.get("compression_exhausted") is True
        # Terminated in a bounded number of API calls (default max attempts=3
        # => ~4 create calls), NOT an unbounded retry loop.
        assert agent.client.chat.completions.create.call_count <= 6

class TestRetryExhaustion:
    """Regression: retry_count > max_retries was dead code (off-by-one).

    When retries were exhausted the condition never triggered, causing
    the loop to exit and fall through to response.choices[0] on an
    invalid response, raising IndexError.
    """

    def _setup_agent(self, agent):
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
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
        # The conversation loop was extracted out of run_agent.py; the retry
        # paths import jittered_backoff lazily from agent.retry_utils — patch
        # all of them so the retry waits don't burn 18+ seconds of real wall-clock time here.
        from agent import conversation_loop as _conv_loop
        from agent import retry_utils as _retry_utils
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.time", self._make_fast_time_mock()),
            patch.object(_conv_loop, "time", self._make_fast_time_mock()),
            patch.object(_retry_utils, "jittered_backoff", lambda *a, **k: 0.0),
        ):
            result = agent.run_conversation("hello")
        assert result.get("completed") is False, (
            f"Expected completed=False, got: {result}"
        )
        assert result.get("failed") is True
        assert "error" in result
        assert result["failure_reason"] == "invalid_response"
        assert result["failure_retryable"] is True

    def test_invalid_response_retry_completes_one_logical_call(self, agent):
        self._setup_agent(agent)
        agent.client.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[], model="test/model", usage=None),
            _mock_response(content="recovered"),
        ]
        relay_attempts = []
        logical_completions = []

        def execute(request, callback, **kwargs):
            relay_attempts.append(kwargs)
            return callback(request)

        from agent import conversation_loop as _conv_loop
        from agent import retry_utils as _retry_utils

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.time", self._make_fast_time_mock()),
            patch.object(_conv_loop, "time", self._make_fast_time_mock()),
            patch.object(_retry_utils, "jittered_backoff", lambda *a, **k: 0.0),
            patch("agent.relay_llm.execute", side_effect=execute),
            patch(
                "agent.relay_llm.complete_logical_call",
                side_effect=lambda request_id, *, outcome: logical_completions.append(
                    (request_id, outcome)
                ),
            ),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert len(relay_attempts) == 2
        assert all(
            attempt["defer_logical_completion"] is True
            for attempt in relay_attempts
        )
        request_ids = {
            attempt["metadata"]["api_request_id"] for attempt in relay_attempts
        }
        assert len(request_ids) == 1
        assert logical_completions == [(request_ids.pop(), "success")]

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
