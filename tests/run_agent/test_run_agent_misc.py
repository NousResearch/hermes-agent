"""Assorted run_agent.AIAgent unit tests that do not fit a larger theme.

Split out of the former monolithic ``test_run_agent.py`` (8,699 lines) so the
per-file CI shard runner (``scripts/run_tests_parallel.py`` spawns one
subprocess per file and cannot split within a file) is not pinned by a single
file. Pure move: test bodies are byte-identical to the original. Shared
fixtures live in ``conftest.py``; shared mock builders in
``_run_agent_helpers.py``.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import run_agent
from run_agent import AIAgent

from tests.run_agent._run_agent_helpers import (
    _make_tool_defs,
    _mock_assistant_msg,
    _mock_tool_call,
    _mock_response,
)


def test_is_destructive_command_treats_cp_as_mutating():
    assert run_agent._is_destructive_command("cp .env.local .env") is True

def test_is_destructive_command_treats_install_as_mutating():
    assert run_agent._is_destructive_command("install template.env .env") is True

def test_aiagent_reuses_existing_errors_log_handler():
    """Repeated AIAgent init should not accumulate duplicate errors.log handlers."""
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    error_log_path = (run_agent._hermes_home / "logs" / "errors.log").resolve()

    try:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        preexisting_handler = RotatingFileHandler(
            error_log_path,
            maxBytes=2 * 1024 * 1024,
            backupCount=2,
        )
        root_logger.addHandler(preexisting_handler)

        with (
            patch(
                "run_agent.get_tool_definitions",
                return_value=_make_tool_defs("web_search"),
            ),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            AIAgent(
                api_key="test-k...7890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            AIAgent(
                api_key="test-k...7890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        matching_handlers = [
            handler for handler in root_logger.handlers
            if isinstance(handler, RotatingFileHandler)
            and error_log_path == Path(handler.baseFilename).resolve()
        ]
        assert len(matching_handlers) == 1
    finally:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            if handler not in original_handlers:
                handler.close()
        for handler in original_handlers:
            root_logger.addHandler(handler)

class TestMaskApiKey:
    def test_none_returns_none(self, agent):
        assert agent._mask_api_key_for_logs(None) is None

    def test_short_key_returns_stars(self, agent):
        assert agent._mask_api_key_for_logs("short") == "***"

    def test_long_key_masked(self, agent):
        key = "sk-or-v1-abcdefghijklmnop"
        result = agent._mask_api_key_for_logs(key)
        assert result.startswith("sk-or-v1")
        assert result.endswith("mnop")
        assert "..." in result

class TestBuildAssistantMessage:
    def test_basic_message(self, agent):
        msg = _mock_assistant_msg(content="Hello!")
        result = agent._build_assistant_message(msg, "stop")
        assert result["role"] == "assistant"
        assert result["content"] == "Hello!"
        assert result["finish_reason"] == "stop"

    def test_with_reasoning(self, agent):
        msg = _mock_assistant_msg(content="answer", reasoning="thinking")
        result = agent._build_assistant_message(msg, "stop")
        assert result["reasoning"] == "thinking"

    def test_reasoning_content_preserved_separately(self, agent):
        msg = _mock_assistant_msg(
            content="answer",
            reasoning="summary",
            reasoning_content="provider scratchpad",
        )
        result = agent._build_assistant_message(msg, "stop")
        assert result["reasoning_content"] == "provider scratchpad"

    def test_with_tool_calls(self, agent):
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        msg = _mock_assistant_msg(content="", tool_calls=[tc])
        result = agent._build_assistant_message(msg, "tool_calls")
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "web_search"

    def test_with_reasoning_details(self, agent):
        details = [{"type": "reasoning.summary", "text": "step1", "signature": "sig1"}]
        msg = _mock_assistant_msg(content="ans", reasoning_details=details)
        result = agent._build_assistant_message(msg, "stop")
        assert "reasoning_details" in result
        assert result["reasoning_details"][0]["text"] == "step1"

    def test_empty_content(self, agent):
        msg = _mock_assistant_msg(content=None)
        result = agent._build_assistant_message(msg, "stop")
        assert result["content"] == ""

    def test_streaming_only_reasoning_promoted_to_reasoning_content(self, agent):
        """Refs #16844 / #16884. Streaming-only providers (glm, MiniMax,
        gpt-5.x via aigw, Anthropic via openai-compat shims) accumulate
        reasoning through delta chunks but never expose
        ``reasoning_content`` as a top-level attribute on the finalized
        message — only ``reasoning`` (or the internal accumulator).

        Without write-side promotion, the persisted message stores the
        chain-of-thought under the internal ``reasoning`` key and omits
        ``reasoning_content``. When the user later replays that history
        through a DeepSeek-v4 / Kimi thinking model, the missing field
        causes HTTP 400 ("The reasoning_content in the thinking mode
        must be passed back to the API.").

        Fix: when ``reasoning_content`` wasn't written by an earlier
        branch AND we captured reasoning text from streaming deltas,
        promote it to ``reasoning_content`` at write time.
        """
        # SDK-style object that exposes ``reasoning`` but NOT
        # ``reasoning_content`` — the streaming-only provider shape.
        msg = _mock_assistant_msg(content="answer", reasoning="hidden thinking")
        assert not hasattr(msg, "reasoning_content")

        result = agent._build_assistant_message(msg, "stop")

        assert result["reasoning"] == "hidden thinking"
        assert result["reasoning_content"] == "hidden thinking"

    def test_sdk_reasoning_content_still_wins_over_fallback(self, agent):
        """Additive fallback must not override SDK-supplied reasoning_content.

        When both ``reasoning`` and ``reasoning_content`` are present, the
        SDK's own ``reasoning_content`` is authoritative (may carry
        structured data the accumulator doesn't have).
        """
        msg = _mock_assistant_msg(
            content="answer",
            reasoning="summary only",
            reasoning_content="structured provider scratchpad",
        )
        result = agent._build_assistant_message(msg, "stop")
        assert result["reasoning_content"] == "structured provider scratchpad"

    def test_no_reasoning_text_leaves_field_absent(self, agent):
        """Non-thinking turns with no reasoning leave reasoning_content absent.

        This preserves ``_copy_reasoning_content_for_api``'s downstream
        tiers at replay time — cross-provider leak guard (#15748),
        promote-from-``reasoning``, and DeepSeek/Kimi " "-pad — which
        would all be bypassed if we eagerly wrote ``reasoning_content=" "``
        on every assistant turn regardless of provider.
        """
        msg = _mock_assistant_msg(content="plain answer")
        result = agent._build_assistant_message(msg, "stop")
        assert "reasoning_content" not in result

    def test_tool_call_extra_content_preserved(self, agent):
        """Gemini thinking models attach extra_content with thought_signature
        to tool calls. This must be preserved so subsequent API calls include it."""
        tc = _mock_tool_call(
            name="get_weather", arguments='{"city":"NYC"}', call_id="c2"
        )
        tc.extra_content = {"google": {"thought_signature": "abc123"}}
        msg = _mock_assistant_msg(content="", tool_calls=[tc])
        result = agent._build_assistant_message(msg, "tool_calls")
        assert result["tool_calls"][0]["extra_content"] == {
            "google": {"thought_signature": "abc123"}
        }

    def test_tool_call_without_extra_content(self, agent):
        """Standard tool calls (no thinking model) should not have extra_content."""
        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c3")
        msg = _mock_assistant_msg(content="", tool_calls=[tc])
        result = agent._build_assistant_message(msg, "tool_calls")
        assert "extra_content" not in result["tool_calls"][0]

    def test_think_blocks_stripped_from_content(self, agent):
        """Inline <think> blocks are stripped from stored content (#8878, #9568).

        The reasoning is captured into ``msg['reasoning']`` via the inline
        fallback in ``_extract_reasoning``; the raw tags in ``content`` are
        redundant and leak to messaging platforms / pollute titles /
        inflate context if left in place.
        """
        msg = _mock_assistant_msg(
            content="<think>internal reasoning</think>The actual answer."
        )
        result = agent._build_assistant_message(msg, "stop")
        assert "<think>" not in result["content"]
        assert "internal reasoning" not in result["content"]
        assert "The actual answer." in result["content"]
        # Reasoning preserved separately via inline extraction fallback
        assert result["reasoning"] == "internal reasoning"

    def test_think_blocks_stripped_preserves_normal_content(self, agent):
        """Content without reasoning tags passes through unchanged."""
        msg = _mock_assistant_msg(content="No thinking here.")
        result = agent._build_assistant_message(msg, "stop")
        assert result["content"] == "No thinking here."

    def test_memory_context_in_stored_content_is_preserved(self, agent):
        """`_build_assistant_message` must not silently mutate model output
        containing literal <memory-context> markers — that's legitimate text
        (e.g. documentation, code) that the model may emit.  Streaming-path
        leak prevention is handled by StreamingContextScrubber upstream."""
        original = (
            "<memory-context>\n"
            "[System note: The following is recalled memory context, NOT new user input. Treat as informational background data.]\n\n"
            "## Honcho Context\n"
            "stale memory\n"
            "</memory-context>\n\n"
            "Visible answer"
        )
        msg = _mock_assistant_msg(content=original)
        result = agent._build_assistant_message(msg, "stop")
        assert "<memory-context>" in result["content"]
        assert "Visible answer" in result["content"]

    def test_unterminated_think_block_stripped(self, agent):
        """Unterminated <think> block (MiniMax / NIM dropped close tag) is
        fully stripped from stored content."""
        msg = _mock_assistant_msg(
            content="<think>reasoning that never closes on this NIM endpoint"
        )
        result = agent._build_assistant_message(msg, "stop")
        assert "<think>" not in result["content"]
        assert "reasoning that never closes" not in result["content"]
        assert result["content"] == ""

class TestHookPayloadSanitizesSimpleNamespace:
    """Regression: ``_hook_jsonable`` referenced ``SimpleNamespace`` without
    importing it, so sanitizing any hook payload that contained one raised
    ``NameError: name 'SimpleNamespace' is not defined``.

    The non-OpenAI providers (Bedrock, Codex responses, the auxiliary client,
    and the chat-completion stream stub) build their response / message /
    tool_call objects as ``types.SimpleNamespace`` — see
    ``agent/bedrock_adapter.py``, ``agent/codex_responses_adapter.py``, and
    ``agent/auxiliary_client.py``. Those raw objects are handed straight to
    ``_api_response_payload_for_hook`` for the ``post_api_request`` hook, so the
    crash silently killed observability hooks for every one of those providers
    (the call sites swallow the exception with ``except Exception: pass``).
    """

    def test_hook_jsonable_normalizes_simplenamespace(self):
        ns = SimpleNamespace(id="call_1", value=42, nested=SimpleNamespace(name="x"))
        result = AIAgent._sanitize_hook_payload(ns)
        assert result == {"id": "call_1", "value": 42, "nested": {"name": "x"}}

    def test_api_response_payload_for_hook_normalizes_simplenamespace_tool_calls(self, agent):
        # Shape mirrors agent/bedrock_adapter.py::normalize_converse_response and
        # agent/codex_responses_adapter.py — raw SDK objects are SimpleNamespace.
        tool_call = SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(name="web_search", arguments='{"q": "hi"}'),
        )
        assistant_message = SimpleNamespace(
            role="assistant",
            content="",
            tool_calls=[tool_call],
        )
        response = SimpleNamespace(model="anthropic.claude-3", usage=None)

        payload = agent._api_response_payload_for_hook(
            response, assistant_message, finish_reason="tool_calls"
        )

        assert payload["model"] == "anthropic.claude-3"
        assert payload["finish_reason"] == "tool_calls"
        normalized_call = payload["assistant_message"]["tool_calls"][0]
        assert normalized_call["id"] == "call_1"
        assert normalized_call["function"]["name"] == "web_search"

class TestSafeWriter:
    """Verify _SafeWriter guards stdout against OSError (broken pipes)."""

    def test_write_delegates_normally(self):
        """When stdout is healthy, _SafeWriter is transparent."""
        from run_agent import _SafeWriter
        from io import StringIO
        inner = StringIO()
        writer = _SafeWriter(inner)
        writer.write("hello")
        assert inner.getvalue() == "hello"

    def test_write_catches_oserror(self):
        """OSError on write is silently caught, returns len(data)."""
        from run_agent import _SafeWriter
        from unittest.mock import MagicMock
        inner = MagicMock()
        inner.write.side_effect = OSError(5, "Input/output error")
        writer = _SafeWriter(inner)
        result = writer.write("hello")
        assert result == 5  # len("hello")

    def test_flush_catches_oserror(self):
        """OSError on flush is silently caught."""
        from run_agent import _SafeWriter
        from unittest.mock import MagicMock
        inner = MagicMock()
        inner.flush.side_effect = OSError(5, "Input/output error")
        writer = _SafeWriter(inner)
        writer.flush()  # should not raise

    def test_print_survives_broken_stdout(self, monkeypatch):
        """print() through _SafeWriter doesn't crash on broken pipe."""
        import sys
        from run_agent import _SafeWriter
        from unittest.mock import MagicMock
        broken = MagicMock()
        broken.write.side_effect = OSError(5, "Input/output error")
        original = sys.stdout
        sys.stdout = _SafeWriter(broken)
        try:
            print("this should not crash")  # would raise without _SafeWriter
        finally:
            sys.stdout = original

    def test_installed_in_run_conversation(self, agent):
        """run_conversation installs _SafeWriter on stdio."""
        import sys
        from run_agent import _SafeWriter
        resp = _mock_response(content="Done", finish_reason="stop")
        agent.client.chat.completions.create.return_value = resp
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            with (
                patch.object(agent, "_persist_session"),
                patch.object(agent, "_save_trajectory"),
                patch.object(agent, "_cleanup_task_resources"),
            ):
                agent.run_conversation("test")
            assert isinstance(sys.stdout, _SafeWriter)
            assert isinstance(sys.stderr, _SafeWriter)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    # test_installed_before_init_time_honcho_error_prints removed —
    # Honcho integration extracted to plugin (PR #4154).

    def test_double_wrap_prevented(self):
        """Wrapping an already-wrapped stream doesn't add layers."""
        from run_agent import _SafeWriter
        from io import StringIO
        inner = StringIO()
        wrapped = _SafeWriter(inner)
        # isinstance check should prevent double-wrapping
        assert isinstance(wrapped, _SafeWriter)
        # The guard in run_conversation checks isinstance before wrapping
        if not isinstance(wrapped, _SafeWriter):
            wrapped = _SafeWriter(wrapped)
        # Still just one layer
        wrapped.write("test")
        assert inner.getvalue() == "test"

def test_aiagent_uses_copilot_acp_client():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI") as mock_openai,
        patch("agent.copilot_acp_client.CopilotACPClient") as mock_acp_client,
    ):
        acp_client = MagicMock()
        mock_acp_client.return_value = acp_client

        agent = AIAgent(
            api_key="copilot-acp",
            base_url="acp://copilot",
            provider="copilot-acp",
            acp_command="/usr/local/bin/copilot",
            acp_args=["--acp", "--stdio"],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.client is acp_client
    mock_openai.assert_not_called()
    mock_acp_client.assert_called_once()
    assert mock_acp_client.call_args.kwargs["base_url"] == "acp://copilot"
    assert mock_acp_client.call_args.kwargs["api_key"] == "copilot-acp"
    assert mock_acp_client.call_args.kwargs["command"] == "/usr/local/bin/copilot"
    assert mock_acp_client.call_args.kwargs["args"] == ["--acp", "--stdio"]

def test_quiet_spinner_allowed_with_explicit_print_fn(agent):
    agent._print_fn = lambda *_a, **_kw: None
    with patch.object(run_agent.sys.stdout, "isatty", return_value=False):
        assert agent._should_start_quiet_spinner() is True

def test_quiet_spinner_allowed_on_real_tty(agent):
    agent._print_fn = None
    with patch.object(run_agent.sys.stdout, "isatty", return_value=True):
        assert agent._should_start_quiet_spinner() is True

def test_quiet_spinner_suppressed_on_non_tty_without_print_fn(agent):
    agent._print_fn = None
    with patch.object(run_agent.sys.stdout, "isatty", return_value=False):
        assert agent._should_start_quiet_spinner() is False

def test_is_openai_client_closed_honors_custom_client_flag():
    assert AIAgent._is_openai_client_closed(SimpleNamespace(is_closed=True)) is True
    assert AIAgent._is_openai_client_closed(SimpleNamespace(is_closed=False)) is False

def test_is_openai_client_closed_handles_method_form():
    """Fix for issue #4377: is_closed as method (openai SDK) vs property (httpx).

    The openai SDK's is_closed is a method, not a property. Prior to this fix,
    getattr(client, "is_closed", False) returned the bound method object, which
    is always truthy, causing the function to incorrectly report all clients as
    closed and triggering unnecessary client recreation on every API call.
    """

    class MethodFormClient:
        """Mimics openai.OpenAI where is_closed() is a method."""

        def __init__(self, closed: bool):
            self._closed = closed

        def is_closed(self) -> bool:
            return self._closed

    # Method returning False - client is open
    open_client = MethodFormClient(closed=False)
    assert AIAgent._is_openai_client_closed(open_client) is False

    # Method returning True - client is closed
    closed_client = MethodFormClient(closed=True)
    assert AIAgent._is_openai_client_closed(closed_client) is True

def test_is_openai_client_closed_falls_back_to_http_client():
    """Verify fallback to _client.is_closed when top-level is_closed is None."""

    class ClientWithHttpClient:
        is_closed = None  # No top-level is_closed

        def __init__(self, http_closed: bool):
            self._client = SimpleNamespace(is_closed=http_closed)

    assert AIAgent._is_openai_client_closed(ClientWithHttpClient(http_closed=False)) is False
    assert AIAgent._is_openai_client_closed(ClientWithHttpClient(http_closed=True)) is True
