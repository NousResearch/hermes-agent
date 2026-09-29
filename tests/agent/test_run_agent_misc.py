"""test_run_agent_misc: split from the former tests/agent/test_run_agent.py monolith (#99982)."""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
import run_agent
from run_agent import AIAgent
from tests.agent._run_agent_helpers import _make_tool_defs, _mock_assistant_msg, _mock_response, _mock_tool_call


def test_is_destructive_command_treats_cp_as_mutating():
    from agent.tool_dispatch_helpers import _is_destructive_command
    assert _is_destructive_command("cp .env.local .env") is True

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
                "model_tools.get_tool_definitions",
                return_value=_make_tool_defs("web_search"),
            ),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
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


    def test_long_key_masked(self, agent):
        key = "sk-or-v1-abcdefghijklmnop"
        result = agent._mask_api_key_for_logs(key)
        assert result.startswith("sk-or-v1")
        assert result.endswith("mnop")
        assert "..." in result

class TestBuildAssistantMessage:
    @staticmethod
    def _enable_native_compaction(agent):
        agent.api_mode = "codex_responses"
        agent.provider = "openai-codex"
        agent.model = "gpt-5.6-sol"
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent._base_url_hostname = "chatgpt.com"
        agent._base_url_lower = agent.base_url
        agent.codex_responses_native_compaction = True
        agent.compression_enabled = True
        agent.runtime_capabilities = {"native_compaction": True}

    def test_basic_message(self, agent):
        msg = _mock_assistant_msg(content="Hello!")
        result = agent._build_assistant_message(msg, "stop")
        assert result["role"] == "assistant"
        assert result["content"] == "Hello!"
        assert result["finish_reason"] == "stop"

    def test_native_checkpoint_arms_real_usage_preflight_deferral(self, agent):
        checkpoint = {
            "type": "compaction",
            "encrypted_content": "opaque-checkpoint",
            "_issuer_kind": "codex_backend",
        }
        msg = _mock_assistant_msg(content="Compacted")
        msg.codex_reasoning_items = [checkpoint]
        agent.context_compressor.note_native_compaction_checkpoint = MagicMock()
        self._enable_native_compaction(agent)

        from agent.usage_anchor import capture_usage_anchor, set_usage_anchor

        history = [{"role": "user", "content": "before compaction"}]
        set_usage_anchor(agent, capture_usage_anchor(255_000, 100, history), turn_base=True)
        result = agent._build_assistant_message(msg, "stop")

        assert result["codex_reasoning_items"] == [checkpoint]
        agent.context_compressor.note_native_compaction_checkpoint.assert_called_once_with()
        assert agent._usage_anchor is None
        assert agent._turn_base_usage_anchor is None

    def test_native_checkpoint_remains_compatible_with_plugin_context_engine(self, agent):
        checkpoint = {
            "type": "compaction",
            "encrypted_content": "opaque-checkpoint",
            "_issuer_kind": "codex_backend",
        }
        msg = _mock_assistant_msg(content="Compacted")
        msg.codex_reasoning_items = [checkpoint]
        agent.context_compressor = SimpleNamespace(threshold_tokens=204_000)
        self._enable_native_compaction(agent)

        result = agent._build_assistant_message(msg, "stop")

        assert result["codex_reasoning_items"] == [checkpoint]

    @pytest.mark.parametrize("encrypted_content", ["", " "])
    def test_malformed_checkpoint_does_not_arm_deferral(
        self, agent, encrypted_content
    ):
        note_checkpoint = MagicMock()
        agent.context_compressor.note_native_compaction_checkpoint = note_checkpoint
        malformed = {
            "type": "compaction",
            "encrypted_content": encrypted_content,
        }
        msg = _mock_assistant_msg(content="Compacted")
        msg.codex_reasoning_items = [malformed]
        self._enable_native_compaction(agent)

        result = agent._build_assistant_message(msg, "stop")

        assert result["codex_reasoning_items"] == [malformed]
        note_checkpoint.assert_not_called()

    def test_ineligible_route_checkpoint_does_not_arm_deferral(self, agent):
        note_checkpoint = MagicMock()
        agent.context_compressor.note_native_compaction_checkpoint = note_checkpoint
        checkpoint = {"type": "compaction", "encrypted_content": "opaque-checkpoint"}
        msg = _mock_assistant_msg(content="Compacted")
        msg.codex_reasoning_items = [checkpoint]

        result = agent._build_assistant_message(msg, "stop")

        assert result["codex_reasoning_items"] == [checkpoint]
        note_checkpoint.assert_not_called()


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
        from agent.process_bootstrap import _SafeWriter
        from io import StringIO
        inner = StringIO()
        writer = _SafeWriter(inner)
        writer.write("hello")
        assert inner.getvalue() == "hello"


    def test_installed_in_run_conversation(self, agent):
        """run_conversation installs _SafeWriter on stdio."""
        import sys
        from agent.process_bootstrap import _SafeWriter
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

def test_aiagent_uses_copilot_acp_client():
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI") as mock_openai,
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
