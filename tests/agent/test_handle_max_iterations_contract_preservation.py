"""Test strict output contract preservation in handle_max_iterations (#62862)."""

from unittest.mock import MagicMock, patch
import pytest

from agent.chat_completion_helpers import handle_max_iterations


class _MockAgent:
    """Minimal agent mock for handle_max_iterations testing."""

    def __init__(self):
        self.max_iterations = 10
        self._should_sanitize_tool_calls = lambda: False
        self._copy_reasoning_content_for_api = lambda src, dst: None
        self._sanitize_tool_calls_for_strict_api = lambda msg, model: None
        self._sanitize_api_messages = lambda msgs: msgs
        self._drop_thinking_only_and_merge_users = lambda msgs: msgs
        self._cached_system_prompt = ""
        self.ephemeral_system_prompt = None
        self.prefill_messages = []
        self.model = "gpt-4"
        self.provider = "openai"
        self.base_url = "https://api.openai.com/v1"
        self.max_tokens = 4096
        self.api_mode = "chat_completions"
        self._base_url_lower = "https://api.openai.com/v1"
        self.reasoning_config = None
        self._supports_reasoning_extra_body = lambda: False
        self._is_anthropic_oauth = False
        self._anthropic_preserve_dots = lambda: False
        self._get_transport = MagicMock()
        self._get_transport.return_value.normalize_response.return_value.content = "test summary"
        self._ensure_primary_openai_client = MagicMock()
        self.openrouter_min_coding_score = None
        self._is_openrouter_url = lambda: False
        self._build_api_kwargs = lambda msgs: {"model": self.model, "messages": msgs, "tools": []}
        self._run_codex_stream = MagicMock()
        self._anthropic_messages_create = MagicMock()
        self._resolve_lmstudio_summary_reasoning_effort = lambda: None


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = _MockAgent()
    # Mock OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "API response"
    mock_client.chat.completions.create.return_value = mock_response
    agent._ensure_primary_openai_client.return_value = mock_client
    return agent


def test_return_only_contract_preserved(mock_agent):
    """When user message contains 'Return ONLY', summary request preserves contract."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Return ONLY the email address, no extra text."},
    ]

    handle_max_iterations(mock_agent, messages, 10)

    # Check that the appended summary request enforces contract compliance
    appended_request = messages[-1]["content"]
    assert "strictly follows the original output format request" in appended_request
    assert "Do NOT add any preamble" in appended_request
    assert "meta narration" in appended_request


def test_only_return_contract_preserved(mock_agent):
    """When user message contains 'only return', summary request preserves contract."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "only return the table, nothing else"},
    ]

    handle_max_iterations(mock_agent, messages, 10)

    appended_request = messages[-1]["content"]
    assert "strictly follows the original output format request" in appended_request


def test_table_format_contract_preserved(mock_agent):
    """When user requests table format with 'Return ONLY', summary preserves structure."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "Return ONLY this table:\n"
                "Profile | Auth | Live API | Mailbox | Scopes | Verdict\n"
                "If any check fails, add one short \"Failure reason\" line."
            ),
        },
    ]

    handle_max_iterations(mock_agent, messages, 10)

    appended_request = messages[-1]["content"]
    assert "table format" in appended_request
    assert "partial results" in appended_request.lower()
    assert "structured failure" in appended_request.lower()


def test_no_preamble_contract_preserved(mock_agent):
    """When user explicitly requests no preamble, summary respects it."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "No preamble, just the results in CSV format"},
    ]

    handle_max_iterations(mock_agent, messages, 10)

    appended_request = messages[-1]["content"]
    assert "Do NOT add any preamble" in appended_request


def test_no_contract_defaults_to_summary(mock_agent):
    """When no strict contract is detected, use default summary request."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like?"},
    ]

    handle_max_iterations(mock_agent, messages, 10)

    appended_request = messages[-1]["content"]
    assert "summarizing what you've found and accomplished" in appended_request
    assert "strictly follows" not in appended_request


def test_contract_detection_case_insensitive(mock_agent):
    """Contract markers are detected case-insensitively."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "RETURN ONLY the results"},
    ]

    handle_max_iterations(mock_agent, messages, 10)

    appended_request = messages[-1]["content"]
    assert "strictly follows" in appended_request


def test_strict_output_contract_marker(mock_agent):
    """'strict output' marker triggers contract preservation."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Use strict output format: JSON only"},
    ]

    handle_max_iterations(mock_agent, messages, 10)

    appended_request = messages[-1]["content"]
    assert "strictly follows" in appended_request


def test_no_narration_contract_marker(mock_agent):
    """'no narration' marker triggers contract preservation."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Output the data with no narration"},
    ]

    handle_max_iterations(mock_agent, messages, 10)

    appended_request = messages[-1]["content"]
    assert "Do NOT add any preamble" in appended_request
    assert "meta narration" in appended_request


def test_exactly_this_contract_marker(mock_agent):
    """'exactly this' marker triggers contract preservation."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Output exactly this format: YYYY-MM-DD"},
    ]

    handle_max_iterations(mock_agent, messages, 10)

    appended_request = messages[-1]["content"]
    assert "strictly follows" in appended_request


def test_multiple_user_messages_uses_latest(mock_agent):
    """When there are multiple user messages, use the last one for contract detection."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "Response"},
        {"role": "user", "content": "Return ONLY the final answer"},
    ]

    handle_max_iterations(mock_agent, messages, 10)

    appended_request = messages[-1]["content"]
    assert "strictly follows" in appended_request


def test_empty_user_message_defaults_to_summary(mock_agent):
    """When user message is empty, default to regular summary."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": ""},
    ]

    handle_max_iterations(mock_agent, messages, 10)

    appended_request = messages[-1]["content"]
    assert "summarizing what you've found and accomplished" in appended_request


def test_no_extra_text_contract_marker(mock_agent):
    """'no extra text' marker triggers contract preservation."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Return the data with no extra text"},
    ]

    handle_max_iterations(mock_agent, messages, 10)

    appended_request = messages[-1]["content"]
    assert "strictly follows" in appended_request