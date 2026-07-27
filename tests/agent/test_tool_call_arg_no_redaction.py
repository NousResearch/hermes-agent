"""Regression test for #43083.

``build_assistant_message`` must NOT redact tool-call arguments. The dict it
returns enters the in-memory conversation history that is replayed to the model
on every subsequent turn AND is persisted to state.db, which is itself replayed
verbatim on session resume. Masking a credential to ``***`` there poisons the
replay: the model reads back its own ``PGPASSWORD='***' psql ...`` call and
copies the placeholder into the next tool call, breaking every
credential-dependent command on the second turn.
"""

from unittest.mock import MagicMock

from agent.chat_completion_helpers import build_assistant_message


class _FakeToolCall:
    def __init__(self, tc_id, name, arguments):
        self.id = tc_id
        self.type = "function"
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = arguments
        self.extra_content = None

    def __getattr__(self, _name):
        return None


class _FakeAssistantMsg:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls
        self.function_call = None
        self.reasoning_content = None
        self.model_extra = None
        self.reasoning_details = None

    def __getattr__(self, _name):
        return None


class _FakeAgent:
    stream_delta_callback = None
    _stream_callback = None
    reasoning_callback = None
    verbose_logging = False

    def _extract_reasoning(self, _msg):
        return None

    def _strip_think_blocks(self, text):
        return text

    def _needs_thinking_reasoning_pad(self):
        return False

    def _split_responses_tool_id(self, _raw):
        return (None, None)

    def _derive_responses_function_call_id(self, _call_id, _resp_id):
        return None

    def _deterministic_call_id(self, _name, _args, idx):
        return f"det_{idx}"


def _build(arguments):
    tc = _FakeToolCall("call_1", "terminal", arguments)
    msg = build_assistant_message(_FakeAgent(), _FakeAssistantMsg("ok", [tc]), "tool_calls")
    return msg["tool_calls"][0]["function"]["arguments"]


def test_pgpassword_preserved_verbatim(monkeypatch):
    # Force redaction ON to prove build_assistant_message bypasses it for
    # tool-call args regardless of the global toggle.
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True, raising=False)
    args = '{"command": "PGPASSWORD=\'honchorulez\' psql -h 127.0.0.1"}'
    got = _build(args)
    assert got == args
    assert "honchorulez" in got
    assert "***" not in got


def test_bearer_token_preserved_verbatim(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True, raising=False)
    args = '{"command": "curl -H \'Authorization: Bearer sk-abcdef1234567890\'"}'
    got = _build(args)
    assert got == args
    assert "sk-abcdef1234567890" in got
    assert "***" not in got
