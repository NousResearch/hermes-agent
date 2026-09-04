"""Offline Grok Bot transport tests. No network."""

from __future__ import annotations

from types import SimpleNamespace

from agent.error_classifier import classify_api_error
from agent.grokbot.client import openai_messages_to_history, ROLE_ASSISTANT, ROLE_USER
from agent.transports import get_transport
from agent.transports.grokbot import grokbot_runtime_active
from hermes_cli.config import _canonical_api_mode
from hermes_cli.runtime_provider import _parse_api_mode, _VALID_API_MODES


def test_grokbot_is_a_valid_api_mode():
    assert "grokbot" in _VALID_API_MODES
    assert _parse_api_mode("grokbot") == "grokbot"
    assert _canonical_api_mode("grok-bot") == "grokbot"
    assert _canonical_api_mode("connect_inference") == "grokbot"


def test_get_transport_grokbot():
    t = get_transport("grokbot")
    assert t is not None
    assert t.api_mode == "grokbot"


def test_runtime_active_by_api_mode_and_provider():
    assert grokbot_runtime_active(SimpleNamespace(api_mode="grokbot", provider="x", base_url=""))
    assert grokbot_runtime_active(SimpleNamespace(api_mode="", provider="grokbot", base_url=""))
    assert grokbot_runtime_active(SimpleNamespace(api_mode="", provider="grok-bot", base_url=""))
    assert not grokbot_runtime_active(SimpleNamespace(api_mode="chat_completions", provider="openai", base_url=""))


def test_runtime_active_exact_hostname_only():
    assert grokbot_runtime_active(
        SimpleNamespace(api_mode="", provider="custom", base_url="https://api2.cursor.sh")
    )
    assert grokbot_runtime_active(
        SimpleNamespace(api_mode="", provider="custom", base_url="https://api2.cursor.sh/v1")
    )
    assert not grokbot_runtime_active(
        SimpleNamespace(
            api_mode="",
            provider="custom",
            base_url="https://evil-api2.cursor.sh.example",
        )
    )
    assert not grokbot_runtime_active(
        SimpleNamespace(api_mode="", provider="custom", base_url="https://api.cursor.com")
    )


def test_openai_history_does_not_stuff_called_tools():
    prompt, history = openai_messages_to_history(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "write_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "name": "write_file", "content": "ok"},
        ]
    )
    assert "Called tools:" not in prompt
    assert "Called tools:" not in "".join(t for _, t in history)
    assert any(role == ROLE_USER and t.startswith("[SYSTEM]") for role, t in history)
    assert prompt.startswith("TOOL_RESULT write_file:")
    assert ROLE_ASSISTANT not in {r for r, t in history if not t}


def test_cursor_chat_completions_404_is_not_retried():
    class Err(Exception):
        def __init__(self):
            super().__init__("Error code: 404")
            self.status_code = 404
            self.body = {"message": "Route POST:/chat/completions not found", "error": "Not Found"}
            self.response = SimpleNamespace(headers={})

    classified = classify_api_error(Err(), provider="grokbot", model="grok-4.6")
    assert classified.retryable is False
