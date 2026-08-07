"""Tests for system_prompt_mode in the chat-completions transport legacy path.

Custom (unregistered) providers go through the legacy kwargs path. The
``system_prompt_mode`` param must apply the same compatibility transform as
the provider-profile ``prepare_messages`` hook (#76783).
"""
from __future__ import annotations

from agent.transports.chat_completions import ChatCompletionsTransport


class TestLegacySystemPromptMode:

    def _build(self, system_prompt_mode=None):
        transport = ChatCompletionsTransport()
        return transport.build_kwargs(
            model="gemini-3.1-pro-low",
            messages=[
                {"role": "system", "content": "You are Hermes Agent.\nBig runtime prompt."},
                {"role": "user", "content": "Hello"},
            ],
            tools=None,
            base_url="http://localhost:9876/v1",
            system_prompt_mode=system_prompt_mode,
            is_custom_provider=True,
        )

    def test_default_system_mode_passes_through(self):
        kwargs = self._build()  # None → default behavior
        msgs = kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are Hermes Agent.\nBig runtime prompt."
        assert msgs[1]["content"] == "Hello"

    def test_user_mode_moves_prompt_to_first_user_message(self):
        kwargs = self._build(system_prompt_mode="user")
        msgs = kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert "Big runtime prompt." not in msgs[0]["content"]
        assert "[Hermes runtime instructions]" in msgs[1]["content"]
        assert msgs[1]["content"].endswith("Hello")

    def test_developer_mode_swaps_first_role(self):
        kwargs = self._build(system_prompt_mode="developer")
        msgs = kwargs["messages"]
        assert msgs[0]["role"] == "developer"
        assert msgs[0]["content"] == "You are Hermes Agent.\nBig runtime prompt."

    def test_input_not_mutated(self):
        messages = [
            {"role": "system", "content": "You are Hermes Agent.\nBig runtime prompt."},
            {"role": "user", "content": "Hello"},
        ]
        transport = ChatCompletionsTransport()
        transport.build_kwargs(
            model="gemini-3.1-pro-low",
            messages=messages,
            tools=None,
            base_url="http://localhost:9876/v1",
            system_prompt_mode="user",
            is_custom_provider=True,
        )
        assert messages[0]["content"] == "You are Hermes Agent.\nBig runtime prompt."
        assert messages[1]["content"] == "Hello"
