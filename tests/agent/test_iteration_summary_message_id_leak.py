"""Regression test for #103769.

The iteration-summary call hand-builds its wire messages and calls
``chat.completions.create()`` directly, bypassing ``ChatCompletionsTransport
.convert_messages()`` — so ``_SUMMARY_FOREIGN_MESSAGE_KEYS`` must mirror the
transport's strip list. It already stripped ``platform_message_id`` but not
its session-replay alias ``message_id`` (``_rows_to_conversation`` re-exposes
the stored platform id under that top-level key for JSONL transcript compat),
so the alias reached strict Chat Completions providers and 400'd.
"""

from agent.chat_completion_helpers import _iteration_summary_api_messages


class _FakeAgent:
    provider = "openai"
    model = "gpt-4o"
    _cached_system_prompt = ""
    ephemeral_system_prompt = ""
    prefill_messages = []

    def _should_sanitize_tool_calls(self):
        return False

    def _copy_reasoning_content_for_api(self, msg, api_msg):
        pass

    def _sanitize_api_messages(self, msgs):
        return msgs

    def _drop_thinking_only_and_merge_users(self, msgs):
        return msgs


def test_iteration_summary_strips_replayed_message_id():
    agent = _FakeAgent()
    messages = [
        {"role": "user", "content": "hi", "message_id": "1545511361084129351"},
    ]
    api_messages = _iteration_summary_api_messages(agent, messages)
    assert "message_id" not in api_messages[0]
    assert api_messages[0]["content"] == "hi"
    # Original untouched — recovery dedup via has_platform_message_id still
    # needs the alias in persisted history.
    assert messages[0]["message_id"] == "1545511361084129351"
