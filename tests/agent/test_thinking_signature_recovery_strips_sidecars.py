"""The thinking-signature 400 retry must remove EVERY carrier of signed thinking from the replayed request.

``reasoning_details`` is only the flat copy; ``bedrock_content_blocks`` / ``anthropic_content_blocks`` are the
provider-native carriers the adapters consult first, so a retry that strips only the flat copy replays the same
dead signature and fails over for nothing. Canonical ``messages`` (state.db) stay untouched."""
import copy
from types import SimpleNamespace

from agent.error_classifier import ClassifiedError, FailoverReason
from agent.turn_recovery import _recover_format_errors
from agent.turn_retry_state import TurnRetryState


def test_thinking_signature_retry_strips_native_sidecars_from_api_messages_only():
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant", "content": "Hi",
            "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            "reasoning_details": [{"type": "redacted_thinking", "data": "AA=="}],
            "bedrock_content_blocks": [{"reasoningContent": {"text": "plan", "signature": "sig"}}, {"text": "Hi"}],
            "anthropic_content_blocks": [{"type": "thinking", "thinking": "plan", "signature": "sig"}],
        },
    ]
    api_messages = copy.deepcopy(messages)
    agent = SimpleNamespace(log_prefix="", _vprint=lambda *args, **kwargs: None)
    classified = ClassifiedError(reason=FailoverReason.thinking_signature, status_code=400)

    assert _recover_format_errors(agent, RuntimeError("400"), classified, TurnRetryState(), messages, api_messages)

    assistant = api_messages[1]
    assert not {"reasoning_details", "bedrock_content_blocks", "anthropic_content_blocks"} & assistant.keys()
    assert assistant["content"] == "Hi" and assistant["tool_calls"] == messages[1]["tool_calls"]
    assert {"reasoning_details", "bedrock_content_blocks", "anthropic_content_blocks"} <= messages[1].keys()
