"""An Anthropic refusal reached through the MoA aggregator keeps its reason (``stop_details``).

The aggregator is called through the OpenAI-shaped aux adapter and handed to the agent as one
stream chunk; each hop used to drop ``stop_details``, so the turn failed as a "safety refusal"
whose only "reason" was the reply the classifier cut short.
"""
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import handle_content_policy_refusal

PARTIAL = "I'm picking up the P4e-2-D grounding and turning to the build plan."
DETAILS = {"type": "refusal", "category": "reasoning_extraction", "explanation": "Declined by a classifier."}


def _refused_aggregator_response():
    from agent.auxiliary_client import _AnthropicCompletionsAdapter

    class _Messages:
        def create(self, **kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=PARTIAL)], stop_reason="refusal",
                stop_details=DETAILS, usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
            )

    adapter = _AnthropicCompletionsAdapter(SimpleNamespace(messages=_Messages()), "claude-fable-5-2")
    return adapter.create(model="claude-fable-5-2", messages=[{"role": "user", "content": "hi"}])


def _refuse(response):
    from agent.transports import get_transport

    agent = MagicMock()
    agent.api_mode, agent.provider, agent.model, agent.log_prefix = "chat_completions", "moa", "default", ""
    agent._get_transport.return_value = get_transport("chat_completions")
    agent._has_pending_fallback.return_value = False
    agent._try_activate_fallback.return_value = False
    agent._extract_reasoning.return_value = ""
    return handle_content_policy_refusal(
        agent, response, TurnRetryState(), thinking_spinner=None, messages=[], api_messages=[],
        api_kwargs={}, active_system_prompt=None, conversation_history=[], api_call_count=1, effective_task_id="t",
        turn_id="u", api_request_id="r", api_start_time=0.0, retry_count=0, max_retries=0,
    )


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_a_refused_aggregator_turn_reports_the_classifier_reason(mock_close, mock_create, caplog):
    """Adapter -> MoA stream chunk -> stream consumer -> refusal handler, as a MoA turn runs them."""
    from agent.moa_loop import _completed_response_as_stream_chunk
    from run_agent import AIAgent

    with caplog.at_level(logging.WARNING, logger="agent.auxiliary_anthropic_response"):
        chunk = _completed_response_as_stream_chunk(_refused_aggregator_response())
    client = MagicMock()
    client.chat.completions.create.return_value = iter([chunk])
    mock_create.return_value = client
    agent = AIAgent(api_key="k", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
                    skip_context_files=True, skip_memory=True)
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False

    error = _refuse(agent._interruptible_streaming_api_call({})).result["error"]

    assert "Declined by a classifier." in error
    assert "reasoning_extraction" in error
    assert PARTIAL not in error
    assert "reasoning_extraction" in caplog.text  # the profile's agent.log names it too


def test_without_stop_details_the_reply_text_is_still_reported():
    message = SimpleNamespace(role="assistant", content=PARTIAL, tool_calls=None, reasoning_content=None, refusal=None)
    response = SimpleNamespace(choices=[SimpleNamespace(index=0, message=message, finish_reason="content_filter")],
                               model="m", usage=None)
    assert PARTIAL in _refuse(response).result["error"]
