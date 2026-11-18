"""HTTP-200 router timeout shim (#68396) in the OpenAI-compatible consumers that bypass
``validate_response``: the stream assembler and the auxiliary ``_validate_llm_response``.

Some routers answer an upstream connect timeout with a 200 ChatCompletion whose only
assistant text is ``Connect timeout, please try again later.`` and zero completion tokens.
The shared ``is_router_timeout_shim`` predicate must keep that text out of every surface.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

SHIM = "Connect timeout, please try again later."


def _chunk(content=None, finish_reason=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
                           model="m", usage=usage)


@pytest.mark.parametrize(
    ("chunks", "expect_streamed", "expect_valid"),
    [
        # Shim split across deltas, zero completion tokens: nothing reaches the live display and
        # the assembled response is rejected so the main loop retries / falls back.
        ([_chunk("Connect timeout, "), _chunk("please try again later."),
          _chunk(finish_reason="stop", usage=SimpleNamespace(completion_tokens=0))], "", False),
        # Control: the same words really generated (positive usage) are released and valid.
        ([_chunk(SHIM), _chunk(finish_reason="stop", usage=SimpleNamespace(completion_tokens=9))], SHIM, True),
        # Control: text that merely starts like the shim is released once it diverges.
        ([_chunk("Connect timeout, "), _chunk("then success: done."), _chunk(finish_reason="stop")],
         "Connect timeout, then success: done.", True),
    ],
)
@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_stream_holds_router_timeout_shim_until_judged(mock_close, mock_create, chunks, expect_streamed, expect_valid):
    from run_agent import AIAgent

    deltas = []
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter(chunks)
    mock_create.return_value = mock_client
    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, stream_delta_callback=deltas.append)
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False

    response = agent._interruptible_streaming_api_call({})

    assert "".join(deltas) == expect_streamed
    assert agent._get_transport().validate_response(response) is expect_valid


def test_auxiliary_validation_rejects_router_timeout_shim():
    """The auxiliary fallback chain treats the shim like a malformed response, not a title."""
    from agent.auxiliary_client import _validate_llm_response

    shim = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=SHIM, tool_calls=None))],
                           usage=SimpleNamespace(completion_tokens=0), model="m")
    with pytest.raises(RuntimeError):
        _validate_llm_response(shim, "title")

    generated = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=SHIM, tool_calls=None))],
                                usage=SimpleNamespace(completion_tokens=9), model="m")
    assert _validate_llm_response(generated, "title") is generated


@patch("agent.process_bootstrap.OpenAI")
def test_iteration_limit_summary_retries_past_router_timeout_shim(_mock_openai):
    """The iteration-limit summary is the fourth consumer: a shim takes the retry slot and the
    real summary from the second call is returned instead of the sentinel text."""
    from run_agent import AIAgent

    def _completion(content, completion_tokens):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None),
                                                        finish_reason="stop")],
                               usage=SimpleNamespace(completion_tokens=completion_tokens), model="m")

    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, enabled_toolsets=[])
    agent.api_mode = "chat_completions"
    agent._cached_system_prompt = "You are helpful."
    agent.client.chat.completions.create.side_effect = [_completion(SHIM, 0), _completion("Real summary.", 3)]

    assert agent._handle_max_iterations([{"role": "user", "content": "do stuff"}], 60) == "Real summary."
    assert agent.client.chat.completions.create.call_count == 2
