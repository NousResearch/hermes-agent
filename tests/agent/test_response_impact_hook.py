"""The optional gateway ``impact`` object (per-call energy / emissions) reaches observers.

Some OpenAI-compatible gateways attach a top-level ``impact`` object to every completion
(non-streaming: response body; streaming: the final usage chunk). The OpenAI SDK keeps it as a
pydantic extra; Hermes passes it through ``post_api_request`` / ``post_auxiliary_call`` as
``impact`` (a size-capped dict, or None) without touching the request or the transcript.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from agent.api_request_hooks import response_impact
from tests.agent.test_first_chunk_at_hook import _make_stream_chunk, _run_with_hooks, agent

IMPACT = {
    "inferenceTime": {"total": 155, "unit": "ms"},
    "energy": {"total": 5468, "unit": "Wms"},
    "emissions": {"total": 106, "unit": "ugCO2e"},
    "version": "20250922",
}


def _completion_body(**extra):
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "glm",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        **extra,
    }


class TestResponseImpact:
    def test_sdk_model_extra_is_read(self):
        # construct() mirrors how the SDK builds responses (no validation, extras kept).
        body = _completion_body(impact=IMPACT)
        assert response_impact(ChatCompletion.model_validate(body)) == IMPACT
        assert response_impact(ChatCompletion.construct(**body)) == IMPACT

    def test_streaming_chunk_extra_is_read(self):
        chunk = ChatCompletionChunk.model_validate({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "glm", "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}, "impact": IMPACT,
        })
        assert response_impact(chunk) == IMPACT

    def test_absent_or_malformed_is_none(self):
        assert response_impact(None) is None
        assert response_impact(ChatCompletion.model_validate(_completion_body())) is None
        assert response_impact(SimpleNamespace(impact="106ug")) is None
        assert response_impact(SimpleNamespace(impact={})) is None
        assert response_impact(MagicMock()) is None  # Mock attrs are never mistaken for data

    def test_size_capped(self):
        big = {f"k{i}": "x" * 5000 for i in range(100)}
        out = response_impact({"impact": big})
        assert out is not None
        assert len(out) <= 33 and all(len(v) < 300 for v in out.values() if isinstance(v, str))


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_streamed_final_usage_chunk_impact_reaches_post_api_request(_close, mock_create, agent):
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    final = SimpleNamespace(choices=[], model="glm", usage=usage, impact=IMPACT)
    chunks = [
        _make_stream_chunk(content="Hello"),
        _make_stream_chunk(content=" world", finish_reason="stop", model="glm"),
        final,
    ]
    client = MagicMock()
    client.chat.completions.create.return_value = iter(chunks)
    mock_create.return_value = client
    agent.stream_delta_callback = lambda _t: None

    result, post = _run_with_hooks(agent)

    assert result["final_response"] == "Hello world"
    assert post[0]["impact"] == IMPACT
    assert post[0]["response"]["impact"] == IMPACT


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_stream_without_impact_reports_none(_close, mock_create, agent):
    chunks = [_make_stream_chunk(content="ok", finish_reason="stop", model="m")]
    client = MagicMock()
    client.chat.completions.create.return_value = iter(chunks)
    mock_create.return_value = client
    agent.stream_delta_callback = lambda _t: None

    _result, post = _run_with_hooks(agent)

    assert post[0]["impact"] is None
