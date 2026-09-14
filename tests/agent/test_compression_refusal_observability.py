"""Replay a recorded provider refusal through the real SDK and compression path.

The SSE fixture is a captured HTTP-200 response to a benign inventory summary;
only its message identifier was replaced. No request, headers, or credentials
are included. Control variants below deliberately mutate this recorded response.
"""
import copy
from pathlib import Path
from unittest.mock import patch

import anthropic
import httpx
import pytest

from agent.anthropic_adapter import create_anthropic_message
from agent.auxiliary_client import _AnthropicCompletionsAdapter
from agent.context_compressor import ContextCompressor


FIXTURE = Path(__file__).parents[1] / "fixtures/anthropic_compression_refusal.sse"


def client_for(body, calls):
    def respond(request):
        calls.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    return anthropic.Anthropic(api_key="fixture-only", max_retries=0,
                               http_client=httpx.Client(transport=httpx.MockTransport(respond)))


@pytest.mark.parametrize("with_callback", [False, True])
def test_recorded_delta_details_survive_sdk_aggregation(with_callback):
    calls = []
    with client_for(FIXTURE.read_text(), calls) as client:
        message = create_anthropic_message(client, {
            "model": "claude-opus-5", "max_tokens": 100,
            "messages": [{"role": "user", "content": "Summarize inventory."}],
        }, on_stream_event=(lambda event: None) if with_callback else None)
    details = message.stop_details
    assert details is not None
    assert (details if isinstance(details, dict) else details.model_dump())["category"] == "cyber"
    assert len(calls) == 1


def test_recorded_refusal_preserves_context_and_reports_resolved_route(caplog):
    calls = []
    with client_for(FIXTURE.read_text(), calls) as client:
        adapter = _AnthropicCompletionsAdapter(client, "claude-opus-5")
        def call(**kwargs):
            kwargs["route_info"].update(provider="claude-apx-1", model="claude-opus-5")
            return adapter.create(messages=kwargs["messages"], model="claude-opus-5", max_tokens=100)
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            compressor = ContextCompressor(model="main-model",
                                           provider="claude-apr", protect_first_n=2,
                                           protect_last_n=2, abort_on_summary_failure=False)
            compressor.summary_model = "configured-summary"
            messages = [{"role": "user" if i % 2 == 0 else "assistant",
                         "content": "Observatory inventory item " + str(i)} for i in range(12)]
            original = copy.deepcopy(messages)
            with patch("agent.context_compressor.call_llm", side_effect=call):
                result = compressor.compress(messages, current_tokens=999999, force=True)
    assert result == original
    assert messages == original
    assert len(calls) == 1  # Refusals must not enter the empty-response retry path.
    assert compressor._last_compress_aborted
    assert not compressor._last_summary_fallback_used
    assert compressor._last_summary_dropped_count == 0
    error = compressor._last_summary_error
    assert "refus" in error.lower()
    assert "cyber" in error
    assert "claude-apx-1" in error and "claude-opus-5" in error
    assert "claude-apr" not in error
    assert "empty content" not in error
    assert "Usage Policy" not in error  # Never echo provider free text.
    assert "refus" in caplog.text.lower()


@pytest.mark.parametrize("kind", ["empty", "success", "missing_details"])
def test_controls(kind):
    body = FIXTURE.read_text()
    import json
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    if kind == "missing_details":
        events[1]["delta"].pop("stop_details")
    else:
        events[1]["delta"]["stop_reason"] = "end_turn"
        events[1]["delta"].pop("stop_details")
        if kind == "success":
            events[1:1] = [
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "The inventory was checked; continue recording observations."}},
                {"type": "content_block_stop", "index": 0},
            ]
    body = "".join("event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n" for e in events)
    calls = []
    with client_for(body, calls) as client:
        adapter = _AnthropicCompletionsAdapter(client, "claude-opus-5")
        def call(**kwargs):
            kwargs["route_info"].update(provider="resolved-provider", model="claude-opus-5")
            return adapter.create(messages=kwargs["messages"], max_tokens=100)
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            compressor = ContextCompressor(model="claude-opus-5", quiet_mode=True)
            with patch("agent.context_compressor.call_llm", side_effect=call):
                result = compressor._generate_summary([{"role": "user", "content": "Inventory checked."}])
    assert len(calls) == 1
    if kind == "success":
        assert "inventory was checked" in result
        assert compressor._last_summary_error is None
    elif kind == "empty":
        assert result is None
        assert "empty content" in compressor._last_summary_error
        assert "resolved-provider" in compressor._last_summary_error
    else:
        assert result is None
        assert "refus" in compressor._last_summary_error.lower()


@pytest.mark.parametrize("category", ["untrusted-category", {"unexpected": "shape"}, None])
@pytest.mark.parametrize("as_dict", [False, True])
def test_refusal_metadata_is_bounded_and_success_clears_failure(category, as_dict, caplog):
    from types import SimpleNamespace

    response = {
        "choices": [{"finish_reason": "content_filter", "message": {"content": "not a summary"}}],
        "provider_data": {"stop_details": {"category": category, "explanation": "private explanation"}},
    }
    if not as_dict:
        response = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="content_filter", message=SimpleNamespace(content="not a summary"))],
            provider_data=response["provider_data"],
        )
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="main-model", quiet_mode=True)
        compressor.summary_model = "distinct-summary"
        turns = [{"role": "user", "content": "Inventory checked."}]
        with patch("agent.context_compressor.call_llm", return_value=response) as call:
            assert compressor._generate_summary(turns) is None
        assert call.call_count == 1
        assert compressor._last_summary_refusal_failure
        assert "category=unspecified" in compressor._last_summary_error
        assert "untrusted-category" not in caplog.text
        assert "private explanation" not in caplog.text
        success = {"choices": [{"finish_reason": "stop", "message": {"content": "Inventory checked; continue observations."}}]}
        with patch("agent.context_compressor.call_llm", return_value=success):
            assert compressor._generate_summary(turns, bypass_cooldown=True)
        assert not compressor._last_summary_refusal_failure
        assert compressor._last_summary_error is None


def test_stream_cancellation_is_not_refusal_or_create_fallback():
    class Cancelled(BaseException):
        pass
    calls = []
    with client_for(FIXTURE.read_text(), calls) as client:
        def cancel(event):
            raise Cancelled()
        with pytest.raises(Cancelled):
            create_anthropic_message(client, {
                "model": "claude-opus-5", "max_tokens": 100,
                "messages": [{"role": "user", "content": "Inventory."}],
            }, on_stream_event=cancel)
    assert len(calls) == 1
