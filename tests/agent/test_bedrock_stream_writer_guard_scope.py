"""Regression for #110402: the single-writer stream fence must gate UI delta emission on the
Bedrock Converse streaming path, never Relay's own chunk consumption.

``agent/chat_completion_helpers.py::_BedrockStream._worker`` used to wire the fence check as
Relay's ``accept_chunk`` gate. That gate runs INSIDE ``relay_llm.py``'s raw provider-stream
iteration (``_provider_stream``), so the instant a newer stream attempt claimed the writer slot
mid-response, ``accept_chunk`` returned False and ``_provider_stream`` broke out of its loop —
truncating ``intercepted_events`` to only the chunks seen before that point. The Bedrock
finalizer then rebuilds the persisted ``content``/``bedrock_content_blocks`` from that truncated
buffer, silently dropping the rest of a response the provider delivered IN FULL: no truncation
warning, no interrupt log, normal ``usage.output_tokens`` on the API-call log line, but the
stored assistant message stops after 1-2 words.

The fence's actual job is to stop DUPLICATE delta emission to the user-visible stream sink
across superseded attempts (the same contract ``_writer_still_current`` enforces on the
chat_completions streaming path, which likewise never wires it into Relay's chunk gate) — it
must never cut the buffer the persisted message is built from.
"""

from unittest.mock import MagicMock, patch

from agent.chat_completion_helpers import _BedrockStream


def _agent():
    agent = MagicMock()
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._interrupt_requested = False
    agent._has_stream_consumers.return_value = False
    return agent


def test_worker_never_wires_the_writer_fence_as_relays_chunk_gate():
    """``relay_llm.stream(...)`` inside ``_worker`` must not receive an ``accept_chunk`` kwarg
    derived from the single-writer fence — that gate lives inside Relay's raw provider iteration
    and would truncate the finalizer's event buffer on supersession instead of just muting UI
    delta emission."""
    agent = _agent()
    stream_obj = _BedrockStream(agent, {"__bedrock_region__": "us-east-1"}, on_first_delta=None)

    captured_kwargs = {}

    fake_stream = MagicMock()
    fake_stream.final_response = None

    def _fake_relay_stream(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return fake_stream

    with patch("agent.relay_llm.stream", side_effect=_fake_relay_stream), \
         patch("agent.bedrock_adapter.stream_converse_with_callbacks", return_value=MagicMock(choices=[])), \
         patch.object(stream_obj, "_open_stream", return_value=None):
        stream_obj._worker()

    assert "accept_chunk" not in captured_kwargs, (
        "the Bedrock stream worker must not pass accept_chunk to relay_llm.stream() -- that "
        "gate breaks Relay's raw provider-stream iteration and truncates intercepted_events "
        "on writer supersession, dropping response text the provider actually delivered"
    )
