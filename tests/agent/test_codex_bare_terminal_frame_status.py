"""A Codex terminal frame that names no status reports the kind of the frame itself.

A ``response.failed`` without ``response.status`` is a failed turn: its partial text must never be
accepted as a completed answer. A bare ``response.incomplete`` stays incomplete, so the turn
continues instead of ending on the cut-off text.
"""

from types import SimpleNamespace

import pytest

from agent.codex_responses_adapter import _normalize_codex_response
from agent.codex_runtime import _consume_codex_event_stream

_BARE_FRAMES = {"no response object": None, "response without status": SimpleNamespace(id="resp_1")}


def _stream(event_type, response):
    terminal = SimpleNamespace(type=event_type) if response is None else SimpleNamespace(type=event_type, response=response)
    return [SimpleNamespace(type="response.output_text.delta", delta="The forecast for"), terminal]


@pytest.mark.parametrize("response", _BARE_FRAMES.values(), ids=_BARE_FRAMES.keys())
def test_bare_failed_frame_fails_the_turn(response):
    final = _consume_codex_event_stream(_stream("response.failed", response), model="gpt-test")
    assert final.status == "failed"
    with pytest.raises(RuntimeError, match="status 'failed'"):
        _normalize_codex_response(final)


@pytest.mark.parametrize("response", _BARE_FRAMES.values(), ids=_BARE_FRAMES.keys())
def test_bare_incomplete_frame_leaves_the_turn_incomplete(response):
    final = _consume_codex_event_stream(_stream("response.incomplete", response), model="gpt-test")
    assert final.status == "incomplete"
    _message, finish_reason = _normalize_codex_response(final)
    assert finish_reason == "incomplete"
