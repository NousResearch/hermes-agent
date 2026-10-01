"""#99692 sibling wires — the host compression deadline must stop EVERY aux
stream consumer, not only the chat.completions accumulator.

``aux_stream_deadline`` (salvaged from PR #99779 by @JoaoMarcos44) publishes
the ``CompressionCommitFence`` ceiling to the streamed chat.completions path.
Two other auxiliary wires consume their streams internally and were left with
their own, always-larger budgets:

* the Codex Responses adapter (``_CodexCompletionsAdapter.create``) — its
  re-armable watchdog only knew ``_aux_stream_total_ceiling`` (>= 600s);
* the Anthropic Messages adapter — its ``on_stream_event`` hook only ticked
  progress and never stopped the stream at all (nor honoured a hard cancel).

Both now stop at the host's absolute deadline, so an abandoned summary is not
billed to completion on a socket nobody is waiting for.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import auxiliary_client as aux
from agent.anthropic_adapter import create_anthropic_message


# ── Codex Responses wire ─────────────────────────────────────────────────


def _codex_content_event(text="tok"):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def _consume_codex(stream, *, model, on_event):
    del model
    for event in stream:
        on_event(event)
    return SimpleNamespace(
        output=[SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text="summary")],
        )],
        usage=None,
    )


def _make_codex_adapter(event_iter):
    real_client = SimpleNamespace(
        base_url="https://chatgpt.com/backend-api/codex",
        responses=SimpleNamespace(create=lambda **_kwargs: event_iter),
        close=lambda: None,
    )
    return aux._CodexCompletionsAdapter(real_client, "gpt-5.6-sol")


def test_codex_stream_stops_at_the_host_deadline_not_its_own_ceiling():
    """A live (re-arming) Codex stream must die at the host's deadline even
    though its own hard ceiling is >= 600s and every token re-arms the
    no-progress window."""
    yielded = [0]

    def _live_forever():
        while True:
            time.sleep(0.02)
            yielded[0] += 1
            yield _codex_content_event()

    adapter = _make_codex_adapter(_live_forever())
    start = time.monotonic()
    with (
        patch("agent.codex_runtime._consume_codex_event_stream", _consume_codex),
        aux.aux_stream_deadline(time.monotonic() + 0.4),
        pytest.raises(TimeoutError, match="hard ceiling"),
    ):
        adapter.create(
            messages=[{"role": "user", "content": "summarize"}],
            timeout=300,
        )
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"stream outlived the host deadline by {elapsed:.1f}s"
    assert yielded[0] < 100


def test_codex_stream_without_host_deadline_keeps_its_ceiling():
    def _short():
        for _ in range(3):
            yield _codex_content_event()

    adapter = _make_codex_adapter(_short())
    with patch("agent.codex_runtime._consume_codex_event_stream", _consume_codex):
        response = adapter.create(
            messages=[{"role": "user", "content": "summarize"}], timeout=300,
        )
    assert response.choices[0].message.content == "summary"


# ── Anthropic Messages wire ──────────────────────────────────────────────


class _AnthropicStream:
    def __init__(self, count=10_000, delay=0.01):
        self._count, self._delay = count, delay
        self.yielded = 0
        self.exited = False
        self.response = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def __iter__(self):
        for _ in range(self._count):
            time.sleep(self._delay)
            self.yielded += 1
            yield SimpleNamespace(
                type="content_block_delta", delta=SimpleNamespace(text="tok"),
            )

    def get_final_message(self):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="summary")])


def _anthropic_client(stream):
    return SimpleNamespace(
        messages=SimpleNamespace(
            stream=lambda **_kw: stream,
            create=lambda **_kw: pytest.fail("must not fall back to create()"),
        )
    )


def test_anthropic_stream_stops_at_the_host_deadline():
    stream = _AnthropicStream()
    ticks = []
    with (
        aux.aux_progress_hook(lambda: ticks.append(1)),
        aux.aux_stream_deadline(time.monotonic() + 0.3),
    ):
        hook = aux._anthropic_aux_stream_event_hook()
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="timed out at the host compression deadline"):
            create_anthropic_message(
                _anthropic_client(stream), {"model": "m", "messages": []},
                on_stream_event=hook,
            )
    assert time.monotonic() - start < 5.0
    assert stream.exited, "stream context must be closed on the deadline"
    assert ticks, "substantive deltas must still tick the progress hook"
    assert stream.yielded < 1000


def test_anthropic_stream_honours_an_explicit_hard_cancel():
    stream = _AnthropicStream()
    cancelled = {"v": False}
    with (
        aux.aux_progress_hook(lambda: None),
        aux.aux_interrupt_protection(cancel_check=lambda: cancelled["v"]),
    ):
        hook = aux._anthropic_aux_stream_event_hook()

        def _flip_after_first(event, _inner=hook):
            cancelled["v"] = True
            _inner(event)

        with pytest.raises(aux.AuxiliaryExplicitCancellation):
            create_anthropic_message(
                _anthropic_client(stream), {"model": "m", "messages": []},
                on_stream_event=_flip_after_first,
            )
    assert stream.yielded == 1
    assert stream.exited


def test_anthropic_stream_without_host_deadline_runs_to_completion():
    stream = _AnthropicStream(count=5, delay=0)
    with aux.aux_progress_hook(lambda: None):
        hook = aux._anthropic_aux_stream_event_hook()
        message = create_anthropic_message(
            _anthropic_client(stream), {"model": "m", "messages": []},
            on_stream_event=hook,
        )
    assert message.content[0].text == "summary"
    assert stream.yielded == 5
