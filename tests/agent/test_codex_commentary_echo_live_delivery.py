"""Live Codex commentary delivery must suppress leaked tool-call echo text (#125458, #125482 review).

``run_codex_stream`` fires ``on_commentary_message`` the moment ``output_item.done`` arrives —
BEFORE ``_normalize_codex_response`` runs, and the settled path drops commentary messages that
carry the echo. Without the same predicate on the live path (``_fire_streamed_codex_commentary``)
the echo is posted to chat as the turn preview even though normalization later discards it.
"""

from types import SimpleNamespace

from agent.codex_runtime import run_codex_stream
from agent.stream_delivery import StreamDeliveryMixin


class _FakeCodexClient:
    def __init__(self, events):
        self.responses = SimpleNamespace(create=lambda **kwargs: iter(events))


class _LiveDeliveryAgent(StreamDeliveryMixin):
    """Minimal agent exercising the real ``_fire_streamed_codex_commentary`` through run_codex_stream."""

    def __init__(self):
        self.delivered = []
        self.interim_assistant_callback = lambda text, *, already_streamed=False: self.delivered.append(text)
        self._delivered_interim_texts = set()
        self._strip_think_blocks = lambda text: text
        # Writer-fence stubs: this attempt always owns the sink.
        self._claim_stream_writer = lambda: 1
        self._stream_writer_is_current = lambda token: True
        self._fire_stream_delta = lambda text: None
        self._fire_reasoning_delta = lambda text: None
        self._interrupt_requested = False
        self.show_commentary = True
        self._touch_activity = lambda message: None
        self._client_log_context = lambda: "test-context"


def _commentary_events(item_id, text):
    return [
        SimpleNamespace(type="response.output_item.added",
                        item=SimpleNamespace(type="message", role="assistant", phase="commentary", id=item_id)),
        SimpleNamespace(type="response.output_text.delta", delta=text, item_id=item_id),
        SimpleNamespace(type="response.output_item.done",
                        item=SimpleNamespace(type="message", role="assistant", phase="commentary", id=item_id,
                                             content=[SimpleNamespace(type="output_text", text=text)])),
        SimpleNamespace(type="response.output_item.added",
                        item=SimpleNamespace(type="function_call", role="assistant", id="fc_1")),
        SimpleNamespace(type="response.output_item.done",
                        item=SimpleNamespace(type="function_call", id="fc_1", call_id="call_1",
                                             name="web_search", status="completed",
                                             arguments='{"query": "test"}')),
        SimpleNamespace(type="response.completed",
                        response=SimpleNamespace(id="resp_1", status="completed", usage=None, output=[],
                                                 incomplete_details=None, error=None)),
    ]


def test_live_delivery_suppresses_harmony_echo_commentary():
    """The already-recognized function echo must not reach the interim callback before normalization."""
    echo = '{"query":"test"}to=functions.web_search junk'
    agent = _LiveDeliveryAgent()

    final = run_codex_stream(agent, {"model": "gpt-5.6-terra"},
                             client=_FakeCodexClient(_commentary_events("c1", echo)))

    assert final.status == "completed"
    assert agent.delivered == []


def test_live_delivery_suppresses_parallel_wrapper_echo_commentary():
    """Production-observed wrapper shape: token leads, wrapper JSON follows across a line break."""
    echo = ('to=multi_tool_use.parallel 0200\n'
            '{"tool_uses":[{"recipient_name":"functions.web_search","parameters":{"query":"test"}}]} junk')
    agent = _LiveDeliveryAgent()

    final = run_codex_stream(agent, {"model": "gpt-5.6-terra"},
                             client=_FakeCodexClient(_commentary_events("c1", echo)))

    assert final.status == "completed"
    assert agent.delivered == []


def test_live_delivery_keeps_clean_narration_commentary():
    """Control: ordinary commentary narration still reaches the interim callback."""
    agent = _LiveDeliveryAgent()

    final = run_codex_stream(agent, {"model": "gpt-5.6-terra"},
                             client=_FakeCodexClient(_commentary_events("c1", "Checking the failing test first.")))

    assert final.status == "completed"
    assert agent.delivered == ["Checking the failing test first."]


def test_live_delivery_matches_settled_drop_for_bare_token_narration():
    """Parity control: commentary narration naming a bare ``to=functions.*`` token is dropped by the
    settled path's broad commentary match, so the live path applies the same predicate — the preview
    must never show text that normalization discards."""
    narration = "Reference: to=functions.web_search for details."
    agent = _LiveDeliveryAgent()

    final = run_codex_stream(agent, {"model": "gpt-5.6-terra"},
                             client=_FakeCodexClient(_commentary_events("c1", narration)))

    assert final.status == "completed"
    assert agent.delivered == []
