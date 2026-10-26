"""A Codex Responses stream that loses the delta sink to a newer attempt keeps consuming (#69486).

Stopping consumption on supersession returned a ``completed`` response missing its tail, so the gateway
delivered a truncated reply. Supersession must fence only the live callbacks (text/reasoning deltas,
commentary, first-delta) and still assemble the complete final response.

Grafted from PR #69502 (@byungsker) onto the Relay-backed ``run_codex_stream``.
"""

from types import SimpleNamespace

from agent.codex_runtime import run_codex_stream


class _FakeCodexClient:
    def __init__(self, events):
        self.responses = SimpleNamespace(create=lambda **kwargs: iter(events))


def _completed():
    return SimpleNamespace(type="response.completed",
                           response=SimpleNamespace(id="resp_1", status="completed", usage=None, output=[],
                                                    incomplete_details=None, error=None))


def _message_added(phase=None, item_id="m1"):
    return SimpleNamespace(type="response.output_item.added",
                           item=SimpleNamespace(type="message", role="assistant", phase=phase, id=item_id))


def _agent(supersede_after_checks: int):
    """Duck-typed agent whose writer fence reports supersession after ``supersede_after_checks`` checks."""
    live = {"deltas": [], "reasoning": [], "commentary": [], "first_delta": 0, "checks": 0}

    def is_current(_token):
        live["checks"] += 1
        return live["checks"] <= supersede_after_checks

    agent = SimpleNamespace(
        _interrupt_requested=False, show_commentary=True,
        _claim_stream_writer=lambda: 1, _stream_writer_is_current=is_current,
        _fire_stream_delta=live["deltas"].append, _fire_reasoning_delta=live["reasoning"].append,
        _fire_streamed_codex_commentary=live["commentary"].append,
        interim_assistant_callback=lambda *a, **k: None,
        _touch_activity=lambda _message: None, _client_log_context=lambda: "test-context",
    )
    return agent, live


def test_superseded_stream_assembles_complete_final_and_fences_live_deltas():
    events = [
        _message_added(),
        SimpleNamespace(type="response.output_text.delta", delta="I've added the live"),
        SimpleNamespace(type="response.output_text.delta", delta=" tail."),
        SimpleNamespace(type="response.reasoning_text.delta", delta="late reasoning"),
        _completed(),
    ]
    agent, live = _agent(supersede_after_checks=1)

    final = run_codex_stream(agent, {"model": "gpt-5.6-terra"}, client=_FakeCodexClient(events))

    assert final.status == "completed"
    assert final.output_text == "I've added the live tail."
    assert agent._codex_streamed_text_parts == ["I've added the live", " tail."]
    assert live["deltas"] == ["I've added the live"]
    assert live["reasoning"] == []


def test_superseded_stream_fences_commentary_and_first_delta_but_keeps_final():
    events = [
        _message_added(phase="commentary", item_id="c1"),
        SimpleNamespace(type="response.output_text.delta", delta="Let me check.", item_id="c1"),
        SimpleNamespace(type="response.output_item.done",
                        item=SimpleNamespace(type="message", role="assistant", phase="commentary", id="c1",
                                             content=[SimpleNamespace(type="output_text", text="Let me check.")])),
        _message_added(item_id="m1"),
        SimpleNamespace(type="response.output_text.delta", delta="Done.", item_id="m1"),
        _completed(),
    ]
    agent, live = _agent(supersede_after_checks=0)
    first_delta = []

    final = run_codex_stream(agent, {"model": "gpt-5.6-terra"}, client=_FakeCodexClient(events),
                             on_first_delta=lambda: first_delta.append(True))

    assert final.status == "completed"
    assert final.output_text == "Done."
    assert live["commentary"] == []
    assert live["deltas"] == []
    assert first_delta == []
