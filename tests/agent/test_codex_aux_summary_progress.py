from types import SimpleNamespace

from agent import auxiliary_client, codex_runtime


def test_summary_progress_ignores_reasoning_phase_but_accepts_final_text():
    events = [
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="message", phase="analysis", id="reasoning"),
        ),
        SimpleNamespace(type="response.output_text.delta", delta="thinking"),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="message",
                phase="analysis",
                id="reasoning",
                content=[SimpleNamespace(type="output_text", text="thinking")],
            ),
        ),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="message", phase=None, id="answer"),
        ),
        SimpleNamespace(type="response.output_text.delta", delta="summary"),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                id="resp_1",
                status="completed",
                usage=None,
                output=None,
            ),
        ),
    ]
    ticks = []
    seen = []

    def _on_event(event):
        seen.append(event.type)
        # Mirror the production Codex auxiliary callback: every SSE frame is
        # transport activity, while the wrapper decides whether this provider
        # response may reach the outer summary-progress hook.
        auxiliary_client._notify_aux_provider_response()

    with auxiliary_client.aux_progress_hook(lambda: ticks.append("result")):
        response = codex_runtime._consume_codex_event_stream(
            iter(events),
            model="gpt-5.6-sol",
            on_event=_on_event,
        )

    assert response.output_text == "summary"
    assert seen == [event.type for event in events]
    assert ticks == ["result"]


def test_summary_progress_classifier_separates_reasoning_from_results():
    assert codex_runtime._codex_event_advances_aux_result(
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="thought"),
        None,
    ) is False
    assert codex_runtime._codex_event_advances_aux_result(
        SimpleNamespace(type="response.output_text.delta", delta="summary"),
        None,
    ) is True
    assert codex_runtime._codex_event_advances_aux_result(
        SimpleNamespace(
            type="response.output_text.delta",
            delta="analysis chatter",
        ),
        "analysis",
    ) is False
    assert codex_runtime._codex_event_advances_aux_result(
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(type="function_call", name="lookup", arguments="{}"),
        ),
        None,
    ) is True


def test_codex_runtime_facade_preserves_mutable_module_globals(monkeypatch):
    original_event_field = codex_runtime._event_field
    seen = []

    def recording_event_field(event, name, default=None):
        seen.append(name)
        return original_event_field(event, name, default)

    monkeypatch.setattr(codex_runtime, "_event_field", recording_event_field)

    result = codex_runtime._consume_codex_event_stream(
        iter(
            [
                SimpleNamespace(type="response.output_text.delta", delta="ok"),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        id="resp_2",
                        status="completed",
                        usage=None,
                        output=None,
                    ),
                ),
            ]
        ),
        model="gpt-5.6-sol",
        on_event=lambda _event: None,
    )

    assert result.output_text == "ok"
    assert "type" in seen
