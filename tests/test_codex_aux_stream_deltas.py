import types


def test_codex_aux_adapter_collects_stream_deltas_when_final_output_empty():
    """Regression test.

    Some Codex-backed Responses API calls emit text only via
    `response.output_text.delta` events while returning an empty `final.output`
    array. Our Codex Responses -> chat.completions adapter must collect deltas
    so downstream callers (vision, web extract, compression) see content.
    """

    from agent.auxiliary_client import _CodexCompletionsAdapter

    class _Event:
        def __init__(self, delta: str):
            self.type = "response.output_text.delta"
            self.delta = delta

    class _Stream:
        def __init__(self):
            self._events = [_Event("Usage"), _Event(" dashboard")]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(self._events)

        def get_final_response(self):
            # Mimic a ParsedResponse with empty output.
            return types.SimpleNamespace(output=[], usage=None)

    class _Responses:
        def stream(self, **_kwargs):
            return _Stream()

    class _FakeClient:
        def __init__(self):
            self.responses = _Responses()

    adapter = _CodexCompletionsAdapter(real_client=_FakeClient(), model="gpt-5.2-codex")

    resp = adapter.create(messages=[{"role": "user", "content": "hi"}])

    assert resp.choices[0].message.content == "Usage dashboard"
