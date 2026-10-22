"""Native Anthropic consumption must obey a bounded assistance deadline."""
import json
from types import SimpleNamespace

import httpx
import pytest


@pytest.mark.parametrize("progress_active", [False, True])
def test_native_stream_checks_deadline_without_progress_hook(monkeypatch, progress_active):
    import anthropic
    from agent import auxiliary_client as aux, connected_assistance as pipeline
    from agent.auxiliary_call_budget import AuxiliaryCallBudget
    from tools.connectors import assistance as source_module

    clock, consumed = [100.0], []
    events = [
        {"type": "message_start", "message": {"id": "msg_fixture", "type": "message", "role": "assistant", "content": [], "model": "claude-fixture", "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": '{"reads":[]}'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            for index, event in enumerate(events):
                if index >= 2:
                    clock[0] = 161.0 + index
                consumed.append(clock[0])
                yield ("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n").encode()

    def request(_request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())

    sdk = anthropic.Anthropic(api_key="fixture", max_retries=0, base_url="https://fixture.test",
                             http_client=httpx.Client(transport=httpx.MockTransport(request)))
    client = aux.AnthropicAuxiliaryClient(sdk, "claude-fixture", "fixture", "https://fixture.test")
    monkeypatch.setattr(aux, "_resolve_task_provider_model", lambda *a, **kw: ("anthropic", "claude-fixture", "https://fixture.test", "fixture", None))
    monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **kw: (client, "claude-fixture"))
    for module in (aux, pipeline, source_module):
        monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    try:
        try:
            with aux.aux_progress_hook((lambda: None) if progress_active else None):
                pipeline._model_json(None, "Return JSON.", {}, 160.0,
                                     AuxiliaryCallBudget(160.0, 2, clock=lambda: clock[0]))
        except TimeoutError:
            pass
        assert len(consumed) >= 3
        assert len([at for at in consumed if at >= 160.0]) <= 1
    finally:
        client.close()
