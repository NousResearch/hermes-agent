"""A reference model that answers with no text is a failed reference, not advice.

Regression for #159793: providers report refusals and content-filter stops as HTTP 200 with empty
content, and the aggregator used to receive that as "(empty response)" alongside real advice.
"""

from types import SimpleNamespace

from agent.moa_loop import _guidance_inputs, _run_reference


def _empty_response(finish_reason: str):
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=0,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
                            output_tokens_details=None)
    message = SimpleNamespace(content="", tool_calls=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
                           usage=usage, model="fake-model")


def test_an_empty_reference_reaches_the_aggregator_as_degraded_not_as_advice(monkeypatch):
    monkeypatch.setattr("agent.moa_loop.call_llm", lambda **kw: _empty_response("content_filter"))
    monkeypatch.setattr("agent.moa_loop._slot_runtime", lambda slot: {"provider": "openrouter", "model": slot.get("model")})

    label, text, acct = _run_reference({"provider": "openrouter", "model": "vendor/advisor"},
                                       [{"role": "user", "content": "plan?"}])

    assert "content_filter" in text
    assert acct.usage.input_tokens == 120  # the provider billed the call; it still counts
    advice, notice, all_failed = _guidance_inputs([(label, text, acct)], privacy_full=False, policy="loud")
    assert advice == [] and all_failed
    assert label in notice
