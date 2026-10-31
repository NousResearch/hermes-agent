"""Auxiliary Responses usage must keep cached/reasoning tokens and the raw usage (#135657).

``_parse_codex_final_response`` reshapes a Responses final into a Chat Completions
response. Its usage namespace used to carry only the three totals, so aux accounting
(``record_aux_usage`` → ``normalize_usage`` without ``api_mode`` → Chat shape) booked
cached input as uncached, dropped reasoning tokens, and lost the raw usage a provider
``get_usage_cost`` reads (e.g. xAI's ``cost_in_usd_ticks``).
"""

from types import SimpleNamespace

from agent.auxiliary_codex_response import _parse_codex_final_response
from agent.usage_pricing import normalize_usage


def _final_with_usage(usage):
    message = SimpleNamespace(
        type="message", role="assistant", phase="final_answer", status="completed",
        content=[SimpleNamespace(type="output_text", text="ok")],
    )
    return SimpleNamespace(
        status="completed", output=[message], output_text=None,
        incomplete_details=None, error=None, usage=usage,
    )


# xAI-style Responses usage: totals include cached input; cost rides the raw payload.
_OBJECT_USAGE = SimpleNamespace(
    input_tokens=1000, output_tokens=200, total_tokens=1200,
    input_tokens_details=SimpleNamespace(cached_tokens=600),
    output_tokens_details=SimpleNamespace(reasoning_tokens=150),
    cost_in_usd_ticks=26078,
)

_DICT_USAGE = {
    "input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200,
    "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 50},
    "output_tokens_details": {"reasoning_tokens": 150},
    "cost_in_usd_ticks": 26078,
}


def test_chat_shape_keeps_cached_and_reasoning_tokens():
    _, _, usage, _ = _parse_codex_final_response(_final_with_usage(_OBJECT_USAGE))
    # record_aux_usage normalizes without api_mode: provider "xai-oauth" → Chat shape.
    cu = normalize_usage(usage, provider="xai-oauth")
    assert (cu.cache_read_tokens, cu.reasoning_tokens) == (600, 150)
    # Chat/Codex totals INCLUDE cached tokens; the canonical input excludes them.
    assert cu.input_tokens == 400
    assert cu.prompt_tokens == 1000


def test_codex_shape_keeps_cached_and_reasoning_tokens():
    _, _, usage, _ = _parse_codex_final_response(_final_with_usage(_DICT_USAGE))
    # Aux hooks pass the route's api_mode: the Responses shape must read the same buckets.
    cu = normalize_usage(usage, provider="xai-oauth", api_mode="codex_responses")
    assert (cu.cache_read_tokens, cu.cache_write_tokens, cu.reasoning_tokens) == (600, 50, 150)
    assert cu.input_tokens == 350  # 1000 - 600 cached - 50 written


def test_raw_usage_survives_for_provider_cost_hooks():
    _, _, usage, _ = _parse_codex_final_response(_final_with_usage(_DICT_USAGE))
    cu = normalize_usage(usage, provider="xai-oauth")
    assert cu.raw_usage["cost_in_usd_ticks"] == 26078
    assert cu.raw_usage["input_tokens_details"]["cached_tokens"] == 600


def test_totals_only_usage_still_normalizes():
    # Hosts that report bare totals must keep working (zero cache/reasoning buckets).
    _, _, usage, _ = _parse_codex_final_response(
        _final_with_usage(SimpleNamespace(input_tokens=11, output_tokens=3, total_tokens=14))
    )
    cu = normalize_usage(usage, provider="xai-oauth")
    assert (cu.input_tokens, cu.output_tokens) == (11, 3)
    assert (cu.cache_read_tokens, cu.reasoning_tokens) == (0, 0)


def test_record_aux_usage_books_cached_and_reasoning(monkeypatch):
    from agent import aux_accounting

    recorded = {}

    class FakeSessionDB:
        def record_auxiliary_usage(self, session_id, task, **kwargs):
            recorded.update(kwargs)

    # Keep the e2e path offline: cost estimation is not under test here.
    monkeypatch.setattr(
        "agent.usage_pricing.estimate_usage_cost",
        lambda *a, **k: SimpleNamespace(amount_usd=None),
    )
    _, _, usage, _ = _parse_codex_final_response(_final_with_usage(_OBJECT_USAGE))
    token = aux_accounting.set_accounting_context(FakeSessionDB(), "s1")
    try:
        aux_accounting.record_aux_usage(
            SimpleNamespace(usage=usage, model="grok-4.6"), "title_generation", provider="xai-oauth",
        )
    finally:
        aux_accounting.reset_accounting_context(token)
    assert recorded["cache_read_tokens"] == 600
    assert recorded["reasoning_tokens"] == 150
    assert recorded["input_tokens"] == 400
    assert recorded["output_tokens"] == 200
