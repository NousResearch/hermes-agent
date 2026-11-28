"""Auxiliary calls served by Anthropic must reach the usage ledger with their real token counts."""
from types import SimpleNamespace

from agent import aux_accounting, usage_pricing
from agent.usage_pricing import normalize_usage


def _record(response, provider):
    rows = []
    db = SimpleNamespace(record_auxiliary_usage=lambda *args, **kwargs: rows.append(kwargs))
    token = aux_accounting.set_accounting_context(db, "session-1")
    try:
        aux_accounting.record_aux_usage(response, "title_generation", provider=provider)
    finally:
        aux_accounting.reset_accounting_context(token)
    return rows


def test_anthropic_normalizer_falls_back_to_chat_field_names():
    usage = normalize_usage(SimpleNamespace(prompt_tokens=120, completion_tokens=30), provider="anthropic")
    assert (usage.input_tokens, usage.output_tokens) == (120, 30)


def test_anthropic_native_fields_still_win():
    usage = normalize_usage(
        SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=400,
                        cache_creation_input_tokens=50, prompt_tokens=999, completion_tokens=999),
        provider="anthropic",
    )
    assert (usage.input_tokens, usage.output_tokens) == (100, 20)
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (400, 50)


def test_chat_shaped_anthropic_aux_response_is_recorded(monkeypatch):
    monkeypatch.setattr(usage_pricing, "estimate_usage_cost",
                        lambda model, usage, **kw: SimpleNamespace(amount_usd=0.0))
    response = SimpleNamespace(
        model="claude-haiku", usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150))

    rows = _record(response, "anthropic")

    assert len(rows) == 1
    assert (rows[0]["input_tokens"], rows[0]["output_tokens"]) == (120, 30)


def test_adapter_usage_keeps_anthropic_cache_fields(monkeypatch):
    """The aux Anthropic adapter's usage object carries both shapes, so cache reads are priced."""
    import agent.anthropic_adapter as anthropic_adapter
    import agent.transports as transports
    from agent.auxiliary_client import _AnthropicCompletionsAdapter

    native = SimpleNamespace(input_tokens=80, output_tokens=12, cache_read_input_tokens=640,
                             cache_creation_input_tokens=0)
    monkeypatch.setattr(anthropic_adapter, "create_anthropic_message",
                        lambda client, kwargs, **kw: SimpleNamespace(usage=native))
    normalized = SimpleNamespace(content="ok", tool_calls=None, reasoning=None, finish_reason="stop")
    monkeypatch.setattr(transports, "get_transport",
                        lambda name: SimpleNamespace(normalize_response=lambda resp, **kw: normalized))
    adapter = _AnthropicCompletionsAdapter(object(), "claude-haiku", base_url="https://api.anthropic.com")

    response = adapter.create(model="claude-haiku", messages=[{"role": "user", "content": "hi"}], max_tokens=16)

    usage = normalize_usage(response.usage, provider="anthropic")
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (80, 12, 640)
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (80, 12)
