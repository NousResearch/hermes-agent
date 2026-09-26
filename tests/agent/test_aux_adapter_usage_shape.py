"""Auxiliary wire adapters must hand every usage consumer the provider's real token buckets.

Regression for #71242 (and the MoA reference-slot zeros of #123157). The adapters rebuild
provider usage for ``call_llm`` callers, and each consumer then picks a ``normalize_usage``
shape from the route it knows: session accounting passes only ``provider``, the
``post_auxiliary_call`` hook and MoA reference slots pass the route's own ``api_mode``, and the
MoA aggregator is read as Chat Completions. Whatever the selection, the adapted usage has to
normalize to the same buckets as the native response.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from anthropic.types import Message, TextBlock, Usage
from openai.types.responses import ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from agent.usage_pricing import normalize_usage


def _buckets(usage):
    return (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens,
            usage.cache_write_tokens, usage.reasoning_tokens)


def _anthropic_message(usage: Usage) -> Message:
    return Message(
        id="msg_test", type="message", role="assistant", model="claude-sonnet-4-6",
        content=[TextBlock(type="text", text="ok")], stop_reason="end_turn", stop_sequence=None, usage=usage,
    )


def _adapt_anthropic(native_usage, monkeypatch):
    from agent.auxiliary_client import _AnthropicCompletionsAdapter

    native = _anthropic_message(native_usage)
    monkeypatch.setattr("agent.anthropic_adapter.create_anthropic_message", lambda *a, **k: native)
    adapter = _AnthropicCompletionsAdapter(MagicMock(), "claude-sonnet-4-6")
    return adapter.create(messages=[{"role": "user", "content": "hi"}], max_tokens=32).usage


_ANTHROPIC_USAGE = {
    "cache-read-and-write": Usage(input_tokens=1200, output_tokens=9, cache_read_input_tokens=3000,
                                  cache_creation_input_tokens=250),
    "cache-write-only": Usage(input_tokens=80, output_tokens=10, cache_read_input_tokens=0,
                              cache_creation_input_tokens=4211),
    "no-cache-fields": Usage(input_tokens=5, output_tokens=7),
}


def _adapt_codex(native_usage, monkeypatch):
    from agent.auxiliary_client import _parse_codex_final_response

    return _parse_codex_final_response(SimpleNamespace(output=[], usage=native_usage))[2]


_CODEX_USAGE = {
    "cached-and-reasoning": ResponseUsage(
        input_tokens=4450, input_tokens_details=InputTokensDetails(cached_tokens=3000), output_tokens=90,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=64), total_tokens=4540,
    ),
    "cache-write-dict": {"input_tokens": 1000, "output_tokens": 20, "total_tokens": 1020,
                         "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 100}},
    "no-details": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
}

# wire -> (adapter, the native route's (provider, api_mode), native usage samples,
#          the (provider, api_mode) each consumer passes for a route on that wire)
_WIRES = {
    "anthropic": (_adapt_anthropic, ("anthropic", "anthropic_messages"), _ANTHROPIC_USAGE, {
        "session-accounting-native": ("anthropic", None),
        "session-accounting-other-provider": ("custom", None),
        "hook-and-moa-reference": ("custom", "anthropic_messages"),
        "moa-aggregator": ("moa", "chat_completions"),
    }),
    "codex": (_adapt_codex, ("openai-codex", "codex_responses"), _CODEX_USAGE, {
        "session-accounting": ("openai-codex", None),
        "hook-and-moa-reference": ("openai-codex", "codex_responses"),
        "moa-aggregator": ("moa", "chat_completions"),
    }),
}
_CASES = [
    pytest.param(wire, sample, consumer, id=f"{wire}-{sample}-{consumer}")
    for wire, (_adapt, _route, samples, consumers) in _WIRES.items()
    for sample in samples for consumer in consumers
]


@pytest.mark.parametrize(("wire", "sample", "consumer"), _CASES)
def test_adapted_usage_normalizes_like_the_native_response(wire, sample, consumer, monkeypatch):
    adapt, (native_provider, native_mode), samples, consumers = _WIRES[wire]
    provider, api_mode = consumers[consumer]

    adapted = adapt(samples[sample], monkeypatch)

    truth = normalize_usage(samples[sample], provider=native_provider, api_mode=native_mode)
    assert _buckets(normalize_usage(adapted, provider=provider, api_mode=api_mode)) == _buckets(truth)


def test_native_anthropic_aux_call_records_its_usage_row(tmp_path, monkeypatch):
    """The reported symptom end to end: a provider="anthropic" aux call wrote no usage row."""
    from agent.aux_accounting import reset_accounting_context, set_accounting_context
    from agent.auxiliary_client import call_llm
    from hermes_state import SessionDB

    native = _anthropic_message(_ANTHROPIC_USAGE["cache-read-and-write"])
    monkeypatch.setattr("agent.anthropic_adapter.create_anthropic_message", lambda *a, **k: native)
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli")
    token = set_accounting_context(db, "s1")
    try:
        call_llm(
            task="title_generation", provider="anthropic", model="claude-sonnet-4-6",
            base_url="https://api.anthropic.com", api_key="sk-ant-api03-test",
            messages=[{"role": "user", "content": "Title this: deploy to staging"}], max_tokens=32,
        )
    finally:
        reset_accounting_context(token)

    row = db.auxiliary_usage_by_task("s1").get("title_generation")
    assert row is not None, "the auxiliary call recorded no usage row"
    truth = normalize_usage(native.usage, provider="anthropic")
    assert (row["input_tokens"], row["output_tokens"], row["cache_read_tokens"], row["cache_write_tokens"]) == (
        truth.input_tokens, truth.output_tokens, truth.cache_read_tokens, truth.cache_write_tokens,
    )
