"""Auxiliary parameter-rejection rungs chain in any order, on the primary AND the fallback path.

Reasoning models reject several request fields at once (``temperature`` and ``max_tokens`` on
gpt-5, #78273), a reasoning-strip retry can then 400 on ``temperature`` (#72351), and strict-schema
gateways phrase an unknown ``reasoning`` field as "Extra inputs are not permitted" (#109774).
Each rejected field must be stripped in whatever order the provider raises them, and a fallback
candidate must get the same recovery as the primary request.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.auxiliary_client import (
    _call_fallback_candidate_sync,
    _drive_ladder,
    _is_reasoning_field_rejection,
    _ladder_parameter_rungs,
    _LadderRoute,
)


class _Bad400(Exception):
    status_code = 400


def _ok(text="ok"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])


def _rejecting_client(*rejected_fields):
    """Fake OpenAI client that 400s while any of ``rejected_fields`` is still on the wire."""
    client = MagicMock(base_url="https://api.example/v1")

    def create(**kwargs):
        body = dict(kwargs)
        body.update(body.pop("extra_body", None) or {})
        for field in rejected_fields:
            if field in body:
                phrase = {
                    "reasoning": "Extra inputs are not permitted, field: 'reasoning'",
                    "reasoning_effort": "Unsupported value: 'reasoning_effort' does not support 'none' with this model.",
                    "temperature": "Unsupported value: 'temperature' does not support 0.3 with this model.",
                    "max_tokens": "Unsupported parameter: 'max_tokens' is not supported with this model.",
                }[field]
                raise _Bad400(f"Error code: 400 - {phrase}")
        return _ok()

    client.chat.completions.create.side_effect = create
    return client


def test_primary_rungs_chain_in_provider_order_and_strip_each_field_once():
    client = _rejecting_client("reasoning_effort", "temperature", "max_tokens")
    kwargs = {"model": "m", "messages": [], "temperature": 0.3, "max_tokens": 64, "reasoning_effort": "none"}
    route = _LadderRoute(**{**dict.fromkeys(_LadderRoute._fields), "client": client, "task": "title_generation",
                            "tag": "", "async_mode": False, "base_info": "", "resolved_provider": ""})
    first_err = _Bad400("Error code: 400 - Unsupported value: 'reasoning_effort' does not support 'none'")

    def perform(step):
        step_client, step_kwargs = step.args
        return step_client.chat.completions.create(**step_kwargs)

    resp, err, final_kwargs = _drive_ladder(_ladder_parameter_rungs(first_err, route, kwargs, 64), perform)

    assert err is None and resp.choices[0].message.content == "ok"
    assert not {"temperature", "max_tokens", "reasoning_effort"} & set(final_kwargs)
    # One retry per rejected field: reasoning_effort → temperature → max_tokens, nothing re-sent unchanged.
    assert client.chat.completions.create.call_count == 3
    assert _is_reasoning_field_rejection(_Bad400("Extra inputs are not permitted, field: 'reasoning'"))


def test_reasoning_effort_none_unsupported_reversed_wording():
    """Relays that put the adjective last (``reasoning_effort 'none' unsupported; use ...``) fire the
    strip-and-retry rung like the forward wordings do; route gating that merely names a thinking
    model, or an adjective-only "unsupported" far from any reasoning field, does not."""
    assert _is_reasoning_field_rejection(
        _Bad400("Error code: 400 - reasoning_effort 'none' unsupported; use minimal|low|medium|high|xhigh")
    )
    assert not _is_reasoning_field_rejection(
        _Bad400("The model kimi-k2-thinking is not supported when using this account")
    )
    assert not _is_reasoning_field_rejection(_Bad400("reasoning models: tool_choice 'required' is unsupported"))


def test_venice_reasoning_value_constraint_wording_is_a_field_rejection():
    """Venice validates its nested reasoning object as max_tokens-shaped; its value error must strip
    that object, including close schema wording variants, while a plain output-cap error remains separate."""
    assert _is_reasoning_field_rejection(_Bad400("reasoning.max_tokens must be positive"))
    assert _is_reasoning_field_rejection(_Bad400("reasoning.max_tokens must be greater than 0"))
    assert _is_reasoning_field_rejection(_Bad400("reasoning.max_tokens must be at least 1"))
    assert not _is_reasoning_field_rejection(_Bad400("max_tokens must be positive"))


def test_structured_param_rejection_strips_reasoning_effort_on_retry():
    """commandcode.ai rejects ``reasoning_effort`` as an enum violation with no "unsupported" marker
    (#115277) and a custom Responses relay sends a message-less structured 400 whose only signal is
    ``param`` / ``invalid_reasoning_effort`` (#100536). Both must land the strip-and-retry rung: the
    second call goes out without ``reasoning_effort`` and succeeds."""
    client = MagicMock(base_url="https://api.example/v1")

    def create(**kwargs):
        body = dict(kwargs)
        body.update(body.pop("extra_body", None) or {})
        if "reasoning_effort" in body:
            raise _Bad400(
                "Error code: 400 - {'error': {'param': 'reasoning.effort', "
                "'error_code': 'invalid_reasoning_effort', 'retryable': False}}"
            )
        return _ok()

    client.chat.completions.create.side_effect = create
    resp = _call_fallback_candidate_sync(
        client, "custom-relay", "fallback_chain[0](custom)", task="title_generation",
        messages=[{"role": "user", "content": "hi"}], temperature=0.3, max_tokens=16, tools=None,
        effective_timeout=30.0, effective_extra_body={}, reasoning_config={"enabled": True, "effort": "max"},
    )
    assert resp.choices[0].message.content == "ok"
    sent = [c.kwargs for c in client.chat.completions.create.call_args_list]
    assert [("reasoning_effort" in k) for k in sent] == [True, False]


def test_fallback_candidate_recovers_from_rejected_temperature():
    client = _rejecting_client("temperature")
    resp = _call_fallback_candidate_sync(
        client, "relay-model-x", "fallback_chain[0](openai)", task="title_generation",
        messages=[{"role": "user", "content": "hi"}], temperature=0.3, max_tokens=16, tools=None,
        effective_timeout=30.0, effective_extra_body={}, reasoning_config=None,
    )
    assert resp.choices[0].message.content == "ok"
    sent = [c.kwargs for c in client.chat.completions.create.call_args_list]
    assert [("temperature" in k) for k in sent] == [True, False]


def test_rate_limit_after_parameter_strip_falls_through_to_later_rungs():
    """A 429 on the stripped retry belongs to the credential/provider-fallback rungs; the
    parameter rungs must hand it on with the stripped kwargs, not raise out of the ladder."""
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.example/v1/chat/completions")
    rate_limited = openai.RateLimitError(
        "Error code: 429 - Rate limit exceeded", body=None,
        response=httpx.Response(429, request=request, json={"error": {"message": "Rate limit exceeded"}}))
    client = MagicMock(base_url="https://api.example/v1")
    client.chat.completions.create.side_effect = rate_limited
    route = _LadderRoute(**{**dict.fromkeys(_LadderRoute._fields), "client": client, "task": "title_generation",
                            "tag": "", "async_mode": False, "base_info": "", "resolved_provider": ""})
    kwargs = {"model": "m", "messages": [], "max_tokens": 64}
    first_err = _Bad400("Error code: 400 - Unsupported parameter: 'max_tokens' is not supported with this model.")

    resp, err, final_kwargs = _drive_ladder(
        _ladder_parameter_rungs(first_err, route, kwargs, 64),
        lambda step: step.args[0].chat.completions.create(**step.args[1]))

    assert resp is None and err is rate_limited
    assert "max_tokens" not in final_kwargs
