"""The served model is recorded at the turn seam.

``agent/served_model.py``'s httpx hook sees response HEADERS only — reading a body there
buffers the whole SSE stream — so the served MODEL (``response.model``, which the router sets
to whoever actually answered) is read at the turn seam instead. Two contracts:

* it is compared against the model SENT on the wire, not ``agent.model``: OpenRouter's
  speed-tier pins rewrite the outgoing model, so a pinned slug answering with the base slug is
  not a reroute;
* a same-or-absent answer CLEARS the stamp (``tests/agent/test_served_model.py`` pins
  clear-on-absent), so a stale served model can never leak into a later turn result.

Synthetic responses only — no provider I/O, no real client.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from types import SimpleNamespace

import pytest


class _Compressor:
    threshold_tokens = 1_000_000
    awaiting_real_usage_after_compression = False
    _verify_compaction_cleared_threshold = False

    def update_from_response(self, usage_dict):
        pass

    def note_usage_less_response(self):
        pass


class _DB:
    def queue_token_counts(self, session_id, **kwargs):
        pass

    def patch_session_model_config(self, *args, **kwargs):
        return None


def _agent(model="openai/gpt-6-astra-fast", **overrides):
    agent = SimpleNamespace(
        model=model, provider="openrouter", api_mode="chat_completions",
        base_url="https://openrouter.ai/api/v1", api_key="k", session_id="t",
        quiet_mode=True, verbose_logging=False, log_prefix="", client=None,
        context_compressor=_Compressor(), _session_db=_DB(), _session_db_created=True,
        _safe_print=lambda *a, **k: None, _vprint=lambda *a, **k: None,
        session_api_calls=0, session_prompt_tokens=0, session_completion_tokens=0,
        session_total_tokens=0, session_input_tokens=0, session_output_tokens=0,
        session_cache_read_tokens=0, session_cache_write_tokens=0,
        session_reasoning_tokens=0, session_estimated_cost_usd=0.0,
        session_actual_cost_usd=0.0, session_cost_status="unknown", session_cost_source="none",
        _usage_anchor=None, _turn_base_usage_anchor=None, _persist_disabled=True,
        _api_latency_history=None, _api_output_history=None,
        last_served_model=None,
    )
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


def _response(model, provider="SomeUpstream", usage=True):
    return SimpleNamespace(
        usage={"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
               "cost": 0.0004} if usage else None,
        model=model, provider=provider,
    )


_UNSET = object()


def _run(agent, response, monkeypatch, wire_model=_UNSET):
    from agent import turn_usage

    monkeypatch.setattr(
        turn_usage, "estimate_usage_cost",
        lambda *a, **k: SimpleNamespace(amount_usd=Decimal("0.1"), status="estimated",
                                        source="provider_models_api"),
    )
    monkeypatch.setattr(
        turn_usage, "_loop_mod",
        lambda: SimpleNamespace(_should_rearm_compression_budget=lambda *a, **k: False),
    )
    kwargs = {} if wire_model is _UNSET else {"wire_model": wire_model}
    turn_usage.record_response_usage(
        agent, response, messages=[{"role": "user", "content": "hi"}], api_call_count=1,
        api_duration=0.2, compression_attempts=0, max_compression_attempts=3, **kwargs,
    )


def test_differing_served_model_is_captured_and_logged(monkeypatch, caplog):
    agent = _agent()
    with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
        _run(agent, _response("anthropic/claude-opus-4.7"), monkeypatch,
             wire_model="openai/gpt-6-astra-fast")

    assert agent.last_served_model == "anthropic/claude-opus-4.7"
    line = next(r.getMessage() for r in caplog.records if "API call #1" in r.getMessage())
    assert "served=anthropic/claude-opus-4.7" in line
    # The requested model is still what the line/agent reports; served is additive.
    assert "model=openai/gpt-6-astra-fast" in line


def test_same_model_captures_nothing_and_logs_no_served_field(monkeypatch, caplog):
    agent = _agent()
    with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
        _run(agent, _response("openai/gpt-6-astra-fast"), monkeypatch,
             wire_model="openai/gpt-6-astra-fast")

    assert agent.last_served_model is None
    line = next(r.getMessage() for r in caplog.records if "API call #1" in r.getMessage())
    assert "served=" not in line


def test_absent_served_model_clears_a_previous_stamp(monkeypatch):
    """A stale value must never survive into the next turn result (clear-on-absent)."""
    agent = _agent()
    _run(agent, _response("anthropic/claude-opus-4.7"), monkeypatch, wire_model="openai/gpt-6-astra-fast")
    assert agent.last_served_model == "anthropic/claude-opus-4.7"

    _run(agent, _response(None), monkeypatch, wire_model="openai/gpt-6-astra-fast")
    assert agent.last_served_model is None

    from agent.served_model import result_model_fields
    assert result_model_fields(agent)["served_model"] is None


def test_usage_less_response_still_clears_a_previous_stamp(monkeypatch):
    """The clear must not depend on the usage fold running."""
    agent = _agent()
    _run(agent, _response("anthropic/claude-opus-4.7"), monkeypatch, wire_model="openai/gpt-6-astra-fast")
    _run(agent, _response(None, usage=False), monkeypatch, wire_model="openai/gpt-6-astra-fast")
    assert agent.last_served_model is None


def test_endpoint_pin_base_slug_is_not_a_reroute(monkeypatch):
    """OpenRouter speed-tier pins send the BASE slug on the wire; the router answering with
    that same slug must not produce a bogus served= value (compare the wire model, not
    agent.model)."""
    from providers import get_provider_profile

    agent = _agent(model="openai/gpt-6-astra-fast")
    # Real profile: the pin rewrites top_level["model"] to the base slug.
    _, top_level = get_provider_profile("openrouter").build_api_kwargs_extras(model=agent.model)
    import inspect
    base = inspect.getmodule(type(get_provider_profile("openrouter"))).OPENROUTER_ENDPOINT_PINS[agent.model][0]
    assert top_level["model"] == base

    from agent.turn_usage import wire_model_from_api_kwargs
    api_kwargs = {"model": agent.model, **top_level}
    assert wire_model_from_api_kwargs(api_kwargs) == base

    _run(agent, _response(base), monkeypatch, wire_model=wire_model_from_api_kwargs(api_kwargs))
    assert agent.last_served_model is None


def test_pin_with_a_real_reroute_is_still_captured(monkeypatch):
    """Control for the pin test: a pin whose route actually lands elsewhere still reports."""
    agent = _agent(model="openai/gpt-6-astra-fast")
    _run(agent, _response("openai/gpt-6-astra"),
         monkeypatch, wire_model="openai/gpt-6-astra")
    assert agent.last_served_model is None  # same as the wire model -> nothing to report

    _run(agent, _response("azure/gpt-6-astra"),
         monkeypatch, wire_model="openai/gpt-6-astra")
    assert agent.last_served_model == "azure/gpt-6-astra"


def test_wire_model_falls_back_to_agent_model(monkeypatch):
    """No api_kwargs plumbed (e.g. an exotic dispatch): compare against agent.model, as before."""
    agent = _agent(model="hermes-router")
    _run(agent, _response("anthropic/claude-opus-4.7"), monkeypatch)
    assert agent.last_served_model == "anthropic/claude-opus-4.7"


def test_header_capture_from_the_hook_is_not_clobbered_by_a_matching_body(monkeypatch):
    """The proxy deployment id (header hook) is more specific than a body echo of the wire
    model: a matching body must clear only this seam's own stamp."""
    agent = _agent()
    agent.last_served_model = "gpt-4o-2024-11-20"  # as agent/served_model.py's hook left it
    _run(agent, _response(agent.model), monkeypatch, wire_model=agent.model)
    assert agent.last_served_model == "gpt-4o-2024-11-20"


def test_turn_result_surfaces_the_captured_model(monkeypatch):
    from agent.served_model import result_model_fields

    agent = _agent()
    _run(agent, _response("anthropic/claude-opus-4.7"), monkeypatch, wire_model="openai/gpt-6-astra-fast")
    fields = result_model_fields(agent)
    assert fields["served_model"] == "anthropic/claude-opus-4.7"
    assert fields["requested_model"] == "openai/gpt-6-astra-fast"


@pytest.mark.parametrize("api_kwargs", [None, {}, {"model": ""}, {"model": 5}])
def test_wire_model_helper_is_fail_open(api_kwargs):
    from agent.turn_usage import wire_model_from_api_kwargs

    assert wire_model_from_api_kwargs(api_kwargs) is None
