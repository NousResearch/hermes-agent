"""OpenAI-spec routes get the full lifecycle (no cache_control markers: one automatic cache, 30 minutes
renewed by a read), and a mid-chat route switch reads as a cold cache so the next request offloads first."""

from decimal import Decimal
from types import SimpleNamespace

import pytest

import agent.prompt_cache_lifecycle as pcl
from agent.prompt_cache_clock import attempt_facts, record_cache_response
from agent.usage_pricing import PricingEntry, normalize_usage

_CHEAP = PricingEntry(input_cost_per_million=Decimal("0.09"), output_cost_per_million=Decimal("0.18"),
                      cache_read_cost_per_million=Decimal("0.018"), cache_write_cost_per_million=None,
                      source="provider_models_api")


class _Timer:
    def __init__(self, delay, fn, args=()):
        self.delay, self.fn, self.args, self.daemon, self.name = delay, fn, args, True, ""

    def start(self):
        pass

    def cancel(self):
        pass


class _Client:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @staticmethod
    def effective_output_cap(*, max_tokens=None, max_completion_tokens=None, **_):
        return max_completion_tokens or max_tokens  # an SDK-shaped double: sends the cap it is given

    def with_options(self, **kw):
        return self

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(choices=[], usage=SimpleNamespace(
            prompt_tokens=40_000, completion_tokens=1, prompt_tokens_details=SimpleNamespace(cached_tokens=39_900)))


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    for key in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT", "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("agent.usage_pricing.get_pricing_entry", lambda *a, **k: _CHEAP)
    monkeypatch.setattr(pcl.threading, "Timer", _Timer)


def _agent(model):
    client = _Client()
    return SimpleNamespace(
        api_mode="chat_completions", _use_prompt_caching=False, _cache_disabled=False, _interrupt_requested=False,
        client=client, model=model, provider="nous", base_url="https://inference-api.nousresearch.com/v1",
        api_key="", session_api_calls=3, session_prompt_tokens=0, session_completion_tokens=0,
        session_total_tokens=0, session_input_tokens=0, session_output_tokens=0, session_cache_read_tokens=0,
        session_cache_write_tokens=0, session_reasoning_tokens=0, session_estimated_cost_usd=0.0, _session_db=None, session_id="s",
        log_prefix="", platform="cli", compression_enabled=True, _user_turn_count=1,
        context_compressor=SimpleNamespace(compression_count=0, last_prompt_tokens=40_000, threshold_tokens=60_000,
                                           summary_target_ratio=0.2, last_compression_rough_tokens=0,
                                           awaiting_real_usage_after_compression=False,
                                           get_active_compression_failure_cooldown=lambda: None),
    )


def _respond(agent, request):
    usage = SimpleNamespace(prompt_tokens=40_000, completion_tokens=20,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=39_000))
    agent._prompt_cache_attempt = attempt_facts(request)
    record_cache_response(agent, normalize_usage(usage, provider=agent.provider, api_mode=agent.api_mode),
                          model=agent.model, provider=agent.provider, base_url=agent.base_url, api_mode=agent.api_mode)
    return pcl.get_prompt_cache_warmer(agent).start(request, SimpleNamespace(usage=usage))


@pytest.mark.parametrize("model, cap", [("deepseek/deepseek-v4-flash", 1), ("openai/gpt-5.6-luna", 16)])
def test_an_openai_spec_route_is_warmed_at_25_minutes_and_replayed_with_the_smallest_accepted_cap(model, cap):
    agent = _agent(model)
    request = {"model": model, "max_tokens": 4096, "stream": True, "messages": [
        {"role": "system", "content": "sys " * 9000}, {"role": "user", "content": "hi"}]}
    assert _respond(agent, request) is None
    pcl.end_turn_prompt_cache_warming(agent)
    warmer = pcl.get_prompt_cache_warmer(agent)
    run = warmer._run
    assert run.ttl_s == 1800 and run.deadline_at - run.next_warm_at == pytest.approx(300, rel=0, abs=1e-6)
    warmer._refresh(run)
    (sent,) = agent.client.calls
    assert sent["max_tokens"] == cap and sent["messages"] == request["messages"] and "stream" not in sent
    assert warmer._run is not None and warmer._run.warm_count == 1  # a hit: the next keep-alive is armed


def test_a_mid_chat_route_switch_makes_the_next_request_offload_first():
    """/model or a fallback moves the chat to a route that never saw its prefix: the cache is cold
    there, so the turn-start offload runs before the first request on the new route."""
    from agent.prompt_cache_clock import prompt_cache_is_cold

    agent = _agent("deepseek/deepseek-v4-flash")
    _respond(agent, {"model": agent.model, "messages": [{"role": "user", "content": "hi"}]})
    assert prompt_cache_is_cold(agent) is False
    agent.model = "z-ai/glm-5.1"
    assert prompt_cache_is_cold(agent) is True
