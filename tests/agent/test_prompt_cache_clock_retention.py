"""The clock's expiry estimate follows what the request asked for and where it went.

A request that asks ``prompt_cache_retention: 24h`` keeps 30 minutes as its warming guarantee but is
not treated as gone at 31 minutes; a local or private-network endpoint has no time-based expiry at all.
Both must stay out of the clock-driven cold prune, while a route switch still offloads.
"""

from types import SimpleNamespace

import pytest

from agent.prompt_cache_clock import cache_certainly_warm, cache_window, record_attempt_sent, record_cache_response
from agent.transports.codex import _default_prompt_cache_retention_for_request
from agent.prompt_cache_clock import prompt_cache_is_cold
from agent.usage_pricing import normalize_usage

T0 = 1_000.0
_USAGE = {"prompt_tokens": 5000, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4096}}


def _agent(base_url, model):
    return SimpleNamespace(
        provider="custom", base_url=base_url, model=model, api_mode="chat_completions", session_id="s1",
        _cache_ttl=None, _use_prompt_caching=False, _session_db=None,
        context_compressor=SimpleNamespace(compression_count=0),
    )


def _turn(agent, monkeypatch, *, retention=None, sent_at=T0):
    import agent.prompt_cache_clock as clock

    monkeypatch.setattr(clock.time, "time", lambda: sent_at)
    record_attempt_sent(agent, {"messages": [], **({"prompt_cache_retention": retention} if retention else {})})
    usage = normalize_usage(_USAGE, provider=agent.provider, api_mode=agent.api_mode)
    record_cache_response(agent, usage, model=agent.model, provider=agent.provider, base_url=agent.base_url,
                          api_mode=agent.api_mode)


def _retention(base_url, model):
    return _default_prompt_cache_retention_for_request(model, base_url)


@pytest.mark.parametrize(("base_url", "model", "after_s", "cold"), [
    # Asked for 24h: guarantee (warming) stays 30m, but the entry is not gone at 35 min, even after a read.
    ("https://api.meta.ai/v1", "llama-5", 35 * 60, False),
    ("https://bedrock-mantle.us-east-1.api.aws/v1", "openai.gpt-5.5", 35 * 60, False),
    # Local / private endpoints evict on memory pressure, never on a clock.
    ("http://127.0.0.1:8080/v1", "qwen3", 2 * 3600, False),
    ("http://localhost:11434/v1", "qwen3", 2 * 3600, False),
    ("http://192.168.1.20:8000/v1", "qwen3", 2 * 3600, False),
    ("http://10.0.0.5:8000/v1", "qwen3", 2 * 3600, False),
    ("http://[::1]:8000/v1", "qwen3", 2 * 3600, False),
    ("http://gpu-box.local:8000/v1", "qwen3", 2 * 3600, False),
    # Controls: a public OpenAI-spec route without retention is cold at 31 min; chatgpt.com Codex gets no
    # retention from Hermes, so its 30m stands.
    ("https://api.deepseek.com/v1", "deepseek-chat", 31 * 60, True),
    ("https://chatgpt.com/backend-api/codex", "gpt-5.5", 31 * 60, True),
])
def test_expiry_estimate_follows_requested_retention_and_host(base_url, model, after_s, cold, monkeypatch):
    agent = _agent(base_url, model)
    retention = _retention(base_url, model)
    _turn(agent, monkeypatch, retention=retention)
    _turn(agent, monkeypatch, retention=retention, sent_at=T0 + 60)  # a read renews; the estimate survives it
    window = cache_window(agent)
    assert window.guaranteed_until == T0 + 60 + 1800  # the warming deadline is 30m on every route
    assert cache_certainly_warm(window, now=T0 + 60 + 1799)
    assert prompt_cache_is_cold(agent, now=T0 + 60 + after_s) is cold


def test_route_switch_still_offloads_from_a_local_endpoint(monkeypatch):
    agent = _agent("http://127.0.0.1:8080/v1", "qwen3")
    _turn(agent, monkeypatch)
    assert not prompt_cache_is_cold(agent, now=T0 + 10 * 3600)
    agent.model = "llama-4"  # /model or a fallback: the new route never saw this prefix
    assert cache_window(agent).reason == "route_changed"
    assert prompt_cache_is_cold(agent, now=T0 + 60)
