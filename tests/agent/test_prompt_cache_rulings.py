"""Teknium's cache-spec rulings on the prompt-cache lifecycle, one invariant each.

1. The cache spec follows the API spec: a request with ``cache_control`` markers is Claude rules on any
   wire or gateway (the 5m/1h it asks for is declared and trusted, a route ceiling still clamps); a
   request with none is OpenAI rules (30 minutes, renewed by a read).
2. The first keep-alive that misses ends warming for the rest of the session: a new user message does
   not re-arm it, a compaction does not either, a model/provider switch (a different cache) does.
3. While a tool call runs the cache is kept warm until it returns, past the idle budget of three,
   bounded only by the cost of the one rewrite those keep-alives prevent.

Timers never start: ``_fire`` runs the armed callback inline, so no test sleeps."""

from decimal import Decimal
from types import SimpleNamespace

import pytest

import agent.prompt_cache_lifecycle as pcl
from agent.usage_pricing import get_pricing_entry as _REAL_PRICING

_OPUS = SimpleNamespace(input_cost_per_million=5, output_cost_per_million=25,
                        cache_read_cost_per_million=0.5, cache_write_cost_per_million=10)


class _Usage:
    def __init__(self, inp, out, read, write):
        self.input_tokens, self.output_tokens = inp, out
        self.cache_read_input_tokens, self.cache_creation_input_tokens = read, write


class _Client:
    """Anthropic-shaped double that sends the cap it is given and answers a hit (or a miss)."""

    def __init__(self, read=119_997, write=0):
        self.calls, self.read, self.write = [], read, write
        self.messages = SimpleNamespace(create=self._create)

    @staticmethod
    def effective_output_cap(*, max_tokens=None, **_):
        return max_tokens

    def with_options(self, **kw):
        return self

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(usage=_Usage(3, 1, self.read, self.write), content=[], stop_reason="max_tokens")


class _InertTimer:
    def __init__(self, delay, fn, args=()):
        self.delay, self.fn, self.args, self.daemon, self.name = delay, fn, args, True, ""

    def start(self):
        pass

    def cancel(self):
        pass


@pytest.fixture(autouse=True)
def _setup(monkeypatch, tmp_path):
    from agent.usage_pricing import PricingEntry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT", "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(key, raising=False)
    entry = PricingEntry(**{k: Decimal(str(v)) for k, v in vars(_OPUS).items()}, source="official_docs_snapshot")
    monkeypatch.setattr("agent.usage_pricing.get_pricing_entry", lambda *a, **k: entry)
    monkeypatch.setattr(pcl.PromptCacheWarmer, "_pricing", lambda self: _OPUS)
    monkeypatch.setattr(pcl.threading, "Timer", _InertTimer)


def _chat(client=None, **over):
    agent = SimpleNamespace(
        api_mode="anthropic_messages", _use_prompt_caching=True, _cache_disabled=False, _cache_ttl="1h",
        _interrupt_requested=False, _anthropic_client=client or _Client(), model="claude-opus-4-7",
        provider="anthropic", base_url="https://api.anthropic.com", api_key="", session_api_calls=7,
        session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0, session_input_tokens=0,
        session_output_tokens=0, session_cache_read_tokens=0, session_cache_write_tokens=0,
        session_estimated_cost_usd=0.0, _session_db=None, session_id="s", log_prefix="", platform="cli",
        compression_enabled=True, _user_turn_count=1, _executing_tools=False,
        context_compressor=SimpleNamespace(
            compression_count=0, last_prompt_tokens=150_000, threshold_tokens=200_000, summary_target_ratio=0.2,
            last_compression_rough_tokens=0, awaiting_real_usage_after_compression=False,
            get_active_compression_failure_cooldown=lambda: None),
    )
    for key, value in over.items():
        setattr(agent, key, value)
    return agent


_MARK = {"type": "ephemeral", "ttl": "1h"}
_REQUEST = {"model": "claude-opus-4-7", "max_tokens": 8192, "stream": True,
            "system": [{"type": "text", "text": "sys", "cache_control": _MARK}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": _MARK}]}]}
_WRITE = SimpleNamespace(usage=_Usage(50, 30, 0, 119_950))


def _respond(agent, request=_REQUEST, response=_WRITE):
    """One real response of the turn loop: the next request cancels warming, its reply is folded into
    the clock, then the controller arms on it."""
    from agent.prompt_cache_clock import attempt_facts, record_cache_response
    from agent.usage_pricing import normalize_usage

    pcl.cancel_prompt_cache_warming(agent, "new request in flight")
    agent.session_api_calls += 1
    agent._prompt_cache_attempt = attempt_facts(request)
    record_cache_response(agent, normalize_usage(response.usage, provider=agent.provider, api_mode=agent.api_mode),
                          model=agent.model, provider=agent.provider, base_url=agent.base_url, api_mode=agent.api_mode)
    return pcl.get_prompt_cache_warmer(agent).start(request, response)


def _fire(warmer):
    run = warmer._run
    assert run is not None and run.timer is not None, warmer.status
    warmer._refresh(run)


@pytest.mark.parametrize("route", [
    {"provider": "custom", "base_url": "https://gw.example/v1", "api_mode": "chat_completions"},
    {"provider": "minimax", "base_url": "https://api.minimax.io/anthropic", "api_mode": "anthropic_messages"},
])
def test_ruling_1_a_marked_request_is_claude_rules_on_any_wire_and_an_unmarked_one_is_openai_rules(route):
    from agent.prompt_cache_clock import attempt_facts, cache_window, record_cache_response
    from agent.usage_pricing import CanonicalUsage

    agent = _chat(**route)
    agent._prompt_cache_attempt = dict(attempt_facts(_REQUEST), sent_at=1_000.0)
    event = record_cache_response(agent, CanonicalUsage(input_tokens=50, cache_write_tokens=119_950),
                                  model=agent.model, provider=agent.provider, base_url=agent.base_url,
                                  api_mode=agent.api_mode)
    window = cache_window(agent)
    assert (event.route_class, window.tier, window.confidence) == ("anthropic_spec", "1h", "declared")
    assert window.guaranteed_until == 4_600.0  # the 1h it asked for is trusted: warm/compact/offload act on it

    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text(
        f"prompt_caching:\n  route_ttl_ceilings:\n    {route['provider']}: 5m\n", encoding="utf-8")
    event = record_cache_response(agent, CanonicalUsage(input_tokens=50, cache_write_tokens=119_950),
                                  model=agent.model, provider=agent.provider, base_url=agent.base_url,
                                  api_mode=agent.api_mode)
    assert (event.tier, event.reason) == ("5m", "route_ceiling_5m")  # a declared ceiling still clamps

    unmarked = _chat(provider="nous", base_url="https://inference-api.nousresearch.com/v1",
                     api_mode="chat_completions", model="qwen/qwen3.7-max")
    unmarked._prompt_cache_attempt = {"sent_at": 1_000.0, "marker_ttl": None}
    event = record_cache_response(unmarked, CanonicalUsage(input_tokens=50, cache_read_tokens=40_000),
                                  model=unmarked.model, provider=unmarked.provider, base_url=unmarked.base_url,
                                  api_mode=unmarked.api_mode)
    assert (event.route_class, event.tier, event.guaranteed_until) == ("openai_spec", "30m", 2_800.0)


def test_ruling_2_the_first_miss_ends_warming_for_the_rest_of_the_session():
    client = _Client(read=0, write=119_950)  # the entry was gone: the keep-alive re-wrote it
    agent = _chat(client=client)
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent) is None
    pcl.end_turn_prompt_cache_warming(agent)
    _fire(warmer)
    assert len(client.calls) == 1 and warmer._run is None
    assert pcl.compaction_due(agent, now=warmer._spent.due_at) == (True, "due")  # the fallthrough still runs

    agent._user_turn_count = 2  # the person sends another message, then goes idle again
    _respond(agent)
    pcl.end_turn_prompt_cache_warming(agent)
    assert warmer._run is None, "a new user message re-armed warming after a miss"
    agent.context_compressor.compression_count = 1  # a compaction keeps the session, and its miss
    agent._user_turn_count = 3
    _respond(agent)
    assert warmer._run is None and len(client.calls) == 1
    assert warmer.status["reason"] == "prompt cache warming disabled for session: keep-alive missed"
    assert pcl.prompt_cache_warming_disabled(agent)
    assert pcl.compaction_due(agent, now=warmer._spent.due_at)[1] != "budget_not_spent"  # steps 2/3 still decide

    agent.model = "claude-sonnet-4-6"  # /model: a different cache, warmed afresh
    agent._user_turn_count = 4
    assert _respond(agent, {**_REQUEST, "model": "claude-sonnet-4-6"}) is None
    assert not pcl.prompt_cache_warming_disabled(agent)


def test_ruling_3_a_running_tool_is_kept_warm_until_it_returns_bounded_by_one_rewrite():
    client = _Client()
    agent = _chat(client=client)
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent) is None
    agent._executing_tools = True  # the model asked for a long tool call; it is running
    while warmer._run is not None and len(client.calls) < 100:
        _fire(warmer)
    run_decision = warmer._last_decision
    sent = len(client.calls)
    assert sent > pcl.WARM_BUDGET, "a running tool keeps warming past the idle budget"
    per = run_decision.warm_cost
    assert sent * per <= run_decision.miss_cost < (sent + 1) * per  # stopped at one rewrite's cost
    assert warmer._budget.used == 0  # the idle budget was not touched

    agent._executing_tools = False  # the tool returned; the model answered; the turn ended
    assert _respond(agent) is None
    pcl.end_turn_prompt_cache_warming(agent)
    idle = 0
    while warmer._run is not None and idle < 10:
        _fire(warmer)
        idle += 1
    assert len(client.calls) - sent == pcl.WARM_BUDGET  # idle: three per user message, as before


# ── Nous catalog rule: input_cache_write in /v1/models = explicit markers (Anthropic spec) ──────────

_NOUS = "https://inference-api.nousresearch.com/v1"
_CATALOG = {  # per-token strings, as the Nous /v1/models pricing parses
    "qwen/qwen3.7-max": {"pricing": {"prompt": "0.000001475", "completion": "0.000004425",
                                     "cache_read": "0.000000295", "cache_write": "0.0000018437"}},
    "anthropic/claude-sonnet-4.6": {"pricing": {"prompt": "0.000003", "completion": "0.000015",
                                                "cache_read": "0.0000003", "cache_write": "0.00000375",
                                                "cache_write_1h": "0.000006"}},
    "deepseek/deepseek-v4-flash": {"pricing": {"prompt": "0.00000009", "completion": "0.0000003",
                                               "cache_read": "0.000000018"}},
    "qwen/qwen3.5-397b-a17b": {"pricing": {"prompt": "0.00000045", "completion": "0.0000036",
                                           "cache_read": "0.00000022"}},
}


@pytest.fixture
def nous_catalog(monkeypatch):
    import agent.model_metadata as mm

    monkeypatch.setattr(mm, "fetch_endpoint_model_metadata", lambda base_url, api_key="", **_: dict(_CATALOG))


def _policy(model):
    from agent.agent_runtime_helpers import anthropic_prompt_cache_policy

    agent = SimpleNamespace(provider="nous", base_url=_NOUS, api_mode="chat_completions", model=model,
                            _cache_disabled=False, api_key="")
    return anthropic_prompt_cache_policy(agent)


def test_nous_rule_1_a_catalog_cache_write_rate_means_markers_at_the_tier_asked(nous_catalog):
    """Qwen 3.7 Max is priced with ``input_cache_write`` on Nous: it gets markers, and its 1h ask is
    kept and trusted (measured: a 1h entry still read back after 13 idle minutes)."""
    from agent.prompt_cache_clock import attempt_facts, record_cache_response
    from agent.prompt_caching import effective_cache_ttl
    from agent.usage_pricing import CanonicalUsage

    assert _policy("qwen/qwen3.7-max") == (True, False)
    assert effective_cache_ttl("1h", provider="nous", base_url=_NOUS, model="qwen/qwen3.7-max", auto=True) == "1h"
    agent = _chat(provider="nous", base_url=_NOUS, api_mode="chat_completions", model="qwen/qwen3.7-max")
    agent._prompt_cache_attempt = dict(attempt_facts(_REQUEST), sent_at=1_000.0)
    event = record_cache_response(agent, CanonicalUsage(input_tokens=14, cache_write_tokens=31_297),
                                  model=agent.model, provider="nous", base_url=_NOUS, api_mode="chat_completions")
    assert (event.route_class, event.tier, event.confidence) == ("anthropic_spec", "1h", "declared")


def test_nous_rule_2_no_catalog_cache_write_rate_means_no_markers_and_openai_spec(nous_catalog):
    """Qwen 3.5 397B has no cache-write rate on Nous: no markers (it was marked as 'a Qwen'), its
    automatic cache follows the OpenAI spec; Claude keeps its markers because the catalog prices them."""
    assert _policy("qwen/qwen3.5-397b-a17b") == (False, False)
    assert _policy("deepseek/deepseek-v4-flash") == (False, False)
    assert _policy("anthropic/claude-sonnet-4.6") == (True, False)


def test_nous_rule_3_a_1h_write_is_priced_at_the_catalog_rate_not_a_fixed_2x(nous_catalog, monkeypatch):
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    monkeypatch.setattr("agent.usage_pricing.get_pricing_entry", _REAL_PRICING)  # served from the seeded catalog
    monkeypatch.setattr("agent.usage_pricing.fetch_endpoint_model_metadata",
                        lambda base_url, api_key="", **_: dict(_CATALOG))
    one_hour = CanonicalUsage(cache_write_tokens=1_000_000, cache_write_by_ttl=(("1h", 1_000_000),))
    qwen = estimate_usage_cost("qwen/qwen3.7-max", one_hour, provider="nous", base_url=_NOUS).amount_usd
    claude = estimate_usage_cost("anthropic/claude-sonnet-4.6", one_hour, provider="nous", base_url=_NOUS).amount_usd
    assert (round(qwen, 4), round(claude, 4)) == (Decimal("1.8437"), Decimal("6.0000"))  # 1.25x vs Claude's 2x
