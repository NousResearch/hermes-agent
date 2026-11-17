"""The keep-alive budget belongs to the USER turn, not to one armed run, and every replay is genuinely
bounded on the client that sends it.

- A turn that spends its three idle keep-alives (no tool executing; ruling 3 leaves a running tool's
  keep-alives uncounted) still reaches step 2 (compact) or step 3 (offload): the
  ended budget is carried to the prompt the turn ends on and to the owner's release.
- A keep-alive on the wire when the next tool round replaces its run is still booked against that user
  turn: never more than three paid keep-alives per user message.
- A client that would raise the replay's output cap (native Gemini with thinking, a long-decode budget)
  is never warmed.

Timers never start: the tests fire the armed callback inline, and the mid-flight replacement is held
on a barrier, so nothing sleeps."""

import threading
from decimal import Decimal
from types import SimpleNamespace

import pytest

import agent.prompt_cache_lifecycle as pcl
from agent.usage_pricing import PricingEntry

_OPUS = SimpleNamespace(input_cost_per_million=5, output_cost_per_million=25,
                        cache_read_cost_per_million=0.5, cache_write_cost_per_million=10)


class _Usage:
    def __init__(self, inp, out, read, write):
        self.input_tokens, self.output_tokens = inp, out
        self.cache_read_input_tokens, self.cache_creation_input_tokens = read, write


class _Client:
    """Anthropic-shaped double that sends the cap it is given; ``hold`` blocks a reply until released."""

    def __init__(self):
        self.calls, self.hold, self.entered, self.release = [], False, threading.Event(), threading.Event()
        self.messages = SimpleNamespace(create=self._create)

    @staticmethod
    def effective_output_cap(*, max_tokens=None, **_):
        return max_tokens

    def with_options(self, **kw):
        return self

    def _create(self, **kw):
        self.calls.append(kw)
        if self.hold:
            self.entered.set()
            assert self.release.wait(10)
        return SimpleNamespace(usage=_Usage(3, 1, 119_997, 0), content=[], stop_reason="max_tokens")


class _InertTimer:
    def __init__(self, delay, fn, args=()):
        self.delay, self.fn, self.args, self.daemon, self.name = delay, fn, args, True, ""

    def start(self):
        pass

    def cancel(self):
        pass


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    from agent.usage_pricing import PricingEntry

    for key in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT", "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(key, raising=False)
    entry = PricingEntry(**{k: Decimal(str(v)) for k, v in vars(_OPUS).items()}, source="official_docs_snapshot")
    monkeypatch.setattr("agent.usage_pricing.get_pricing_entry", lambda *a, **k: entry)
    monkeypatch.setattr(pcl.PromptCacheWarmer, "_pricing", lambda self: _OPUS)
    monkeypatch.setattr(pcl.warmer.threading, "Timer", _InertTimer)


def _chat(client=None, tokens=150_000, **over):
    agent = SimpleNamespace(
        api_mode="anthropic_messages", _use_prompt_caching=True, _cache_disabled=False, _cache_ttl="1h",
        _interrupt_requested=False, _anthropic_client=client or _Client(), model="claude-opus-4-7",
        provider="anthropic", base_url="https://api.anthropic.com", api_key="", session_api_calls=7,
        session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0, session_input_tokens=0,
        session_output_tokens=0, session_cache_read_tokens=0, session_cache_write_tokens=0, session_reasoning_tokens=0,
        session_estimated_cost_usd=0.0, _session_db=None, session_id="s", log_prefix="", platform="cli",
        compression_enabled=True, _user_turn_count=1,
        context_compressor=SimpleNamespace(
            compression_count=0, last_prompt_tokens=tokens, threshold_tokens=200_000, summary_target_ratio=0.2,
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
_WRITE_1H = SimpleNamespace(usage=SimpleNamespace(
    input_tokens=50, output_tokens=30, cache_read_input_tokens=0, cache_creation_input_tokens=119_950,
    cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=119_950)))


def _respond(agent, request=_REQUEST, response=_WRITE_1H):
    """One real response of the turn loop: the request about to go out cancels warming, its reply is
    folded into the clock, then the controller arms on it."""
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


def test_a_tool_run_that_spent_its_keep_alives_still_compacts_at_the_end_of_the_turn(monkeypatch):
    """Ruling A: the budget belongs to the user turn. A long tool run uses the three keep-alives, the
    next tool round's reply moves the prompt on, and the turn ends: step 2 still decides on that prompt,
    at idle and at release, instead of reading ``budget_not_spent`` forever."""
    monkeypatch.setattr(pcl.warmer.time, "time", lambda: 1000.0)
    agent = _chat()
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent) is None
    for _ in range(pcl.WARM_BUDGET):  # three idle keep-alives (no tool executing), then the budget ends
        if warmer._run is not None:
            _fire(warmer)
    assert len(agent._anthropic_client.calls) == pcl.WARM_BUDGET
    monkeypatch.setattr(pcl.warmer.time, "time", lambda: 2000.0)
    assert _respond(agent) == "warm budget spent this turn"  # the tool's result went out; the model answered
    assert warmer._spent is not None and warmer._spent.due_at == 5000.0
    pcl.end_turn_prompt_cache_warming(agent)

    due_at = warmer._spent.due_at
    assert pcl.get_prompt_cache_warmer(agent).edge_decision(now=due_at).compact == "due"
    assert pcl.get_prompt_cache_warmer(agent).edge_decision(releasing=True, now=due_at).compact == "due"
    assert pcl.get_prompt_cache_warmer(agent).claim_edge(now=due_at) == "compact"
    assert len(agent._anthropic_client.calls) == pcl.WARM_BUDGET  # nothing more was sent

    small = _chat(tokens=50_000)  # the same run on a small chat ends in step 3 instead
    w2 = pcl.get_prompt_cache_warmer(small)
    _respond(small)
    for _ in range(pcl.WARM_BUDGET):
        _fire(w2)
    _respond(small)
    pcl.end_turn_prompt_cache_warming(small)
    assert pcl.get_prompt_cache_warmer(small).edge_decision(now=small._prompt_cache_clock.estimated_until + 1).offload == "due"


def test_a_keep_alive_on_the_wire_when_its_run_is_replaced_still_counts_against_the_user_turn():
    """The tool round finishes while a keep-alive's reply is outstanding and the next request replaces
    the run: the in-flight keep-alive was reserved at admission, so the user turn still pays for at most
    three keep-alives, and its billed cost lands in that turn's budget."""
    client = _Client()
    agent = _chat(client=client)
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent) is None
    _fire(warmer)
    _fire(warmer)  # two keep-alives sent and folded
    client.hold = True
    stale = warmer._run
    sender = threading.Thread(target=warmer._refresh, args=(stale,))
    sender.start()
    assert client.entered.wait(10)  # the third keep-alive is on the wire...
    client.hold = False
    assert _respond(agent) is not None  # ...when the tool result goes out and its reply replaces the run
    client.release.set()
    sender.join(10)
    assert not sender.is_alive()

    for _ in range(3):  # the next tool rounds of the same user turn
        if warmer._run is not None:
            _fire(warmer)
        _respond(agent)
    assert len(client.calls) == pcl.WARM_BUDGET
    budget = warmer._budget
    assert budget.used == pcl.WARM_BUDGET and budget.spent_usd == pytest.approx(
        pcl.WARM_BUDGET * budget.last_warm_usd, rel=0, abs=1e-9)


def test_a_miss_that_lands_after_its_run_was_replaced_still_ends_the_budget_and_decides_step_two():
    """The first keep-alive is on the wire when the next tool round's reply arms a replacement run, and
    it comes back a miss. The user turn's budget is over: the replacement stops (nothing more is sent)
    and step 2 decides on the prompt the turn now holds, with no further real reply needed."""
    client = _Client()
    agent = _chat(client=client)
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent) is None
    client.hold, stale = True, warmer._run
    client_create = client._create

    def _miss(**kw):
        response = client_create(**kw)
        response.usage = _Usage(3, 1, 0, 119_950)  # the entry was gone: re-written, not read
        return response

    client.messages = SimpleNamespace(create=_miss)
    sender = threading.Thread(target=warmer._refresh, args=(stale,))
    sender.start()
    assert client.entered.wait(10)
    client.hold = False
    assert _respond(agent) is None  # the tool round's reply arms the replacement
    replacement = warmer._run
    client.release.set()
    sender.join(10)
    assert not sender.is_alive()

    pcl.end_turn_prompt_cache_warming(agent)
    assert warmer._run is None and replacement.cancelled and len(client.calls) == 1
    assert pcl.get_prompt_cache_warmer(agent).edge_decision(now=warmer._spent.due_at).compact == "due"


def _gemini_client(sent):
    """The real native Gemini client on a local transport that records each ``generateContent`` body."""
    import json

    import httpx

    from agent.gemini_native_adapter import GeminiNativeClient

    def _answer(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": "."}]}, "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"promptTokenCount": 40_000, "candidatesTokenCount": 1, "totalTokenCount": 40_001,
                              "cachedContentTokenCount": 39_900}})

    return GeminiNativeClient(api_key="local-stub", base_url="https://generativelanguage.googleapis.com/v1beta",
                              http_client=httpx.Client(transport=httpx.MockTransport(_answer)))


@pytest.mark.parametrize("extra_body", [
    {"thinking_config": {"thinkingLevel": "high", "includeThoughts": True}},
    {"thinking_config": {"thinkingBudget": 8192}},
])
def test_a_native_gemini_thinking_request_is_bounded_or_not_warmed(extra_body):
    """Native Gemini lifts any cap to 65,535 output tokens when thinking is on, so a ``max_tokens=1``
    replay would generate a full reasoning answer: such a chat is not warmed. With thinking off the
    same client sends the cap it is given, and the keep-alive goes out with ``maxOutputTokens=1``."""
    sent = []
    client = _gemini_client(sent)
    agent = _chat(api_mode="chat_completions", provider="gemini", model="gemini-3.7-pro",
                  base_url="https://generativelanguage.googleapis.com/v1beta", _anthropic_client=None)
    agent.client = client
    request = {"model": "gemini-3.7-pro", "max_tokens": 8192, "stream": True, "extra_body": extra_body,
               "messages": [{"role": "system", "content": "sys " * 9000}, {"role": "user", "content": "hi"}]}
    usage = SimpleNamespace(prompt_tokens=40_000, completion_tokens=20,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=39_000))
    assert client.effective_output_cap(max_tokens=1, extra_body=extra_body) == 65_535
    assert _respond(agent, request, SimpleNamespace(usage=usage)) == "the client cannot bound the replay's output cap"
    assert sent == []

    request["extra_body"] = {"thinking_config": {"thinkingBudget": 0}}
    assert _respond(agent, request, SimpleNamespace(usage=usage)) is None
    _fire(pcl.get_prompt_cache_warmer(agent))
    (body,) = sent
    assert body["generationConfig"]["maxOutputTokens"] == 1


def test_a_long_decode_budget_is_dropped_from_the_replay_and_unknown_clients_are_never_warmed():
    """Gemini 4's ``extra_body.max_output_tokens`` (up to 1,048,576) would override the replay's cap:
    the bounded replay drops it, so with thinking off the native client's real cap stays 1. A client
    that cannot report its cap (an ACP subprocess, an unknown plugin client) is not warmed."""
    from agent.gemini_native_adapter import GeminiNativeClient

    request = {"model": "gemini-4-pro", "max_tokens": 8192, "messages": [],
               "extra_body": {"thinking_config": {"thinkingBudget": 0}, "max_output_tokens": 1_048_576}}
    replay = pcl.bounded_chat_replay(request, "gemini-4-pro")
    assert replay["max_tokens"] == 1 and "max_output_tokens" not in replay["extra_body"]
    assert GeminiNativeClient.effective_output_cap(**replay) == 1
    assert pcl.replay_cap_honoured(SimpleNamespace(chat=None), request, "gemini-4-pro", "chat_completions") is False


@pytest.mark.parametrize("tail, expected_calls", [(470, 0), (0, 2)])
def test_reasoning_replays_reserve_the_full_output_cap(monkeypatch, tail, expected_calls):
    import json

    import httpx
    from openai import OpenAI

    entry = PricingEntry(input_cost_per_million=Decimal(10), output_cost_per_million=Decimal(40),
                         cache_read_cost_per_million=Decimal("2.5"), source="official_docs_snapshot")
    monkeypatch.setattr("agent.usage_pricing.get_pricing_entry", lambda *a, **kw: entry)
    monkeypatch.setattr(pcl.PromptCacheWarmer, "_pricing", lambda self: entry)
    sent = []

    def transport(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "warm", "object": "chat.completion", "created": 0, "model": "o3",
            "choices": [{"index": 0, "finish_reason": "length",
                         "message": {"role": "assistant", "content": ""}}],
            "usage": {"prompt_tokens": 1024 + tail, "completion_tokens": 16,
                      "prompt_tokens_details": {"cached_tokens": 1024}},
        })

    with OpenAI(api_key="fixture", http_client=httpx.Client(transport=httpx.MockTransport(transport))) as client:
        agent = _chat(api_mode="chat_completions", provider="openai", model="o3",
                      base_url="https://api.openai.com/v1")
        agent.client = client
        request = {"model": "o3", "max_completion_tokens": 4096,
                   "messages": [{"role": "user", "content": "hi"}]}
        usage = SimpleNamespace(prompt_tokens=1024 + tail, completion_tokens=8,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=1024))
        _respond(agent, request, SimpleNamespace(usage=usage))
        warmer = pcl.get_prompt_cache_warmer(agent)
        for _ in range(pcl.WARM_BUDGET):
            if warmer._run is not None:
                _fire(warmer)
        assert len(sent) == expected_calls
        assert all(body["max_completion_tokens"] == 16 for body in sent)
        assert warmer._budget is not None
        assert warmer._budget.spent_usd <= 0.00768  # 1024 * ($10 - $2.50) / 1M


@pytest.mark.parametrize("change", ["disabled", "epoch"])
def test_keep_alive_admission_respects_disabled_cache_and_reset_epoch(change):
    agent = _chat()
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent) is None
    if change == "disabled":
        agent._cache_disabled = True
        assert _respond(agent) == "prompt caching disabled"
    else:
        agent._session_reset_epoch = 1  # the request counter and session id alone can be reused
        assert warmer.status["state"] == "inactive"
        _fire(warmer)
    assert warmer._run is None and agent._anthropic_client.calls == []


# Greptile on d42bea43: session-owned state follows the session; the money limit ends a budget like its
# count; a per-request fee is reserved before the send.
def test_a_new_session_does_not_inherit_the_old_chats_spent_budget():
    agent = _chat()
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent) is None
    warmer._budget.ended = True  # turn 1 spent its keep-alives
    run = warmer._run
    pcl.reset_prompt_cache_session(agent)
    agent._session_reset_epoch = 1
    warmer._disable_warming(run)  # a miss of the old session lands after the reset
    assert not warmer.warming_disabled
    assert _respond(agent) is None  # the new chat's turn 1 warms afresh
    assert warmer._run is not None


def test_an_idle_budget_out_of_money_still_reaches_the_compaction_decision(monkeypatch):
    agent = _chat()
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent) is None
    pcl.end_turn_prompt_cache_warming(agent)
    warmer._budget.spent_usd = 10.0  # the keep-alives so far cost about one rewrite
    _fire(warmer)
    assert warmer._run is None and agent._anthropic_client.calls == []
    assert warmer._spent is not None
    assert warmer.edge_decision(now=warmer._spent.due_at).compact != "budget_not_spent"


@pytest.mark.parametrize("fee, warms", [(Decimal(0), True), (Decimal(5), False)])
def test_a_per_request_fee_is_reserved_before_the_send(fee, warms):
    pricing = SimpleNamespace(**vars(_OPUS), request_cost=fee)
    decision = pcl.evaluate_economics(pricing, 120_000, 3)
    assert (decision.action == "warm") is warms


def test_a_keep_alive_miss_survives_the_agent_rebuild_and_the_compaction_child_session():
    """Greptile on d42bea43: the gateway evicts and rebuilds the agent after a cache-edge compaction, and the
    compaction moves the conversation to a child session id; neither re-arms warming after a miss."""
    agent = _chat(session_id="miss-parent")
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent) is None
    warmer._disable_warming(warmer._run)
    rebuilt = _chat(session_id="miss-parent")
    assert pcl.get_prompt_cache_warmer(rebuilt).warming_disabled
    pcl.carry_missed_route("miss-parent", "miss-child")
    assert pcl.get_prompt_cache_warmer(_chat(session_id="miss-child")).warming_disabled
    assert not pcl.get_prompt_cache_warmer(_chat(session_id="another-chat")).warming_disabled
