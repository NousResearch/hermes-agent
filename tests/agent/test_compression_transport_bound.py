"""CORE3: the summary transport bound holds BEFORE any fallback route can receive a prompt.

The auxiliary recovery ladder re-sends an already-built prompt to a task fallback or to the main
agent model after a capacity error. A window advertised by the pinned route therefore never
licenses a prompt larger than the route-independent ceiling: these checks drive the REAL
``call_llm`` ladder with local fake clients (no network, synthetic text only).
"""
import contextlib
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as ac
from agent.context_compressor import ContextCompressor, pin_summary_route

BOUND = ContextCompressor._SUMMARY_INPUT_MAX_CHARS
SNAPSHOT = "## Historical Task Snapshot\nNone. This session contains no user-authored turns.\n\n## Goal\n"
MAIN_URL = "http://127.0.0.1:1/v1"
MAIN_ROUTE = {"provider": "custom", "model": "main/model", "base_url": MAIN_URL}


@pytest.fixture(autouse=True)
def _fresh_unhealthy_cache():
    ac._reset_aux_unhealthy_cache()
    yield
    ac._reset_aux_unhealthy_cache()


def _ok(text):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = "stop"
    return response


def _payment_402():
    exc = Exception("Error code: 402 - {'error': {'message': 'Insufficient credits'}}")
    exc.status_code = 402
    return exc


def _client(base_url, create, api_key=None):
    client = MagicMock()
    client.base_url = base_url
    client.chat.completions.create = create
    if api_key is not None:
        client.api_key = api_key
    return client


def _main_compressor():
    return ContextCompressor(model="main/model", provider="custom", base_url=MAIN_URL,
                             config_context_length=1_000_000, quiet_mode=True)


def test_long_window_pin_never_hands_an_oversized_prompt_to_the_real_fallback_ladder():
    primary_prompts, fallback_prompts = [], []

    def primary_create(**kwargs):
        primary_prompts.append(kwargs["messages"][0]["content"])
        raise _payment_402()

    def fallback_create(**kwargs):
        fallback_prompts.append(kwargs["messages"][0]["content"])
        return _ok(SNAPSHOT + f"part {len(fallback_prompts)}")

    primary = _client(MAIN_URL, MagicMock(side_effect=primary_create))
    fallback = _client("http://127.0.0.1:2/v1", MagicMock(side_effect=fallback_create))
    source = "S0:" + "x" * 190_000 + ":end"
    with patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("custom", "main/model", MAIN_URL, None, None)), \
         patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "main/model")), \
         patch("agent.auxiliary_client._try_configured_fallback_chain",
               side_effect=lambda *a, **k: (fallback, "fallback/model", "fallback_chain[0](custom:backup)")), \
         patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")), \
         pin_summary_route(dict(MAIN_ROUTE)):
        _main_compressor()._generate_summary([{"role": "tool", "tool_call_id": "t", "content": source}])
    assert fallback_prompts, "the real ladder fallback must be exercised"
    assert max(map(len, primary_prompts + fallback_prompts)) <= BOUND


def test_pinned_route_carries_every_bounded_chunk_in_order():
    compressor = _main_compressor()
    calls = []

    def answer(**kwargs):
        calls.append(kwargs)
        kwargs["route_info"].update(provider="custom", model="main/model")
        return _ok(SNAPSHOT + f"part {len(calls)}")

    source = "".join(f"record {i:05d} " + "y" * 4_000 + "\n" for i in range(100))
    with patch("agent.context_compressor.call_llm", side_effect=answer), \
         pin_summary_route(dict(MAIN_ROUTE)):
        assert compressor._generate_summary([{"role": "tool", "tool_call_id": "t", "content": source}])
    assert len(calls) >= 3
    assert all(c.get("provider") == "custom" and c.get("model") == "main/model" for c in calls)
    prompts = [c["messages"][0]["content"] for c in calls]
    assert max(map(len, prompts)) <= BOUND
    positions = [next((chunk, p.find(f"record {i:05d} ")) for chunk, p in enumerate(prompts)
                      if f"record {i:05d} " in p) for i in range(100)]
    assert positions == sorted(positions), "every record reaches a summary call in source order"


def test_pinned_route_is_dropped_after_its_failure_moved_the_generation_to_the_main_model():
    compressor = ContextCompressor(model="main/model", provider="custom", base_url=MAIN_URL,
                                   config_context_length=1_000_000, quiet_mode=True)
    calls = []

    def answer(**kwargs):
        calls.append(kwargs.get("model"))
        if kwargs.get("model") == "pinned/other" and len(calls) == 1:
            kwargs["route_info"].update(provider="custom", model="pinned/other")
            raise RuntimeError("synthetic pinned-route failure")
        kwargs["route_info"].update(provider="custom", model="main/model")
        return _ok(SNAPSHOT + f"part {len(calls)}")

    source = "z" * 300_000
    with patch("agent.context_compressor.call_llm", side_effect=answer), \
         pin_summary_route({"provider": "custom", "model": "pinned/other", "base_url": MAIN_URL}):
        assert compressor._generate_summary([{"role": "tool", "tool_call_id": "t", "content": source}])
    assert calls[0] == "pinned/other"
    assert "pinned/other" not in calls[1:], calls


def test_oversized_prompt_is_refused_before_dispatch():
    compressor = _main_compressor()
    with patch("agent.context_compressor.call_llm") as call:
        result = compressor._generate_summary_once(
            [{"role": "tool", "tool_call_id": "t", "content": "w"}],
            serialized_chunk="w" * (BOUND + 1),
        )
    assert result is None
    call.assert_not_called()
    assert compressor._last_summary_truncated_failure is True


# --- CORE3B: a successful rescue INSIDE call_llm's ladder moves the remaining chunks ---------------
PIN_URL = "http://127.0.0.1:3/v1"
BACKUP_URL = "http://127.0.0.1:4/v1"
PINNED = {"label": "fallback_chain[0](custom)", "provider": "custom", "model": "pinned/model",
          "base_url": PIN_URL, "api_key": None, "api_mode": None, "timeout": None}
BACKUP = {"label": "fallback_chain[1](custom)", "provider": "custom", "model": "backup/model",
          "base_url": BACKUP_URL, "api_key": None, "api_mode": None, "timeout": None}


def _two_chunk_source():
    source = "".join(f"line {i:06d} " + "q" * 90 + "\n" for i in range(2_600))
    assert BOUND < len(source) < 2 * BOUND - 20_000
    return [{"role": "tool", "tool_call_id": "t", "content": source}]


def _echo_explicit_route(task=None, provider=None, model=None, base_url=None, api_key=None, *_a, **_k):
    return (provider or "custom", model, base_url, api_key, None)


def _run_real_ladder(*, chain_rescue, main_rescue):
    """Drive the REAL call_llm ladder: the pinned backend always answers 402, the rescue answers."""
    pinned_prompts, served = [], []

    def pinned_create(**kwargs):
        pinned_prompts.append(kwargs["messages"][0]["content"])
        raise _payment_402()

    def rescue_create(**kwargs):
        served.append(kwargs["messages"][0]["content"])
        rescue_requests.append(kwargs)
        return _ok(SNAPSHOT + f"part {len(served)}")

    rescue_requests = []
    clients = {PIN_URL: _client(PIN_URL, MagicMock(side_effect=pinned_create), "synthetic-pin-key"),
               BACKUP_URL: _client(BACKUP_URL, MagicMock(side_effect=rescue_create), "synthetic-backup-key"),
               MAIN_URL: _client(MAIN_URL, MagicMock(side_effect=rescue_create), "synthetic-main-key")}
    dispatched = []
    real_call_llm = ac.call_llm

    def spy(**kwargs):
        dispatched.append({k: kwargs.get(k) for k in ("provider", "model", "base_url", "api_key", "timeout")})
        return real_call_llm(**kwargs)

    def cached(provider, model=None, *_a, base_url=None, **_k):
        return clients[str(base_url)], model

    chain = (clients[BACKUP_URL], "backup/model", "fallback_chain[1](custom)") if chain_rescue else (None, None, "")
    main = (clients[MAIN_URL], "main/model", "main-agent(custom)") if main_rescue else (None, None, "")
    compressor = _main_compressor()
    with patch("agent.context_compressor.call_llm", side_effect=spy), \
         patch("agent.auxiliary_client._resolve_task_provider_model", side_effect=_echo_explicit_route), \
         patch("agent.auxiliary_client._get_cached_client", side_effect=cached), \
         patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=chain), \
         patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=main), \
         pin_summary_route(dict(PINNED)):
        result = compressor._generate_summary(_two_chunk_source())
    for call in dispatched:   # the credential and timeout ride along; compare the route fields first
        call["sent_timeout"] = rescue_requests[0].get("timeout") if rescue_requests else None
    return result, dispatched, pinned_prompts, served


def _route(call):
    return {k: call[k] for k in ("provider", "model", "base_url")}


def test_ladder_rescue_moves_the_remaining_chunks_to_the_served_chain_route():
    result, dispatched, pinned_prompts, served = _run_real_ladder(chain_rescue=True, main_rescue=False)
    assert result
    assert len(dispatched) >= 2
    assert _route(dispatched[0]) == {"provider": "custom", "model": "pinned/model", "base_url": PIN_URL}
    # The first dispatch of every later chunk no longer carries the route that already failed, and
    # carries the credential and timeout of the client and request that answered.
    assert all(_route(d) == {"provider": "custom", "model": "backup/model", "base_url": BACKUP_URL}
               for d in dispatched[1:]), dispatched
    assert all(d["api_key"] == "synthetic-backup-key" for d in dispatched[1:])
    assert all(d["timeout"] is not None and d["timeout"] == d["sent_timeout"] for d in dispatched[1:])
    assert len(pinned_prompts) == 1
    assert len(served) == len(dispatched)
    assert max(map(len, pinned_prompts + served)) <= BOUND


def test_ladder_rescue_on_the_main_agent_moves_the_remaining_chunks_to_the_main_runtime():
    result, dispatched, pinned_prompts, served = _run_real_ladder(chain_rescue=False, main_rescue=True)
    assert result
    assert len(dispatched) >= 2
    assert all(_route(d) == {"provider": "custom", "model": "main/model", "base_url": MAIN_URL}
               for d in dispatched[1:]), dispatched
    assert all(d["api_key"] == "synthetic-main-key" for d in dispatched[1:])
    assert len(pinned_prompts) == 1
    assert max(map(len, pinned_prompts + served)) <= BOUND


def test_unobserved_ladder_hop_fails_closed_without_adoption():
    # The ladder left the prepared route, but no dispatch was observed answering: never guess one.
    compressor = _main_compressor()
    compressor._previous_summary = "Prior synthetic summary."
    calls = []

    def answer(**kwargs):
        calls.append(kwargs)
        route = kwargs["route_info"]
        route["provider"], route["model"] = "custom", "pinned/model"     # primary route prepared
        route["provider"], route["model"] = "custom", "unknown/model"    # ladder lane that served
        return _ok(SNAPSHOT + "part")

    with patch("agent.context_compressor.call_llm", side_effect=answer), pin_summary_route(dict(PINNED)):
        result = compressor._generate_summary(_two_chunk_source())
    assert result is None
    assert len(calls) == 1, "no later chunk may go back to the failed route or to a guessed one"
    assert compressor._previous_summary == "Prior synthetic summary."
    # A lost route is a generation failure, not an incomplete (truncated) summary.
    assert compressor._last_summary_truncated_failure is False
    assert "no observed route" in compressor._last_summary_error


def test_route_name_reported_without_a_ladder_hop_keeps_the_pin():
    compressor = _main_compressor()
    calls = []

    def answer(**kwargs):
        calls.append({k: kwargs.get(k) for k in ("provider", "model", "base_url")})
        kwargs["route_info"].update(provider="custom", model="pinned-model-normalized")
        return _ok(SNAPSHOT + f"part {len(calls)}")

    with patch("agent.context_compressor.call_llm", side_effect=answer), pin_summary_route(dict(PINNED)):
        assert compressor._generate_summary(_two_chunk_source())
    assert len(calls) >= 2
    assert all(c == {"provider": "custom", "model": "pinned/model", "base_url": PIN_URL} for c in calls)


# --- CORE3D: the remaining chunks follow the dispatch that ANSWERED, field for field (reviews N1/N2, B1/B2) -
# Host and credential come from the one client that answered; model and timeout from the request it was
# sent. Nothing is matched against configuration, so no route can be guessed or paired with another's
# credential. The fake call_llm below scripts what the real seams record (they are exercised on the real
# ladder above and in test_auxiliary_served_dispatch.py). Synthetic placeholder credentials only.
def _answering_client(base_url, api_key):
    client = MagicMock()
    client.base_url, client.api_key = base_url, api_key
    return client


def _three_chunk_source():
    source = "".join(f"line {i:06d} " + "q" * 90 + "\n" for i in range(4_000))
    assert 2 * BOUND < len(source) < 3 * BOUND - 60_000
    return [{"role": "tool", "tool_call_id": "t", "content": source}]


def _scripted_plan(script, source=None, pinned=True):
    """Fake call_llm: call i hops off the prepared route when ``script[i][0]`` and is answered by the
    dispatch ``script[i][1]`` (client, request, provider, api_mode), recorded as the real seams do."""
    from agent.auxiliary_relay_metadata import _note_served_dispatch

    compressor = _main_compressor()
    compressor._previous_summary = "Prior synthetic summary."
    calls = []

    def answer(**kwargs):
        calls.append(kwargs)
        response = _ok(SNAPSHOT + f"part {len(calls)}")
        hop, dispatch = script[min(len(calls), len(script)) - 1]
        kwargs["route_info"].update(provider="custom", model="prepared/model")
        if hop:
            kwargs["route_info"].update(provider="custom", model="lane/model")
        if dispatch is not None:
            _note_served_dispatch(response, *dispatch)
        return response

    route = pin_summary_route(dict(PINNED)) if pinned else contextlib.nullcontext()
    with patch("agent.context_compressor.call_llm", side_effect=answer), route:
        result = compressor._generate_summary(source or _two_chunk_source())
    return compressor, calls, result


def _route_of(call):
    return tuple(call.get(k) for k in ("provider", "model", "base_url", "api_key", "api_mode", "timeout"))


LANE_B = (_answering_client(BACKUP_URL, "synthetic-placeholder-b"),
          {"model": "backup/model", "timeout": 45.0}, "fallback_chain[1](custom)", "chat_completions")


def test_served_lane_carries_the_answering_clients_host_and_credential():
    _compressor, calls, result = _scripted_plan([(True, LANE_B), (False, None)])
    assert result
    assert len(calls) == 2
    assert _route_of(calls[1]) == ("custom", "backup/model", BACKUP_URL, "synthetic-placeholder-b",
                                   "chat_completions", 45.0)


def test_a_same_pair_rescue_on_another_host_follows_the_host_that_served():
    # Same provider and model as the failed primary, other host and key: never back to the failed host.
    lane = (_answering_client(BACKUP_URL, "synthetic-placeholder-b"), {"model": "pinned/model"}, "custom", None)
    _compressor, calls, result = _scripted_plan([(True, lane), (False, None)])
    assert result
    assert (calls[1]["base_url"], calls[1]["api_key"]) == (BACKUP_URL, "synthetic-placeholder-b")


@pytest.mark.parametrize("pinned", [True, False], ids=["pinned", "unpinned"])
def test_a_same_route_recovery_under_a_pin_moves_the_remaining_chunks_to_the_fresh_credential(pinned):
    # Review B1: chunk 2 answers on the lane's host after a credential refresh (no hop, no new record);
    # chunk 3 must carry the fresh credential, never the one that just failed.
    refreshed = (_answering_client(BACKUP_URL, "synthetic-placeholder-k3"),
                 {"model": "backup/model", "timeout": 45.0}, "custom", "chat_completions")
    _compressor, calls, result = _scripted_plan([(True, LANE_B), (False, refreshed), (False, None)],
                                               source=_three_chunk_source(), pinned=pinned)
    assert result
    assert len(calls) == 3
    assert calls[1]["api_key"] == "synthetic-placeholder-b"
    assert _route_of(calls[2]) == ("custom", "backup/model", BACKUP_URL, "synthetic-placeholder-k3",
                                   "chat_completions", 45.0)


def test_a_pinned_chunk_answered_without_a_complete_observation_keeps_the_pin():
    _compressor, calls, result = _scripted_plan([(False, None), (False, None)])
    assert result
    assert all(_route_of(call)[:3] == ("custom", "pinned/model", PIN_URL) for call in calls)


@pytest.mark.parametrize("missing", ["api_key", "provider"])
def test_a_hop_whose_dispatch_lacks_a_credential_or_provider_fails_closed(missing):
    # Review B2: never let call_llm pair the answering host with a credential or client of its choosing.
    client = _answering_client(BACKUP_URL, None if missing == "api_key" else "synthetic-placeholder-b")
    client._hermes_aux_effective_provider = ""
    lane = (client, {"model": "backup/model"}, "auto" if missing == "provider" else "custom", None)
    compressor, calls, result = _scripted_plan([(True, lane)])
    assert result is None
    assert len(calls) == 1
    assert compressor._previous_summary == "Prior synthetic summary."
    assert compressor._last_summary_truncated_failure is False


# --- CORE3D: a cancellation that lands after the last answer is never adopted (review CORE3C-01) ------------
def test_cancellation_after_the_last_chunk_answered_is_never_adopted():
    from agent.auxiliary_client import AuxiliaryExplicitCancellation

    compressor = _main_compressor()
    compressor._previous_summary = "Prior synthetic summary."
    cancelled = []
    compressor._compression_cancelled_check = lambda: bool(cancelled)   # host-installed hook
    real_once = compressor._generate_summary_once

    def answer_then_cancel(*args, **kwargs):
        result = real_once(*args, **kwargs)
        cancelled.append(True)   # the host cancels between the last answer and the adoption
        return result

    turns = [{"role": "assistant", "content": "Synthetic assistant note about a fictional public record."}]
    with patch.object(compressor, "_generate_summary_once", side_effect=answer_then_cancel), \
         patch("agent.context_compressor.call_llm", return_value=_ok(SNAPSHOT + "complete")) as call:
        with pytest.raises(AuxiliaryExplicitCancellation):
            compressor._generate_summary(turns)
    assert call.call_count == 1
    assert compressor._previous_summary == "Prior synthetic summary."


# --- CORE3D: re-planning for a fallback destination transmits exactly the checked prompt (review 02 limit) --
def _texts(messages):
    """One string per message: its text, whether plain or split into content parts."""
    texts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text") or "" for part in content if isinstance(part, dict))
        texts.append(content if isinstance(content, str) else "")
    return texts


@pytest.mark.parametrize("provider,model,base_url,api_mode", [
    ("anthropic", "claude-sonnet-4-5", "https://api.anthropic.com", "anthropic_messages"),
    ("openrouter", "anthropic/claude-sonnet-4.5", "https://openrouter.ai/api/v1", "chat_completions"),
    ("custom", "backup/model", BACKUP_URL, None),
])
def test_fallback_replanning_transmits_exactly_the_checked_prompt(provider, model, base_url, api_mode):
    import copy

    messages = [{"role": "system", "content": "synthetic instructions " * 80},
                {"role": "user", "content": "public filler " * ((BOUND - 2_000) // 14)}]
    assert BOUND - 2_000 < sum(map(len, _texts(messages))) <= BOUND
    before = copy.deepcopy(messages)
    kwargs = ac._fallback_request_kwargs(
        ac._FallbackDestination(provider, base_url, api_mode, model), task="compression", messages=messages,
        tools=None, temperature=None, max_tokens=1_000, effective_timeout=30.0, effective_extra_body={},
        reasoning_config=None, fallback_entry={"provider": provider, "model": model}, task_config={},
        apply_fast_lane=False)
    assert messages == before, "the checked prompt must not be mutated"
    sent = kwargs["messages"]
    assert [m.get("role") for m in sent] == ["system", "user"]
    assert _texts(sent) == _texts(before), "re-planning may add cache markers, never text"


# --- CORE3C: the micro summary prompt is bounded before ANY dispatch (review 02, finding P1-a) ------------
# Micro-compaction is off by default. Its prompt is one whole exchange (every tool result, each under the
# per-record ceiling) plus the rolling summary, with no segmentation: it must be refused before call_llm,
# so no route, primary or fallback, ever receives more than the transport ceiling.
def _micro_compressor(**overrides):
    options: dict = {"model": "test/model", "config_context_length": 1_000_000, "quiet_mode": True}
    options.update(overrides)
    compressor = ContextCompressor(**options)
    compressor._micro_compact_enabled = True
    return compressor


def _wide_exchange(results):
    rows: list = [{"role": "user", "content": "Read these fictional public records."}]
    for i in range(results):
        rows.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"t{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]})
        rows.append({"role": "tool", "tool_call_id": f"t{i}",
                     "content": f"record {i} head " + "public filler " * 2_000 + f" record {i} tail"})
    return rows


def _prompt_chars(invoked):
    return sum(len(m.get("content") or "") for m in invoked.kwargs["messages"])


def test_micro_exchange_over_the_ceiling_is_refused_before_dispatch():
    compressor = _micro_compressor()
    exchange_text = compressor._serialize_for_summary(_wide_exchange(30))
    assert len(exchange_text) > BOUND
    with patch("agent.auxiliary_client.call_llm", return_value=_ok("Synthetic micro summary.")) as call:
        assert compressor._micro_summarize_one(exchange_text) is None
    call.assert_not_called()


def test_micro_exchange_plus_rolling_summary_over_the_ceiling_is_refused_before_dispatch():
    compressor = _micro_compressor()
    compressor._micro_compact_rolling_summary = "Rolling synthetic summary line.\n" * 3_000
    exchange_text = compressor._serialize_for_summary(_wide_exchange(20))
    assert len(exchange_text) < BOUND < len(exchange_text) + len(compressor._micro_compact_rolling_summary)
    with patch("agent.auxiliary_client.call_llm", return_value=_ok("Synthetic micro summary.")) as call:
        assert compressor._micro_summarize_one(exchange_text) is None
    call.assert_not_called()


def test_micro_prompt_under_the_ceiling_is_still_dispatched_once():
    compressor = _micro_compressor()
    compressor._micro_compact_rolling_summary = "Prior synthetic summary."
    exchange_text = compressor._serialize_for_summary(_wide_exchange(2))
    with patch("agent.auxiliary_client.call_llm", return_value=_ok("Synthetic micro summary.")) as call:
        assert compressor._micro_summarize_one(exchange_text) == "Synthetic micro summary."
    assert call.call_count == 1
    assert _prompt_chars(call.call_args_list[0]) <= BOUND


def test_micro_defrag_of_an_oversized_rolling_summary_is_refused_before_dispatch():
    compressor = _micro_compressor()
    rolling = "Rolling synthetic summary line.\n" * 6_000
    assert len(rolling) > BOUND
    compressor._micro_compact_rolling_summary = rolling
    with patch("agent.auxiliary_client.call_llm", return_value=_ok("Synthetic micro summary.")) as call:
        assert compressor._defrag_rolling_summary([]) is False
    call.assert_not_called()
    assert compressor._micro_compact_rolling_summary == rolling


def test_micro_pass_refused_over_the_ceiling_leaves_the_transcript_and_counts_a_failure():
    # Same window and protections as tests/agent/test_micro_compaction.py; the later turns are long enough
    # that the protected tail starts after the wide exchange, so the pass reaches the summary step.
    compressor = _micro_compressor(config_context_length=40960, provider="test", threshold_percent=0.75,
                                   protect_first_n=1, protect_last_n=2)
    messages = [{"role": "system", "content": "system prompt"}, *_wide_exchange(30),
                {"role": "assistant", "content": "Read all records."}]
    for i in range(3):
        messages += [{"role": "user", "content": f"question {i}"},
                     {"role": "assistant", "content": f"answer {i} " + "z" * 40_000}]
    attempts = []
    real_summarize_one = compressor._micro_summarize_one

    def counted(exchange_text):
        attempts.append(len(exchange_text))
        return real_summarize_one(exchange_text)

    compressor._micro_summarize_one = counted
    with patch("agent.auxiliary_client.call_llm", return_value=_ok("Synthetic micro summary.")) as call:
        result = compressor._micro_compact(list(messages))
    assert len(attempts) == 1 and attempts[0] > BOUND, "the pass must reach the micro summary step"
    call.assert_not_called()
    assert result == messages
    assert compressor._micro_compact_consecutive_failures == 1
    assert compressor._micro_compact_rolling_summary == ""
