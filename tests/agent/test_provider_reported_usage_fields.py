"""Provider-reported usage fields (``ProviderProfile.usage_response_fields``) and hook validation.

A provider that reports a call's billed amount in its own top-level response field (beside ``usage``)
names that field on its profile; core copies it into ``CanonicalUsage.raw_usage`` on the chat
completions, Anthropic Messages and Responses wires, streamed or not, for main-loop and auxiliary
calls, so the profile's ``get_usage_cost`` can return the amount as ``actual``. Core validates what the
hook returns before using it. Providers that declare nothing see no change.
"""

import json
import logging
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

HOST = "https://api.billfake.test/v1"
FIELD = "billfake_usage"


@pytest.fixture
def bill_profile():
    import providers
    from agent import usage_pricing
    from agent.usage_pricing import CostResult
    from providers.base import ProviderProfile

    class BillFakeProfile(ProviderProfile):
        def get_usage_cost(self, model, usage):
            report = (usage.raw_usage or {}).get(FIELD)
            if isinstance(report, dict) and isinstance(report.get("cost"), (int, float)):
                amount = Decimal(str(report["cost"]))
                return CostResult(amount_usd=amount, status="actual", source="provider_cost_api", label=f"${amount}")
            return None

    providers.list_providers()  # discovery first, so the fake survives it
    profile = BillFakeProfile(name="billfake", base_url=HOST, hostnames=("billfake.test",),
                              usage_response_fields=(FIELD, "usage", "id"))
    providers.register_provider(profile)
    getattr(usage_pricing, "_COST_HOOK_WARNED", set()).clear()
    yield profile
    import hermes_cli.auth
    providers._REGISTRY.pop("billfake", None)
    providers._SOURCES.pop("billfake", None)
    providers._PROVIDER_LIST_CACHE = None
    hermes_cli.auth.PROVIDER_REGISTRY.pop("billfake", None)
    getattr(usage_pricing, "_COST_HOOK_WARNED", set()).clear()


def _agent(provider="billfake", base_url=HOST, model="fake-model", **kwargs):
    from run_agent import AIAgent

    with patch("agent.process_bootstrap.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        return AIAgent(api_key="k", base_url=base_url, provider=provider, model=model, quiet_mode=True,
                       skip_context_files=True, skip_memory=True, **kwargs)


def _chat_usage():
    return SimpleNamespace(prompt_tokens=100, completion_tokens=7, total_tokens=107,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=0), completion_tokens_details=None)


# ── field declaration, capture ──────────────────────────────────────────────────────────────


def test_declared_fields_reach_raw_usage_and_others_do_not(bill_profile):
    """Only declared, non-reserved names are copied; usage keys win; an undeclared provider is untouched."""
    from agent.usage_pricing import CanonicalUsage, provider_usage_response_fields, with_response_usage_fields

    assert provider_usage_response_fields("billfake") == (FIELD,)  # "usage"/"id" are response structure
    usage = CanonicalUsage(input_tokens=1, raw_usage={"prompt_tokens": 1})
    report = {"cost": 0.5}
    # attribute, dict key and pydantic model_extra all work
    for response in (SimpleNamespace(**{FIELD: report}), {FIELD: report},
                     SimpleNamespace(model_extra={FIELD: report})):
        assert with_response_usage_fields(usage, response, "billfake").raw_usage == {"prompt_tokens": 1, FIELD: report}
    inner = CanonicalUsage(raw_usage={FIELD: {"cost": 1}})
    assert with_response_usage_fields(inner, {FIELD: report}, "billfake").raw_usage == {FIELD: {"cost": 1}}
    for provider in ("openrouter", "", None):
        assert with_response_usage_fields(usage, {FIELD: report}, provider) is usage


@pytest.mark.parametrize("declared,expected", [
    ((" Billfake_Usage ", "BILLFAKE_USAGE"), ("billfake_usage",)),
    ("billfake_usage", ("billfake_usage",)),  # a bare string is one name, not its characters
    (("Usage", "ID", "usage ", "service_tier", "provider", "system_fingerprint", "tool_calls", "finish_reason",
      "cost", "cost_details", "cache_creation", "speed", "model_dump", "model_extra", "prompt_tokens_details",
      "__class__", "_private", "9lives", "has-dash", "", None, 3), ()),
])
def test_reserved_and_malformed_names_are_never_captured(bill_profile, monkeypatch, declared, expected):
    from agent.usage_pricing import provider_usage_response_fields

    monkeypatch.setattr(bill_profile, "usage_response_fields", declared)
    assert provider_usage_response_fields("billfake") == expected


def test_field_values_are_copied_as_bounded_plain_data():
    """Namespaces (Relay rewrites) and pydantic objects become plain data; mocks, opaque objects and
    oversized or over-deep values are dropped instead of being stored."""
    from agent.usage_pricing import response_usage_field
    from openai.types.chat import ChatCompletionChunk

    ns = SimpleNamespace(**{FIELD: SimpleNamespace(total=5, details=[SimpleNamespace(n=1)])})
    assert response_usage_field(ns, FIELD) == {"total": 5, "details": [{"n": 1}]}
    chunk = ChatCompletionChunk.model_validate({"id": "c", "object": "chat.completion.chunk", "created": 0,
                                                "model": "m", "choices": []})
    assert response_usage_field({FIELD: chunk}, FIELD)["id"] == "c"
    assert response_usage_field(MagicMock(), FIELD) is None  # a mock's auto-attribute is not data
    assert response_usage_field({FIELD: object()}, FIELD) is None
    deep: dict = {}
    for _ in range(12):
        deep = {"d": deep}
    assert response_usage_field({FIELD: deep}, FIELD) is None
    assert response_usage_field({FIELD: list(range(20_000))}, FIELD) is None
    assert response_usage_field({FIELD: {"ok": list(range(100))}}, FIELD) == {"ok": list(range(100))}


def test_capture_never_breaks_a_stream(bill_profile):
    from agent.usage_pricing import CanonicalUsage, StreamUsageFields, with_response_usage_fields

    class Hostile:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    fields = StreamUsageFields("billfake")
    fields.observe(Hostile())
    fields.observe({FIELD: {"cost": 1}})
    assert fields.captured == {FIELD: {"cost": 1}}
    usage = CanonicalUsage(input_tokens=1)
    assert with_response_usage_fields(usage, Hostile(), "billfake") is usage


def test_attach_never_shadows_what_the_response_class_defines():
    """A declared name that is an SDK method (``json``, ``copy``, ...) must not be overwritten on the
    response: only plain attributes the class doesn't define are attached."""
    from agent.usage_pricing import attach_usage_response_fields
    from openai.types.chat import ChatCompletion

    completion = ChatCompletion.model_validate({"id": "c", "object": "chat.completion", "created": 0,
                                                "model": "m", "choices": []})
    captured = {"json": {"cost": 1}, "copy": {"cost": 2}, FIELD: {"cost": 0.5}}
    attach_usage_response_fields(completion, captured)
    assert callable(completion.json) and callable(completion.copy)
    assert getattr(completion, FIELD) == {"cost": 0.5}
    plain = attach_usage_response_fields(SimpleNamespace(), captured)
    assert plain.json == {"cost": 1}  # a plain namespace has no such method, so the field lands


# ── hook validation ─────────────────────────────────────────────────────────────────────────


def _cost(status="actual", amount=Decimal("0.01")):
    from agent.usage_pricing import CostResult

    return CostResult(amount_usd=amount, status=status, source="provider_cost_api", label="x")


class _AlwaysFinite(Decimal):
    def is_finite(self):  # a hostile subclass that lies about being finite
        return True

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return True


@pytest.mark.parametrize("returned", [
    {"amount_usd": 1, "status": "actual"},  # a dict, not a CostResult
    "0.01",
    _cost(amount=Decimal("Infinity")),
    _cost(amount=Decimal("NaN")),
    _cost(amount=Decimal("-0.01")),
    _cost(amount=Decimal("1000.01")),  # over the per-call cap for a reported amount
    _cost(amount=Decimal(10) ** 300),
    _cost(status="estimated", amount=Decimal(10) ** 13),
    _cost(status="ACTUAL", amount=Decimal(5000)),  # not a CostStatus: can't dodge the actual cap
    _cost(status="billed", amount=Decimal("0.01")),
    _cost(status=None, amount=Decimal("0.01")),
    _cost(amount=None),  # actual without an amount
    _cost(amount=_AlwaysFinite("Infinity")),  # a Decimal subclass can't smuggle inf
    _cost(amount=float("nan")),
    _cost(amount=float("inf")),
    _cost(amount=-0.5),
    _cost(amount=True),
    _cost(amount="0.01"),
    _cost(amount=10**5000),  # its repr raises ValueError on Python 3.11 (int str-conversion digit limit)
    _cost(status=10**5000, amount=Decimal("0.01")),
], ids=["dict", "str", "inf", "nan", "negative", "over-cap", "huge", "huge-estimate", "upper-status",
        "unknown-status", "none-status", "actual-no-amount", "decimal-subclass-inf", "float-nan", "float-inf",
        "float-negative", "bool", "str-amount", "huge-int-amount", "huge-int-status"])
def test_unusable_hook_results_fall_back_to_core_pricing(bill_profile, monkeypatch, caplog, returned):
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    monkeypatch.setattr(type(bill_profile), "get_usage_cost", lambda self, model, usage: returned)
    with caplog.at_level(logging.WARNING, logger="agent.usage_pricing"):
        first = estimate_usage_cost("fake-model", CanonicalUsage(input_tokens=5), provider="billfake", base_url=HOST)
        second = estimate_usage_cost("fake-model", CanonicalUsage(input_tokens=5), provider="billfake", base_url=HOST)
    assert (first.status, first.amount_usd) == (second.status, second.amount_usd) == ("unknown", None)
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1  # once, then DEBUG


def test_in_range_hook_results_are_kept(bill_profile, monkeypatch):
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    for result in (_cost(amount=Decimal(1000)), _cost(amount=Decimal(0)), _cost(status="unknown", amount=None),
                   _cost(status="estimated", amount=Decimal(5000))):
        monkeypatch.setattr(type(bill_profile), "get_usage_cost", lambda self, model, usage, r=result: r)
        kept = estimate_usage_cost("fake-model", CanonicalUsage(input_tokens=5), provider="billfake")
        assert kept == result and kept is not result  # a fresh copy, equal field by field


def test_a_cost_result_subclass_cannot_change_its_amount_after_validation(bill_profile, monkeypatch):
    """The hook's result is copied into a fresh plain CostResult, so a getter that returns a valid amount
    while it is checked and inf afterwards can't reach the store."""
    from agent.usage_pricing import CanonicalUsage, CostResult, estimate_usage_cost

    class _Shifty(CostResult):
        reads = 0

        def __getattribute__(self, name):
            if name == "amount_usd":
                type(self).reads += 1
                return Decimal(5) if type(self).reads == 1 else Decimal("Infinity")
            return super().__getattribute__(name)

    shifty = _Shifty(amount_usd=Decimal(5), status="actual", source="provider_cost_api", label="x")
    monkeypatch.setattr(type(bill_profile), "get_usage_cost", lambda self, model, usage: shifty)
    result = estimate_usage_cost("fake-model", CanonicalUsage(input_tokens=5), provider="billfake")
    assert type(result) is CostResult
    assert (result.status, result.amount_usd, result.amount_usd) == ("actual", Decimal(5), Decimal(5))


@pytest.mark.parametrize("amount,expected", [
    (0.5, Decimal("0.5")), (0.1, Decimal("0.1")), (2, Decimal(2)), (0, Decimal(0)),
    (_AlwaysFinite("0.25"), Decimal("0.25")),
])
def test_numeric_hook_amounts_are_normalized_to_a_plain_decimal(bill_profile, monkeypatch, amount, expected):
    """A finite, non-negative int or float is converted via ``str`` (so 0.1 stays 0.1), and a Decimal
    subclass is copied to a plain Decimal; the result is otherwise the hook's own."""
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    monkeypatch.setattr(type(bill_profile), "get_usage_cost", lambda self, model, usage: _cost(amount=amount))
    result = estimate_usage_cost("fake-model", CanonicalUsage(input_tokens=5), provider="billfake")
    assert (result.status, result.source, result.amount_usd) == ("actual", "provider_cost_api", expected)
    assert type(result.amount_usd) is Decimal


def test_unusable_result_warning_is_bounded(bill_profile, monkeypatch, caplog):
    """The warning names the bad amount and status, but a huge value can't flood the log line."""
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    monkeypatch.setattr(type(bill_profile), "get_usage_cost",
                        lambda self, model, usage: _cost(status="x" * 50_000, amount="9" * 50_000))
    with caplog.at_level(logging.WARNING, logger="agent.usage_pricing"):
        estimate_usage_cost("fake-model", CanonicalUsage(input_tokens=5), provider="billfake")
    [record] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(record.getMessage()) < 400


def test_profile_reported_cost_is_actual_and_a_failing_hook_warns_once(bill_profile, monkeypatch, caplog):
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    reported = estimate_usage_cost("fake-model", CanonicalUsage(input_tokens=5, raw_usage={FIELD: {"cost": 0.0123}}),
                                   provider="billfake", base_url=HOST)
    assert (reported.status, reported.source, reported.amount_usd) == ("actual", "provider_cost_api", Decimal("0.0123"))

    def boom(self, model, usage):
        raise RuntimeError("plugin pricing bug")

    monkeypatch.setattr(type(bill_profile), "get_usage_cost", boom)
    with caplog.at_level(logging.DEBUG, logger="agent.usage_pricing"):
        for _ in range(3):
            fallback = estimate_usage_cost("fake-model", CanonicalUsage(input_tokens=5), provider="billfake", base_url=HOST)
            assert (fallback.status, fallback.amount_usd) == ("unknown", None)
    levels = [r.levelno for r in caplog.records if "get_usage_cost failed" in r.getMessage()]
    assert levels == [logging.WARNING, logging.DEBUG, logging.DEBUG]


def test_core_pricing_rejects_astronomic_token_counts():
    """A malformed usage (10**320 tokens) used to price to a Decimal that is inf as a float."""
    from agent.usage_pricing import estimate_usage_cost, normalize_usage

    usage = normalize_usage({"input_tokens": 10**320, "output_tokens": 1}, provider="anthropic",
                            api_mode="anthropic_messages")
    result = estimate_usage_cost("claude-sonnet-4-5", usage, provider="anthropic")
    assert (result.status, result.amount_usd) == ("unknown", None)


# ── main-loop wires, with real SDK objects ───────────────────────────────────────────────────


def _sdk_chunk(**extra):
    from openai.types.chat import ChatCompletionChunk

    return ChatCompletionChunk.model_validate({"id": "c", "object": "chat.completion.chunk", "created": 0,
                                               "model": "fake-model", **extra})


def _text_chunks(text="hi"):
    return [_sdk_chunk(choices=[{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]),
            _sdk_chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])]


def _usage_chunk(report=None):
    extra = {FIELD: report} if report is not None else {}
    return _sdk_chunk(choices=[], usage={"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107}, **extra)


def _stream_call(agent, attempts):
    """Run a streamed chat call; each attempt yields its chunks, then drops the connection when asked."""
    import httpx

    calls = {"n": 0}

    def create(**_kw):
        chunks, drop = attempts[calls["n"]]
        calls["n"] += 1

        def gen():
            yield from chunks
            if drop:
                raise httpx.RemoteProtocolError("peer closed connection")
        return gen()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    agent.api_mode = "chat_completions"
    with patch.object(agent, "_create_request_openai_client", return_value=client), \
            patch.object(agent, "_close_request_openai_client"), \
            patch("agent.chat_completion_helpers._wait_stream_retry_backoff"), \
            patch.dict("os.environ", {"HERMES_STREAM_RETRIES": "2"}):
        return agent._interruptible_streaming_api_call({}), calls["n"]


@pytest.mark.parametrize("provider,captured", [("billfake", True), ("openrouter", False)])
def test_chat_completions_stream_keeps_the_field_from_the_sdk_usage_chunk(bill_profile, provider, captured):
    """The SDK keeps the unknown field in the chunk's ``model_extra``; core carries it to the response."""
    agent = _agent(provider=provider, base_url=HOST if provider == "billfake" else "https://openrouter.ai/api/v1")
    response, _ = _stream_call(agent, [(_text_chunks() + [_usage_chunk({"cost": 0.5})], False)])
    assert response.choices[0].message.content == "hi"
    assert getattr(response, FIELD, None) == ({"cost": 0.5} if captured else None)


@pytest.mark.parametrize("retry_report", [None, {"cost": 0.25}], ids=["retry-without-field", "retry-other-value"])
def test_a_retried_stream_keeps_only_the_latest_attempts_field(bill_profile, retry_report):
    """Attempt 1 carries the field, then drops; the retry's own field (or none) wins, nothing leaks."""
    agent = _agent()
    response, attempts = _stream_call(agent, [
        ([_usage_chunk({"cost": 9.0})], True),  # field seen, then the connection drops before any text
        (_text_chunks() + [_usage_chunk(retry_report)], False),
    ])
    assert attempts == 2
    assert response.choices[0].message.content == "hi"
    assert getattr(response, FIELD, None) == retry_report


def test_relay_rewritten_chunks_keep_the_field(bill_profile):
    """Relay hands back rewritten chunks as nested SimpleNamespaces (``relay_llm._namespace``)."""
    from agent.relay_llm import _jsonable, _namespace

    chunks = [_namespace(_jsonable(c)) for c in _text_chunks() + [_usage_chunk({"cost": 0.5, "parts": [{"n": 1}]})]]
    response, _ = _stream_call(_agent(), [(chunks, False)])
    assert getattr(response, FIELD) == {"cost": 0.5, "parts": [{"n": 1}]}


def _anthropic_sse(report):
    delta = {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
             "usage": {"output_tokens": 2}, **({FIELD: report} if report else {})}
    events = [
        {"type": "message_start", "message": {"id": "msg", "type": "message", "role": "assistant", "content": [],
                                              "model": "claude-fake", "stop_reason": None, "stop_sequence": None,
                                              "usage": {"input_tokens": 10, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0}, delta, {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def _anthropic_client(report):
    import anthropic
    import httpx

    return anthropic.Anthropic(api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=_anthropic_sse(report), headers={"content-type": "text/event-stream"}))))


def test_anthropic_stream_keeps_the_field_from_the_sdk_message_delta(bill_profile):
    """Real SDK stream: the field rides on ``RawMessageDeltaEvent.model_extra`` and the SDK's final
    message snapshot drops it; core re-attaches it to the response."""
    client = _anthropic_client({"cost": 0.25})
    agent = _agent(model="claude-fake")
    agent.api_mode = "anthropic_messages"
    agent._anthropic_client = client
    agent._create_request_anthropic_client = lambda *a, **k: client
    try:
        response = agent._interruptible_streaming_api_call(
            {"model": "claude-fake", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})
    finally:
        client.close()
    assert response.content[0].text == "hi"
    assert getattr(response, FIELD) == {"cost": 0.25}


def _responses_sse(report):
    item = {"type": "message", "id": "m1", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": "hi", "annotations": []}]}
    completed = {"type": "response.completed", "sequence_number": 2, **({FIELD: report} if report else {}),
                 "response": {"id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
                              "model": "gpt-fake", "output": [], "parallel_tool_calls": False,
                              "tool_choice": "auto", "tools": [],
                              "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                                        "input_tokens_details": {"cached_tokens": 0},
                                        "output_tokens_details": {"reasoning_tokens": 0}}}}
    events = [{"type": "response.output_item.done", "output_index": 0, "sequence_number": 1, "item": item}, completed]
    return "".join("data: " + json.dumps(e) + "\n\n" for e in events)


def _sdk_openai(sse):
    import httpx
    import openai

    return openai.OpenAI(api_key="k", base_url=HOST, http_client=httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"}))))


def test_responses_stream_keeps_the_field_from_the_sdk_response_completed(bill_profile):
    from agent.codex_runtime import _consume_codex_event_stream

    client = _sdk_openai(_responses_sse({"cost": 0.75}))
    # The direct helper carries nothing without a collector...
    events = list(client.responses.create(model="gpt-fake", input="hi", stream=True))
    assert not hasattr(_consume_codex_event_stream(iter(events), model="m"), FIELD)
    # ...and run_codex_stream passes one for the provider.
    agent = _agent(model="gpt-fake")
    agent.api_mode = "codex_responses"
    agent.client = client
    response = agent._run_codex_stream({"model": "gpt-fake", "input": [{"role": "user", "content": "hi"}]})
    assert response.output_text == "hi" or response.output[0].content[0].text == "hi"
    assert getattr(response, FIELD) == {"cost": 0.75}


# ── auxiliary wires ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def aux_route():
    """Bind the auxiliary call context to the fake provider, as ``call_llm`` does for one call."""
    from agent.auxiliary_client import _RELAY_AUX_CALL_CONTEXT

    token = _RELAY_AUX_CALL_CONTEXT.set({"task": "title_generation", "provider": "billfake",
                                         "stream_provider": "billfake"})
    yield
    _RELAY_AUX_CALL_CONTEXT.reset(token)


def test_aux_streamed_chat_aggregation_keeps_the_field(bill_profile, aux_route):
    from agent.auxiliary_client import _aggregate_chat_stream

    response = _aggregate_chat_stream(iter(_text_chunks() + [_usage_chunk({"cost": 0.5})]), model="fake-model")
    assert response.choices[0].message.content == "hi"
    assert getattr(response, FIELD) == {"cost": 0.5}


def test_aux_responses_adapter_keeps_the_field(bill_profile, aux_route):
    from agent.auxiliary_client import _CodexCompletionsAdapter

    response = _CodexCompletionsAdapter(_sdk_openai(_responses_sse({"cost": 0.75})), "gpt-fake").create(
        messages=[{"role": "user", "content": "hi"}])
    assert response.choices[0].message.content == "hi"
    assert getattr(response, FIELD) == {"cost": 0.75}


def test_aux_anthropic_adapter_keeps_the_field(bill_profile, aux_route):
    from agent.auxiliary_client import _AnthropicCompletionsAdapter

    client = _anthropic_client({"cost": 0.25})
    try:
        response = _AnthropicCompletionsAdapter(client, "claude-fake").create(
            messages=[{"role": "user", "content": "hi"}], max_tokens=16)
    finally:
        client.close()
    assert response.choices[0].message.content == "hi"
    assert getattr(response, FIELD) == {"cost": 0.25}


def test_aux_adapters_carry_nothing_for_a_provider_that_declares_nothing(bill_profile):
    from agent.auxiliary_client import _RELAY_AUX_CALL_CONTEXT, _aggregate_chat_stream

    token = _RELAY_AUX_CALL_CONTEXT.set({"task": "t", "provider": "openrouter", "stream_provider": "openrouter"})
    try:
        response = _aggregate_chat_stream(iter(_text_chunks() + [_usage_chunk({"cost": 0.5})]), model="m")
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(token)
    assert not hasattr(response, FIELD)


# ── accounting: the reported amount reaches the turn and the aux recorder ─────────────────────
# Which stored column holds it is the persistence layer's business; these only check the amount
# and status, so they hold whichever column an ``actual`` amount is written to.


@pytest.fixture
def db(tmp_path):
    from hermes_state import SessionDB

    store = SessionDB(tmp_path / "state.db")
    yield store
    store.close()


def _stored_spend(row):
    return (row["estimated_cost_usd"] or 0.0) + (row["actual_cost_usd"] or 0.0)


def test_a_turn_records_the_reported_amount_as_actual(bill_profile, db):
    from agent import turn_usage

    db.create_session("t", source="cli", model="fake-model")
    agent = _agent(session_id="t", platform="cli", save_trajectories=False, session_db=db)
    try:
        agent._session_db_created = True
        agent.client = SimpleNamespace()  # a MagicMock client would pass for a MoA client
        turn_usage.record_response_usage(
            agent, SimpleNamespace(usage=_chat_usage(), id=None, model="fake-model", **{FIELD: {"cost": 0.5}}),
            messages=[{"role": "user", "content": "hi"}], api_call_count=1, api_duration=0.1,
            compression_attempts=0, max_compression_attempts=3)
        assert (agent.session_cost_status, agent.session_cost_source) == ("actual", "provider_cost_api")
        assert agent.session_estimated_cost_usd == pytest.approx(0.5)
        db.flush_token_counts()
    finally:
        agent.close()
    row = db.get_session("t")
    assert row["cost_status"] == "actual" and _stored_spend(row) == pytest.approx(0.5)


def test_an_aux_call_records_the_reported_amount_as_actual(bill_profile, db):
    from agent.aux_accounting import record_aux_usage, reset_accounting_context, set_accounting_context

    db.create_session("s", source="cli")
    token = set_accounting_context(db, "s")
    try:
        record_aux_usage(SimpleNamespace(model="fake-model", usage=_chat_usage(), **{FIELD: {"cost": 0.5}}),
                         "title_generation", provider="billfake", base_url=HOST)
    finally:
        reset_accounting_context(token)
    with db._read_ctx() as conn:
        row = dict(conn.execute("SELECT estimated_cost_usd, actual_cost_usd, cost_status FROM session_model_usage"
                                " WHERE session_id = 's' AND task = 'title_generation'").fetchone())
    assert _stored_spend(row) == pytest.approx(0.5)
