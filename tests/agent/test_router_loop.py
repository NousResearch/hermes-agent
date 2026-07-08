"""Router runtime tests: verdict parsing, fail-open, sticky-per-turn routing,
fallback walking, accounting protocol, stream passthrough."""

from types import SimpleNamespace

import pytest

import agent.router_loop as router_loop
from agent.router_loop import (
    RouterChatCompletions,
    RouterClient,
    _build_classifier_messages,
    _parse_verdict,
)


PRESET = {
    "enabled": True,
    "classifier": {"provider": "openai-codex", "model": "gpt-5.5"},
    "classifier_max_tokens": 16,
    "classifier_context_messages": 4,
    "default_route": "simple",
    "routes": {
        "simple": {"provider": "lmstudio", "model": "google/gemma-4-e4b"},
        "complex": {"provider": "openai-codex", "model": "gpt-5.5"},
    },
    "fallbacks": [
        {"provider": "lmstudio", "model": "qwen/qwen3-4b-thinking-2507"},
        {"provider": "openai-codex", "model": "gpt-5.5"},
    ],
    "channel_hints": {"whatsapp": "simple"},
    "short_circuit_chars": 0,
}


def _response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))],
        usage=None,
    )


@pytest.fixture
def patched(monkeypatch):
    """Patch config + runtime resolution + call_llm; returns the call log."""
    calls = []

    def fake_call_llm(*, task, messages, **kwargs):
        calls.append({"task": task, "messages": messages, **kwargs})
        if task == "router_classifier":
            return _response(fake_call_llm.verdict)
        return _response(f"acted by {kwargs.get('model')}")

    fake_call_llm.verdict = "simple"

    monkeypatch.setattr(router_loop, "call_llm", fake_call_llm)
    monkeypatch.setattr(router_loop, "_slot_runtime", lambda slot: dict(slot))
    monkeypatch.setattr(router_loop, "_maybe_apply_moa_cache_control", lambda m, r: m)
    monkeypatch.setattr(
        RouterChatCompletions, "_maybe_preload_lmstudio", lambda self, slot, runtime: None
    )

    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: {"router": {"presets": {"default": PRESET}}})
    return fake_call_llm, calls


def test_parse_verdict_lenient():
    assert _parse_verdict("simple", "simple") == "simple"
    assert _parse_verdict(" COMPLEX.\n", "simple") == "complex"
    # "complex" wins when both words appear.
    assert _parse_verdict("simple, no wait — complex", "simple") == "complex"
    assert _parse_verdict("banana", "simple") == "simple"
    assert _parse_verdict("", "complex") == "complex"
    # Substrings must not match ("simplexity" contains neither exact word).
    assert _parse_verdict("simplexity", "complex") == "complex"


def test_routes_simple_to_local(patched):
    fake, calls = patched
    facade = RouterChatCompletions("default")
    resp = facade.create(messages=[{"role": "user", "content": "hoi"}])
    assert "gemma" in resp.choices[0].message.content
    tasks = [c["task"] for c in calls]
    assert tasks == ["router_classifier", "router_acting"]
    assert facade.last_aggregator_slot["model"] == "google/gemma-4-e4b"


def test_routes_complex_to_codex(patched):
    fake, calls = patched
    fake.verdict = "complex"
    facade = RouterChatCompletions("default")
    resp = facade.create(messages=[{"role": "user", "content": "refactor my app"}])
    assert "gpt-5.5" in resp.choices[0].message.content
    assert facade.last_aggregator_slot["model"] == "gpt-5.5"


def test_classifier_failure_fails_open_to_default_route(patched, monkeypatch):
    fake, calls = patched

    def exploding_call_llm(*, task, messages, **kwargs):
        if task == "router_classifier":
            raise ConnectionError("classifier down")
        calls.append({"task": task, **kwargs})
        return _response(f"acted by {kwargs.get('model')}")

    monkeypatch.setattr(router_loop, "call_llm", exploding_call_llm)
    facade = RouterChatCompletions("default")
    resp = facade.create(messages=[{"role": "user", "content": "hoi"}])
    # default_route=simple → gemma, despite the dead classifier.
    assert "gemma" in resp.choices[0].message.content


def test_sticky_per_user_turn_no_reclassification(patched):
    fake, calls = patched
    facade = RouterChatCompletions("default")
    turn = [{"role": "user", "content": "hoi"}]
    facade.create(messages=turn)
    classifier_calls = [c for c in calls if c["task"] == "router_classifier"]
    assert len(classifier_calls) == 1

    # Tool-loop iteration 2: same user prefix, grown tail → cache HIT.
    grown = turn + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "content": "result", "tool_call_id": "t1"},
    ]
    facade.create(messages=grown)
    classifier_calls = [c for c in calls if c["task"] == "router_classifier"]
    assert len(classifier_calls) == 1  # unchanged — no reclassification

    # New user turn → cache MISS → classifier runs again.
    new_turn = grown + [
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "thanks, another thing"},
    ]
    facade.create(messages=new_turn)
    classifier_calls = [c for c in calls if c["task"] == "router_classifier"]
    assert len(classifier_calls) == 2


def test_fallback_walk_order_and_stickiness(patched, monkeypatch):
    fake, calls = patched
    failed_models = {"google/gemma-4-e4b"}
    acting_calls = []

    def flaky_call_llm(*, task, messages, **kwargs):
        if task == "router_classifier":
            return _response("simple")
        acting_calls.append(kwargs.get("model"))
        if kwargs.get("model") in failed_models:
            raise ConnectionError("lmstudio down")
        return _response(f"acted by {kwargs.get('model')}")

    monkeypatch.setattr(router_loop, "call_llm", flaky_call_llm)
    facade = RouterChatCompletions("default")
    turn = [{"role": "user", "content": "hoi"}]
    resp = facade.create(messages=turn)
    # gemma fails → qwen (fallback #1) serves.
    assert acting_calls == ["google/gemma-4-e4b", "qwen/qwen3-4b-thinking-2507"]
    assert "qwen" in resp.choices[0].message.content
    assert facade.last_aggregator_slot["model"] == "qwen/qwen3-4b-thinking-2507"

    # Iteration 2 of the same turn: gemma is remembered as failed → the
    # walk starts directly at qwen (fallback stickiness).
    acting_calls.clear()
    grown = turn + [{"role": "assistant", "content": "partial"}]
    facade.create(messages=grown)
    assert acting_calls == ["qwen/qwen3-4b-thinking-2507"]


def test_chain_exhausted_reraises(patched, monkeypatch):
    def all_dead(*, task, messages, **kwargs):
        if task == "router_classifier":
            return _response("simple")
        raise ConnectionError(f"{kwargs.get('model')} down")

    monkeypatch.setattr(router_loop, "call_llm", all_dead)
    facade = RouterChatCompletions("default")
    with pytest.raises(ConnectionError):
        facade.create(messages=[{"role": "user", "content": "hoi"}])


def test_mid_stream_death_marks_candidate_failed(patched, monkeypatch):
    """A provider dying MID-stream (after create() returned the raw stream)
    must mark that candidate failed, so the consumer's stale-stream retry
    re-enters create() on the NEXT candidate instead of the dead one."""

    def stream_then_die():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel"))])
        raise ConnectionError("provider died mid-stream")

    def streaming_call_llm(*, task, messages, **kwargs):
        if task == "router_classifier":
            return _response("simple")
        if kwargs.get("model") == "google/gemma-4-e4b":
            return stream_then_die()
        return _response(f"acted by {kwargs.get('model')}")

    monkeypatch.setattr(router_loop, "call_llm", streaming_call_llm)
    facade = RouterChatCompletions("default")
    turn = [{"role": "user", "content": "hoi"}]

    stream = facade.create(messages=turn, stream=True)
    with pytest.raises(ConnectionError):
        list(stream)  # consumer iteration — the death happens HERE

    # Stale-stream retry re-enters create() with the same turn: the dead
    # candidate is skipped and the walk starts at the first fallback.
    facade.create(messages=turn, stream=True)
    assert facade.last_aggregator_slot["model"] == "qwen/qwen3-4b-thinking-2507"


def test_passthrough_kwargs_folded_into_extra_body(patched):
    """tool_choice / response_format / service_tier etc. aren't in call_llm's
    signature — they must ride along via extra_body, with an explicit caller
    extra_body key winning over a passthrough."""
    fake, calls = patched
    facade = RouterChatCompletions("default")
    facade.create(
        messages=[{"role": "user", "content": "hoi"}],
        tool_choice="auto",
        response_format={"type": "json_object"},
        service_tier="fast",
        extra_body={"service_tier": "explicit-wins"},
    )
    acting = [c for c in calls if c["task"] == "router_acting"][0]
    assert acting["extra_body"]["tool_choice"] == "auto"
    assert acting["extra_body"]["response_format"] == {"type": "json_object"}
    assert acting["extra_body"]["service_tier"] == "explicit-wins"


def test_stream_kwargs_forwarded(patched):
    fake, calls = patched
    facade = RouterChatCompletions("default")
    facade.create(
        messages=[{"role": "user", "content": "hoi"}],
        stream=True,
        timeout=42,
    )
    acting = [c for c in calls if c["task"] == "router_acting"][0]
    assert acting["stream"] is True
    assert acting["stream_options"] == {"include_usage": True}
    assert acting["timeout"] == 42
    # The classifier is never streamed.
    classifier = [c for c in calls if c["task"] == "router_classifier"][0]
    assert "stream" not in classifier


def test_decision_events_emitted_once_per_turn(patched):
    fake, calls = patched
    events = []
    facade = RouterChatCompletions(
        "default", decision_callback=lambda event, **kw: events.append((event, kw))
    )
    turn = [{"role": "user", "content": "hoi"}]
    facade.create(messages=turn)
    facade.create(messages=turn + [{"role": "assistant", "content": "x"}])
    decisions = [e for e in events if e[0] == "router.decision"]
    assert len(decisions) == 1
    assert decisions[0][1]["tier"] == "simple"
    assert "lmstudio" in decisions[0][1]["label"]


def test_accounting_protocol_shape(patched):
    fake, calls = patched
    client = RouterClient("default")
    client.chat.completions.create(messages=[{"role": "user", "content": "hoi"}])
    usage, cost = client.consume_reference_usage()
    # Consumed once — a second read is zeroed.
    usage2, cost2 = client.consume_reference_usage()
    assert usage2.input_tokens == 0
    assert client.last_aggregator_slot["model"] == "google/gemma-4-e4b"
    # Trace flush with tracing off must be a silent no-op.
    client.consume_and_save_trace("session-1", aggregator_output_fallback="hi")


def test_disabled_preset_skips_classifier(patched, monkeypatch):
    fake, calls = patched
    disabled = dict(PRESET, enabled=False)

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"router": {"presets": {"default": disabled}}}
    )
    facade = RouterChatCompletions("default")
    resp = facade.create(messages=[{"role": "user", "content": "hoi"}])
    tasks = [c["task"] for c in calls]
    assert tasks == ["router_acting"]  # no classifier call
    assert "gemma" in resp.choices[0].message.content  # default_route


def test_channel_hint_in_classifier_prompt():
    msgs = _build_classifier_messages(
        [{"role": "user", "content": "hoi"}],
        platform="whatsapp",
        channel_hints={"whatsapp": "simple"},
        context_messages=4,
    )
    assert "Channel: whatsapp" in msgs[1]["content"]
    assert 'bias toward "simple"' in msgs[1]["content"]

    no_hint = _build_classifier_messages(
        [{"role": "user", "content": "hoi"}],
        platform="telegram",
        channel_hints={"whatsapp": "simple"},
        context_messages=4,
    )
    assert "Channel:" not in no_hint[1]["content"]
