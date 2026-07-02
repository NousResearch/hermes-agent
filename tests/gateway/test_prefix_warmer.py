"""Tests for the local-backend prompt prefix warmer.

Covers the three pieces:
- ``agent/prefix_warm_registry.py`` — capture rules (local-only, system-head
  only, chat-completions only), keying, and LRU cap.
- ``gateway/prefix_warmer.py`` — warm_once builds the minimal replay request
  from a snapshot and survives per-endpoint failures.
- ``gateway/config.py`` — PrefixWarmerConfig defaults, coercion, roundtrip,
  and the default-off gate.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from agent import prefix_warm_registry as registry
from gateway import prefix_warmer
from gateway.config import GatewayConfig, PrefixWarmerConfig


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear()
    registry.enable_capture()
    yield
    registry.disable_capture()
    registry.clear()


# warm_once config with the idle gate disarmed — most tests record a prefix
# and warm immediately, which the traffic-recency guard would otherwise skip.
def _warm_cfg(**overrides):
    return PrefixWarmerConfig(min_idle_seconds=0.0, **overrides)


def _agent(base_url="http://127.0.0.1:8001/v1", api_mode="chat_completions"):
    return SimpleNamespace(
        base_url=base_url, api_mode=api_mode, model="m", api_key="local"
    )


def _kwargs(system="SYSTEM PROMPT", model="agentworld-35b", tools=None):
    messages: List[Dict[str, Any]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": "hi"})
    kw: Dict[str, Any] = {"model": model, "messages": messages}
    if tools is not None:
        kw["tools"] = tools
    return kw


# ---------------------------------------------------------------- registry --

def test_records_local_system_headed_request():
    kw = _kwargs(tools=[{"function": {"name": "t"}}])
    out = registry.record_local_prefix(_agent(), kw)
    assert out is kw  # passthrough, same object
    snaps = registry.get_snapshots()
    assert len(snaps) == 1
    assert snaps[0]["system_content"] == "SYSTEM PROMPT"
    assert snaps[0]["model"] == "agentworld-35b"
    assert snaps[0]["tools"] == [{"function": {"name": "t"}}]


def test_skips_non_local_endpoint():
    registry.record_local_prefix(
        _agent(base_url="https://openrouter.ai/api/v1"), _kwargs()
    )
    assert registry.get_snapshots() == []


def test_skips_non_chat_completions_api_mode():
    registry.record_local_prefix(_agent(api_mode="anthropic_messages"), _kwargs())
    assert registry.get_snapshots() == []


def test_skips_request_without_system_head():
    registry.record_local_prefix(_agent(), _kwargs(system=None))
    assert registry.get_snapshots() == []


def test_same_prefix_rerecord_updates_not_duplicates():
    registry.record_local_prefix(_agent(), _kwargs(tools=[{"function": {"name": "a"}}]))
    registry.record_local_prefix(_agent(), _kwargs(tools=[{"function": {"name": "b"}}]))
    snaps = registry.get_snapshots()
    assert len(snaps) == 1  # same (endpoint, model, system) -> one entry
    assert snaps[0]["tools"] == [{"function": {"name": "b"}}]  # newest wins


def test_entry_cap_evicts_oldest():
    for i in range(6):
        registry.record_local_prefix(
            _agent(base_url=f"http://127.0.0.1:{8000 + i}/v1"), _kwargs()
        )
    snaps = registry.get_snapshots()
    assert len(snaps) == registry._MAX_ENTRIES
    urls = {s["base_url"] for s in snaps}
    assert "http://127.0.0.1:8000/v1" not in urls  # oldest evicted
    assert "http://127.0.0.1:8005/v1" in urls


def test_never_raises_on_garbage(monkeypatch):
    # A hostile agent object must not break the request path.
    out = registry.record_local_prefix(object(), {"messages": None})
    assert out == {"messages": None}


def test_record_call_prefix_requires_tools():
    # call_llm path: tool-less auxiliary calls (compression, advisors) must
    # not churn the registry...
    registry.record_call_prefix("http://127.0.0.1:8001/v1", "local", _kwargs())
    assert registry.get_snapshots() == []
    # ...but tool-carrying calls (e.g. the MoA acting call) are captured.
    registry.record_call_prefix(
        "http://127.0.0.1:8001/v1", "local", _kwargs(tools=[{"function": {"name": "t"}}])
    )
    assert len(registry.get_snapshots()) == 1


def test_distinct_system_prompts_coexist_per_endpoint():
    tools = [{"function": {"name": "t"}}]
    registry.record_call_prefix("http://127.0.0.1:8001/v1", "local", _kwargs(system="MAIN", tools=tools))
    registry.record_call_prefix("http://127.0.0.1:8001/v1", "local", _kwargs(system="REVIEW", tools=tools))
    heads = {s["system_content"] for s in registry.get_snapshots()}
    assert heads == {"MAIN", "REVIEW"}


# --------------------------------------------------------------- warm_once --

def test_warm_once_replays_minimal_request(monkeypatch):
    tools = [{"function": {"name": "t"}}]
    registry.record_local_prefix(
        _agent(), _kwargs(tools=tools) | {"extra_body": {"chat_template_kwargs": {"x": 1}}}
    )
    sent = []

    def fake_send(base_url, api_key, config, kwargs):
        sent.append((base_url, api_key, kwargs))

    monkeypatch.setattr(prefix_warmer, "_send_warm_request", fake_send)
    assert prefix_warmer.warm_once(_warm_cfg()) == 1

    base_url, api_key, kwargs = sent[0]
    assert base_url == "http://127.0.0.1:8001/v1"
    assert kwargs["max_tokens"] == 1
    assert kwargs["messages"][0] == {"role": "system", "content": "SYSTEM PROMPT"}
    assert kwargs["messages"][1]["role"] == "user"
    assert kwargs["messages"][1]["content"] == prefix_warmer._WARM_USER_CONTENT
    assert kwargs["tools"] == tools
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"x": 1}}


def test_warm_once_continues_past_failures(monkeypatch):
    registry.record_local_prefix(_agent(base_url="http://127.0.0.1:8001/v1"), _kwargs())
    registry.record_local_prefix(_agent(base_url="http://127.0.0.1:8002/v1"), _kwargs())
    calls = []

    def flaky(base_url, api_key, config, kwargs):
        calls.append(base_url)
        if base_url.endswith("8002/v1"):
            raise ConnectionError("server restarting")

    monkeypatch.setattr(prefix_warmer, "_send_warm_request", flaky)
    # One endpoint fails, the other still warms; no exception escapes.
    assert prefix_warmer.warm_once(_warm_cfg()) == 1
    assert len(calls) == 2


def test_warm_once_no_snapshots_is_noop():
    assert prefix_warmer.warm_once(_warm_cfg()) == 0


# ------------------------------------------------------- capture enablement --

def test_capture_disabled_by_default_records_nothing():
    # Simulates a warmer-less process (plain CLI run): with capture off, the
    # hooks must be inert — no snapshots, no traffic timestamps.
    registry.disable_capture()
    kw = _kwargs(tools=[{"function": {"name": "t"}}])
    out = registry.record_local_prefix(_agent(), kw)
    assert out is kw
    registry.record_call_prefix("http://127.0.0.1:8001/v1", "local", kw)
    assert registry.get_snapshots() == []
    assert registry.last_traffic_at("http://127.0.0.1:8001/v1") == 0.0


# ---------------------------------------------------------- snapshot copies --

def test_snapshot_survives_in_place_mutation_of_tools_and_extra_body():
    # conversation_loop can sanitize the built kwargs in place after capture
    # (_force_ascii_payload); the snapshot must not alias those objects.
    tools = [{"function": {"name": "t", "description": "café"}}]
    extra_body = {"chat_template_kwargs": {"x": "café"}}
    registry.record_local_prefix(_agent(), _kwargs(tools=tools) | {"extra_body": extra_body})

    tools[0]["function"]["description"] = "cafe"  # in-place ASCII sanitize
    extra_body["chat_template_kwargs"]["x"] = "cafe"

    snap = registry.get_snapshots()[0]
    assert snap["tools"] is not tools
    assert snap["tools"][0]["function"]["description"] == "café"
    assert snap["extra_body"]["chat_template_kwargs"]["x"] == "café"


# ------------------------------------------------------------- idle gating --

def test_warm_once_skips_endpoint_with_recent_traffic(monkeypatch):
    # Recording a prefix marks the endpoint as actively trafficked, so an
    # immediate warm cycle must stand down (few-slot eviction guard).
    registry.record_local_prefix(_agent(), _kwargs())
    sent = []
    monkeypatch.setattr(
        prefix_warmer, "_send_warm_request", lambda *a: sent.append(a)
    )
    assert prefix_warmer.warm_once(PrefixWarmerConfig()) == 0  # default gate
    assert sent == []
    # Once the endpoint has been idle past the gate, warming resumes.
    assert prefix_warmer.warm_once(_warm_cfg()) == 1
    assert len(sent) == 1


# -------------------------------------------------- real-prefix round trip --

def test_warm_request_reproduces_real_build_kwargs_prefix(monkeypatch):
    """The core contract: the warm request's prefix fields are byte-identical
    to what the real chat-completions transport produced."""
    import json

    from agent.transports.chat_completions import ChatCompletionsTransport

    tools = [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file — supports UTF-8 content: café",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }]
    built = ChatCompletionsTransport().build_kwargs(
        model="agentworld-35b",
        messages=[
            {"role": "system", "content": "SYSTEM HEAD\n\nvolatile tail"},
            {"role": "user", "content": "hello"},
        ],
        tools=tools,
        extra_body_additions={"chat_template_kwargs": {"enable_thinking": False}},
    )
    out = registry.record_local_prefix(_agent(), built)
    assert out is built

    sent = []
    monkeypatch.setattr(
        prefix_warmer, "_send_warm_request",
        lambda base_url, api_key, config, kwargs: sent.append(kwargs),
    )
    assert prefix_warmer.warm_once(_warm_cfg()) == 1

    warm = sent[0]

    def dumps(obj):
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)

    # The prefix a llama.cpp chat template renders ahead of the user turn:
    # leading system message + tool schemas + template-affecting extra_body.
    assert dumps(warm["messages"][0]) == dumps(built["messages"][0])
    assert dumps(warm["tools"]) == dumps(built["tools"])
    assert dumps(warm["extra_body"]) == dumps(built["extra_body"])


# ------------------------------------------------------------ watcher loop --

def _run_watcher(monkeypatch, *, warm_side_effect, ticks, interval_seconds=240):
    """Drive prefix_warmer_watcher with instant sleeps; returns (delays, warms)."""
    import asyncio

    runner = SimpleNamespace(_running=True)
    delays: List[float] = []
    warms: List[int] = []

    def fake_warm(config):
        warms.append(1)
        if len(warms) >= ticks:
            runner._running = False
        return warm_side_effect()

    real_sleep = asyncio.sleep

    async def fast_sleep(delay):
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(prefix_warmer.asyncio, "sleep", fast_sleep)
    monkeypatch.setattr(prefix_warmer, "warm_once", fake_warm)
    cfg = PrefixWarmerConfig(interval_seconds=interval_seconds)
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(prefix_warmer.prefix_warmer_watcher(runner, cfg))
    finally:
        loop.close()
    return delays, warms


def test_watcher_settles_warms_on_interval_and_stops(monkeypatch):
    delays, warms = _run_watcher(
        monkeypatch, warm_side_effect=lambda: 1, ticks=2, interval_seconds=300
    )
    assert delays[0] == 60          # initial settle delay
    assert delays[1:] == [300, 300]  # one interval sleep per tick
    assert len(warms) == 2           # loop exited once _running went False


def test_watcher_confines_tick_exceptions(monkeypatch):
    def boom():
        raise RuntimeError("endpoint exploded")

    # Every tick raises; the watcher must keep ticking and exit cleanly.
    delays, warms = _run_watcher(monkeypatch, warm_side_effect=boom, ticks=3)
    assert len(warms) == 3


def test_watcher_enforces_minimum_interval(monkeypatch):
    delays, _ = _run_watcher(
        monkeypatch, warm_side_effect=lambda: 0, ticks=1, interval_seconds=5
    )
    assert delays[1:] == [30]  # clamped to the 30s floor


# ------------------------------------------------------------------ config --

def test_config_defaults_off():
    cfg = PrefixWarmerConfig.from_dict({})
    assert cfg.enabled is False
    assert cfg.interval_seconds == 240
    assert GatewayConfig.from_dict({}).prefix_warmer.enabled is False


def test_config_roundtrip_and_coercion():
    cfg = PrefixWarmerConfig.from_dict(
        {"enabled": "true", "interval_seconds": "300", "timeout_seconds": "60",
         "min_idle_seconds": "90"}
    )
    assert cfg.enabled is True
    assert cfg.interval_seconds == 300
    assert cfg.timeout_seconds == 60.0
    assert cfg.min_idle_seconds == 90.0
    assert PrefixWarmerConfig.from_dict(cfg.to_dict()) == cfg


def test_gateway_config_roundtrip_includes_prefix_warmer():
    gw = GatewayConfig.from_dict({"prefix_warmer": {"enabled": True}})
    assert gw.prefix_warmer.enabled is True
    assert gw.to_dict()["prefix_warmer"]["enabled"] is True
