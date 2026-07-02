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
    yield
    registry.clear()


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
    assert prefix_warmer.warm_once(PrefixWarmerConfig()) == 1

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
    assert prefix_warmer.warm_once(PrefixWarmerConfig()) == 1
    assert len(calls) == 2


def test_warm_once_no_snapshots_is_noop():
    assert prefix_warmer.warm_once(PrefixWarmerConfig()) == 0


# ------------------------------------------------------------------ config --

def test_config_defaults_off():
    cfg = PrefixWarmerConfig.from_dict({})
    assert cfg.enabled is False
    assert cfg.interval_seconds == 240
    assert GatewayConfig.from_dict({}).prefix_warmer.enabled is False


def test_config_roundtrip_and_coercion():
    cfg = PrefixWarmerConfig.from_dict(
        {"enabled": "true", "interval_seconds": "300", "timeout_seconds": "60"}
    )
    assert cfg.enabled is True
    assert cfg.interval_seconds == 300
    assert cfg.timeout_seconds == 60.0
    assert PrefixWarmerConfig.from_dict(cfg.to_dict()) == cfg


def test_gateway_config_roundtrip_includes_prefix_warmer():
    gw = GatewayConfig.from_dict({"prefix_warmer": {"enabled": True}})
    assert gw.prefix_warmer.enabled is True
    assert gw.to_dict()["prefix_warmer"]["enabled"] is True
